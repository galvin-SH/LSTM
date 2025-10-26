use anyhow::{Context, Result};
// Add VarMap to imports for variable management
use candle_core::{DType, Device, Error, Module, Tensor};
use candle_nn::{lstm, loss, ops, optim, Embedding, Linear, LSTM, VarBuilder, Optimizer, VarMap, RNN};
use candle_nn::rnn::LSTMState;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::{Arc, Mutex};
use rand::prelude::IndexedRandom;
use rayon::prelude::*;
// --- Hyperparameters and Constants ---

const SEQ_LEN: usize = 20;
const HIDDEN_SIZE: usize = 128;  // Reduced from 256 for faster training
const LEARNING_RATE: f64 = 0.001;
const NUM_EPOCHS: usize = 2;     // Reduced from 10 for quick test
const BATCH_SIZE: usize = 16;    // Larger batch size for better parallelism
const GRADIENT_ACCUMULATION_STEPS: usize = 8;  // Accumulate gradients over N batches before updating
const MAX_TRAINING_CHARS: usize = 100_000;  // Limit training data for quick test

// --- Model Definition ---

/// Configuration for the Char-RNN model.
#[derive(Debug, Clone, Copy)]
struct Config {
    vocab_size: usize,
    embedding_size: usize,
    hidden_size: usize,
}

/// The Character-level RNN model using LSTM.
struct LstmModel {
    embedding: Embedding,
    lstm: LSTM,
    output_linear: Linear,
    config: Config,
}

impl LstmModel {
    /// Constructs a new LSTM model.
    fn new(vs: VarBuilder, config: Config) -> Result<Self> {
        // 1. Embedding layer: Maps character indices to dense vectors.
        let embedding = candle_nn::embedding(
            config.vocab_size,
            config.embedding_size,
            vs.pp("embedding"),
        )?;

        // 2. LSTM layer: The core recurrent unit.
        // Note: candle-nn's LSTM creates a single layer. For multi-layer support,
        // you would need to stack multiple LSTM instances manually.
        let lstm = lstm(
            config.embedding_size,
            config.hidden_size,
            Default::default(), // Use default LSTM configuration
            vs.pp("lstm"),
        )?;

        // 3. Output layer: Maps the LSTM's hidden state back to the vocabulary size.
        let output_linear = candle_nn::linear(
            config.hidden_size,
            config.vocab_size,
            vs.pp("output_linear"),
        )?;

        Ok(Self {
            embedding,
            lstm,
            output_linear,
            config,
        })
    }

    /// Performs the forward pass.
    /// The input tensor has shape (BATCH_SIZE, SEQ_LEN).
    /// The output is the logits tensor of shape (BATCH_SIZE * SEQ_LEN, VOCAB_SIZE).
    fn forward(&self, xs: &Tensor, _state: Option<LSTMState>) -> Result<(Tensor, LSTMState)> {
        // 1. Get embeddings: (BATCH_SIZE, SEQ_LEN) -> (BATCH_SIZE, SEQ_LEN, EMBEDDING_SIZE)
        let embedded = self.embedding.forward(xs)?;

        // 2. Pass through LSTM. The seq() method returns Vec<(hidden, cell)> for each timestep
        let states = self.lstm.seq(&embedded)?;

        // Extract hidden states from each timestep and stack them
        // LSTMState has h() method to get hidden state
        let hidden_states: Vec<Tensor> = states.iter().map(|state| state.h().clone()).collect();

        // Stack hidden states: Vec<Tensor> -> (SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE)
        let lstm_out = Tensor::stack(&hidden_states, 0)?;

        // Transpose to (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        let lstm_out = lstm_out.transpose(0, 1)?;

        // Get the final state (last element in states vec)
        let final_state = states.last()
            .ok_or_else(|| Error::Msg("No LSTM states returned".to_string()))?
            .clone();

        // 3. Reshape for linear layer: (BATCH_SIZE * SEQ_LEN, HIDDEN_SIZE)
        let batch_size = xs.dim(0)?;
        let seq_len = xs.dim(1)?;
        let reshaped_out = lstm_out.reshape((batch_size * seq_len, self.config.hidden_size))?;

        // 4. Final linear layer: (BATCH_SIZE * SEQ_LEN, VOCAB_SIZE)
        let logits = self.output_linear.forward(&reshaped_out)?;

        Ok((logits, final_state))
    }
}

// --- Data & Vocabulary Processing ---

/// Prepares the data: creates vocabulary and transforms text into a tensor of indices.
fn prepare_data(text: &str, device: &Device) -> Result<(Tensor, HashMap<char, usize>, Vec<char>)> {
    // 1. Build the vocabulary (unique characters)
    let vocab: HashSet<char> = text.chars().collect();
    let mut sorted_vocab: Vec<char> = vocab.into_iter().collect();
    sorted_vocab.sort_unstable(); // For consistent indexing

    let vocab_size = sorted_vocab.len();
    let char_to_idx: HashMap<char, usize> = sorted_vocab
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    println!("Vocabulary size: {}", vocab_size);

    // 2. Convert the entire text into a sequence of indices (u32 for indices)
    let indices: Vec<u32> = text
        .chars()
        .map(|c| *char_to_idx.get(&c).unwrap() as u32)
        .collect();

    // 3. Convert indices to a Tensor (Shape: N). DType::U32 is suitable for indices.
    let tensor_data = Tensor::from_vec(indices, text.len(), device)?.to_dtype(DType::U32)?;

    Ok((tensor_data, char_to_idx, sorted_vocab))
}

// --- Training Loop and Sampler ---

fn train(
    model: &LstmModel,
    data: &Tensor,
    _vocab_size: usize,
    _device: &Device,
    // CRITICAL FIX: The optimizer needs the VarMap to get all trainable variables.
    varmap: &VarMap,
) -> Result<()> {
    println!("--- Starting Training (with Gradient Accumulation & Parallel Processing) ---");
    println!("Using {} threads for parallel processing", rayon::current_num_threads());
    println!("Gradient accumulation steps: {}", GRADIENT_ACCUMULATION_STEPS);
    let total_len = data.dim(0)?;
    let num_batches = (total_len - 1) / (BATCH_SIZE * SEQ_LEN);

    // Use AdamW for optimization (a common choice for RNNs/Transformers)
    // Get all variables from the VarMap for the optimizer.
    // Wrap optimizer in Arc<Mutex<>> for thread-safe access
    let optimizer = Arc::new(Mutex::new(optim::AdamW::new_lr(varmap.all_vars(), LEARNING_RATE)?));

    for epoch in 1..=NUM_EPOCHS {
        let initial_lstm_state: Option<LSTMState> = None;
        let mut epoch_loss = 0.0;

        // Process batches in chunks for gradient accumulation
        // This reduces mutex contention by grouping optimizer updates
        for chunk_start in (0..num_batches).step_by(GRADIENT_ACCUMULATION_STEPS) {
            let chunk_end = (chunk_start + GRADIENT_ACCUMULATION_STEPS).min(num_batches);
            let chunk_indices: Vec<usize> = (chunk_start..chunk_end).collect();
            let actual_accumulation_steps = chunk_indices.len();

            // Process this chunk of batches in parallel (forward passes and loss computation)
            let batch_data: Vec<Result<(f64, Tensor, Tensor)>> = chunk_indices
                .par_iter()
                .map(|&batch_idx| {
                    // Calculate indices for the batch slice
                    let start_slice = batch_idx * BATCH_SIZE * SEQ_LEN;

                    // To correctly handle sequence and batch dimensions:
                    // 1. Slice the entire data tensor to get the segment for this batch.
                    let chunk = data.narrow(0, start_slice, BATCH_SIZE * SEQ_LEN + 1)?;

                    // 2. Separate into input and target segments
                    let x_chunk = chunk.narrow(0, 0, BATCH_SIZE * SEQ_LEN)?;
                    let y_chunk = chunk.narrow(0, 1, BATCH_SIZE * SEQ_LEN)?;

                    // 3. Reshape into (BATCH_SIZE, SEQ_LEN) for input
                    let x_batch = x_chunk.reshape((BATCH_SIZE, SEQ_LEN))?;

                    // 4. Targets are flattened for cross_entropy loss
                    // (BATCH_SIZE * SEQ_LEN)
                    let flat_targets = y_chunk.reshape(x_chunk.shape())?;

                    // 5. Forward Pass
                    // The state is reset for each batch in this simplified setup.
                    let (logits, _next_state) = model.forward(&x_batch, initial_lstm_state.clone())?;

                    // 6. Loss calculation (Softmax + NLL)
                    let loss = loss::cross_entropy(&logits, &flat_targets)?;
                    let loss_value = loss.to_vec0::<f32>()? as f64;

                    // Return loss value and scaled loss tensor for accumulation
                    Ok((loss_value, loss, flat_targets))
                })
                .collect();

            // Sum the losses for averaging and perform single backward+optimization step
            let mut accumulated_loss_value = 0.0;
            let mut loss_tensors = Vec::new();

            for result in batch_data {
                let (loss_val, loss_tensor, _) = result?;
                accumulated_loss_value += loss_val;
                loss_tensors.push(loss_tensor);
            }

            epoch_loss += accumulated_loss_value;

            // Average the losses across the accumulation window
            if !loss_tensors.is_empty() {
                let mut avg_loss = loss_tensors[0].clone();
                for loss_tensor in &loss_tensors[1..] {
                    avg_loss = (&avg_loss + loss_tensor)?;
                }
                avg_loss = (&avg_loss / actual_accumulation_steps as f64)?;

                // Single backward+optimizer step for the chunk
                // This is the only mutex lock per chunk (instead of per batch)
                {
                    let mut opt = optimizer.lock().unwrap();
                    opt.backward_step(&avg_loss)?;
                }
            }
        }

        let avg_loss = epoch_loss / num_batches as f64;
        println!(
            "Epoch {}/{}: Average Loss = {:.4}",
            epoch, NUM_EPOCHS, avg_loss
        );
    }
    Ok(())
}

/// Generates new text using the trained model.
fn sample(
    model: &LstmModel,
    start_text: &str,
    char_to_idx: &HashMap<char, usize>,
    idx_to_char: &[char],
    device: &Device,
    length: usize,
    temperature: f64,
) -> Result<String> {
    println!("\n--- Sampling Text ---");
    println!("Start Prompt: \"{}\"", start_text);

    let mut generated_text = start_text.to_string();
    let mut current_state: Option<LSTMState> = None;

    // 1. Prime the model with the start text
    for char in start_text.chars() {
        let idx = *char_to_idx
            .get(&char)
            .context(format!("Character '{}' not in vocabulary.", char))?;

        // Input shape: (1, 1) - Batch size 1, Sequence length 1
        let input_tensor = Tensor::new(&[idx as u32], device)?.to_dtype(DType::U32)?;
        let input_tensor = input_tensor.reshape((1, 1))?;

        // Forward pass to update the state, we ignore the logits
        let (_, next_state) = model.forward(&input_tensor, current_state)?;
        current_state = Some(next_state);
    }

    // 2. Generate new characters
    let mut last_char_idx = *char_to_idx
        .get(&start_text.chars().last().unwrap())
        .unwrap() as u32;

    for _ in 0..length {
        // Input shape: (1, 1)
        let input_tensor = Tensor::new(&[last_char_idx], device)?.to_dtype(DType::U32)?;
        let input_tensor = input_tensor.reshape((1, 1))?;

        let (logits, next_state) = model.forward(&input_tensor, current_state)?;
        current_state = Some(next_state);

        // Reshape logits: (1, VOCAB_SIZE)
        let logits = logits.reshape((1, model.config.vocab_size))?;

        // Apply temperature and softmax for sampling
        let logits = (&logits / temperature)?;
        let probabilities = ops::softmax(&logits, 1)?;

        // Move to CPU for sampling (this is already on CPU due to Device::Cpu)
        let probabilities_vec = probabilities.to_vec2::<f32>()?[0].clone();

        // Sample the next character index
        let mut rng = rand::rng();
        let next_idx = probabilities_vec
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as u32, p as f64))
            .collect::<Vec<(u32, f64)>>()
            .choose_weighted(&mut rng, |item| item.1)
            .map(|(idx, _)| *idx)
            .map_err(|e| Error::Msg(format!("Sampling error: {}", e)))?;

        // Convert index back to character and append
        let next_char = idx_to_char[next_idx as usize];
        generated_text.push(next_char);
        last_char_idx = next_idx;
    }

    Ok(generated_text)
}

fn run() -> Result<()> {
    // Try to use CUDA if available, otherwise fall back to CPU
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);

    // Load training text from the data directory
    let raw_text = fs::read_to_string("data/input.txt")
        .context("Failed to read training data from data/input.txt")?;

    println!("Loaded {} characters from data/input.txt", raw_text.len());

    // Limit training data for faster test runs
    let training_text = if raw_text.len() > MAX_TRAINING_CHARS {
        println!("Limiting training data to {} characters for quick test", MAX_TRAINING_CHARS);
        &raw_text[..MAX_TRAINING_CHARS]
    } else {
        &raw_text
    };

    // 1. Prepare Data and Vocabulary
    let (data_tensor, char_to_idx, idx_to_char) =
        prepare_data(training_text.to_lowercase().as_str(), &device)?;

    let config = Config {
        vocab_size: idx_to_char.len(),
        embedding_size: 128,
        hidden_size: HIDDEN_SIZE,
    };

    // 2. Initialize Model and Optimizer
    // Use VarMap to hold and manage all trainable variables.
    let varmap = VarMap::new();
    // Use F32 for the model weights, which is the standard float type.
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = LstmModel::new(vb, config)?;

    // 3. Train the Model (passing varmap)
    train(&model, &data_tensor, config.vocab_size, &device, &varmap)?;

    // 4. Sample Text
    let generated_output = sample(
        &model,
        "the only thing",
        &char_to_idx,
        &idx_to_char,
        &device,
        200, // length of text to generate
        0.5, // temperature (lower = more conservative, higher = more creative)
    )?;

    println!("-----------------------------");
    println!("Generated Text:");
    println!("{}", generated_output);
    println!("-----------------------------");

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error running Char-RNN: {:?}", e);
    }
}