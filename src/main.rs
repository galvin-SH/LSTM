use anyhow::{Context, Result};
// Add VarMap to imports for variable management
use candle_core::{DType, Device, Error, IndexOp, Module, ModuleT, Tensor, Var};
use candle_nn::{lstm, loss, ops, optim, Embedding, Linear, LSTM, VarBuilder, Optimizer, VarMap};
use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet};
use rand::prelude::IndexedRandom;
// --- Hyperparameters and Constants ---

const SEQ_LEN: usize = 100;
const HIDDEN_SIZE: usize = 256;
const NUM_LAYERS: usize = 2;
const LEARNING_RATE: f64 = 0.001;
const NUM_EPOCHS: usize = 10;
const BATCH_SIZE: usize = 32;

// --- Model Definition ---

/// Configuration for the Char-RNN model.
#[derive(Debug, Clone, Copy)]
struct Config {
    vocab_size: usize,
    embedding_size: usize,
    hidden_size: usize,
    num_layers: usize,
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
        let lstm_cfg = candle_nn::rnn::LSTMConfig {
            num_layers: config.num_layers,
            // Candle's LSTM expects batch_first=true by default for simplicity.
            ..Default::default()
        };
        let lstm = lstm(
            config.embedding_size,
            config.hidden_size,
            lstm_cfg,
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
    fn forward(&self, xs: &Tensor, state: Option<LSTMState>) -> Result<(Tensor, LSTMState)> {
        // 1. Get embeddings: (BATCH_SIZE, SEQ_LEN) -> (BATCH_SIZE, SEQ_LEN, EMBEDDING_SIZE)
        let embedded = self.embedding.forward(xs)?;

        // 2. Pass through LSTM. Output is (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        // Use forward_t as it handles training/inference modes if specified in LSTMConfig (not needed here)
        let (lstm_out, new_state) = self.lstm.forward_t(&embedded, state)?;

        // 3. Reshape for linear layer: (BATCH_SIZE * SEQ_LEN, HIDDEN_SIZE)
        let batch_size = xs.dim(0)?;
        let seq_len = xs.dim(1)?;
        let reshaped_out = lstm_out.reshape((batch_size * seq_len, self.config.hidden_size))?;

        // 4. Final linear layer: (BATCH_SIZE * SEQ_LEN, VOCAB_SIZE)
        let logits = self.output_linear.forward(&reshaped_out)?;

        Ok((logits, new_state))
    }
}

// Custom type alias for LSTM state (Hidden and Cell state)
type LSTMState = (Tensor, Tensor);

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
    vocab_size: usize,
    device: &Device,
    // CRITICAL FIX: The optimizer needs the VarMap to get all trainable variables.
    varmap: &VarMap,
) -> Result<()> {
    println!("--- Starting Training ---");
    let total_len = data.dim(0)?;
    let num_batches = (total_len - SEQ_LEN) / BATCH_SIZE;

    // Use AdamW for optimization (a common choice for RNNs/Transformers)
    let params = optim::AdamW::default();

    // CRITICAL FIX: Get all variables from the VarMap for the optimizer.
    let mut optimizer = optim::AdamW::new(varmap.all_vars(), params)?;

    for epoch in 1..=NUM_EPOCHS {
        let mut sum_loss = 0.0;
        let mut initial_lstm_state: Option<LSTMState> = None;

        for batch_idx in 0..num_batches {
            let start = batch_idx * BATCH_SIZE;

            // Calculate indices for the batch slice
            let start_slice = start * SEQ_LEN;
            let end_slice = start_slice + BATCH_SIZE * SEQ_LEN;

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
            // Note: If you want continuous training across batches, uncomment the line below:
            // initial_lstm_state = Some(next_state);

            // 6. Loss calculation (Softmax + NLL)
            let loss = loss::cross_entropy(&logits, &flat_targets)?;

            // 7. Backpropagation and Optimizer step
            optimizer.backward_step(&loss)?;
            sum_loss += loss.to_vec0::<f32>()? as f64;
        }

        let avg_loss = sum_loss / num_batches as f64;
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
        let mut rng = rand::thread_rng();
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
    // We use the CPU device for this example, as it is always available.
    let device = Device::Cpu;

    // A small sample text for training
    let raw_text = "The only thing necessary for the triumph of evil is for good men to do nothing. All that glitters is not gold. A journey of a thousand miles begins with a single step. To be or not to be, that is the question.";

    // 1. Prepare Data and Vocabulary
    let (data_tensor, char_to_idx, idx_to_char) =
        prepare_data(raw_text.to_lowercase().as_str(), &device)?;

    let config = Config {
        vocab_size: idx_to_char.len(),
        embedding_size: 128,
        hidden_size: HIDDEN_SIZE,
        num_layers: NUM_LAYERS,
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