fn main() {
    println!("Available CPU cores: {}", num_cpus::get());
    println!("Rayon default threads: {}", rayon::current_num_threads());
}
