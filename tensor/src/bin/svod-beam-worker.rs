fn main() {
    if let Err(error) = svod_tensor::beam_worker::worker_main() {
        eprintln!("svod-beam-worker: {error}");
        std::process::exit(1);
    }
}
