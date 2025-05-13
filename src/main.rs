mod exec;
mod gguf;
mod handle;
mod infer;
mod loader;
mod memory;
mod model;
mod op;
mod utils;

fn main() {
    let mut args = std::env::args();
    let _ = args.next();
    let path = args.next().unwrap();
    let prompt = args.next().unwrap_or("Once upon a time,".into());
    infer::infer(path, &prompt)
}
