mod chat;
mod generate;
mod logger;

use clap::Parser;
use regex::Regex;
use std::{ffi::c_int, path::PathBuf, sync::LazyLock};

#[macro_use]
extern crate clap;

fn main() {
    logger::init();
    use Commands::*;
    match Cli::parse().command {
        Generate(args) => args.generate(),
        Chat(args) => args.chat(),
        // Service(args) => args.service(),
    }
}

#[derive(Parser)]
#[clap(name = "InfiniLM")]
#[clap(version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// text generation
    Generate(generate::GenerateArgs),
    /// chat in console
    Chat(chat::ChatArgs),
    // /// web service
    // Service(ServiceArgs),
}

#[derive(Args)]
struct BaseArgs {
    model: PathBuf,
    #[clap(long)]
    gpus: Option<String>,
    #[clap(long)]
    max_steps: Option<usize>,
    #[clap(long)]
    no_cuda_graph: bool,
}

impl BaseArgs {
    fn gpus(&self) -> Box<[c_int]> {
        self.gpus
            .as_ref()
            .map(|devices| {
                static NUM_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").unwrap());
                NUM_REGEX
                    .find_iter(devices)
                    .map(|c| c.as_str().parse().unwrap())
                    .collect()
            })
            .unwrap_or_else(|| [0].into())
    }

    fn max_steps(&self) -> usize {
        self.max_steps.unwrap_or(1000)
    }
}

mod macros {
    macro_rules! print_now {
        ($($arg:tt)*) => {{
            use std::io::Write;

            print!($($arg)*);
            std::io::stdout().flush().unwrap();
        }};
    }

    pub(crate) use print_now;
}
