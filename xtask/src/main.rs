mod chat;
mod generate;
mod service;

use chat::ChatArgs;
use clap::Parser;
use generate::GenerateArgs;
use regex::Regex;
use service::ServiceArgs;
use std::{ffi::c_int, path::PathBuf, sync::LazyLock};

#[macro_use]
extern crate clap;

/// <https://docs.rs/flexi_logger/0.30.1/flexi_logger/struct.LogSpecification.html>
const DEFAULT_LOG: &str = "error, llama_cu=trace";

/// <https://docs.rs/flexi_logger/0.30.1/flexi_logger/struct.Logger.html#method.set_palette>
const LOG_PALETTE: &str = "b9;178;34;5;0";

fn main() {
    flexi_logger::Logger::try_with_env_or_str(DEFAULT_LOG)
        .unwrap()
        .set_palette(LOG_PALETTE.into())
        .start()
        .unwrap();

    use Commands::*;
    match Cli::parse().command {
        Generate(args) => args.generate(),
        Chat(args) => args.dialog(),
        Service(args) => args.service(),
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
    Generate(GenerateArgs),
    /// chat in console
    Chat(ChatArgs),
    /// web service
    Service(ServiceArgs),
}

#[derive(Args)]
struct BaseArgs {
    model: PathBuf,
    #[clap(long)]
    gpus: Option<String>,
    #[clap(long)]
    max_steps: Option<usize>,
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
