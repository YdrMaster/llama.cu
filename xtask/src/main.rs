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

fn main() {
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
    /// 文本生成
    Generate(GenerateArgs),
    /// 命令行对话
    Chat(ChatArgs),
    /// web 服务
    Service(ServiceArgs),
}

#[derive(Args)]
pub struct BaseArgs {
    model: PathBuf,
    #[clap(long)]
    gpus: Option<String>,
    #[clap(long)]
    max_steps: Option<usize>,
}

impl BaseArgs {
    pub fn gpus(&self) -> Box<[c_int]> {
        self.gpus
            .as_ref()
            .map(|devices| {
                static NUM_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").unwrap());
                NUM_REGEX
                    .find_iter(devices)
                    .map(|c| c.as_str().parse().unwrap())
                    .collect()
            })
            .unwrap_or_else(|| vec![1].into())
    }

    pub fn max_steps(&self) -> usize {
        self.max_steps.unwrap_or(1000)
    }
}
