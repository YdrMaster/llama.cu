mod generate;

use clap::Parser;
use generate::GenerateArgs;

#[macro_use]
extern crate clap;

fn main() {
    use Commands::*;
    match Cli::parse().command {
        Generate(args) => args.generate(),
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
}
