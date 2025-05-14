mod dialog;
mod generate;

use clap::Parser;
use dialog::DialogArgs;
use generate::GenerateArgs;

#[macro_use]
extern crate clap;

fn main() {
    use Commands::*;
    match Cli::parse().command {
        Generate(args) => args.generate(),
        Dialog(args) => args.dialog(),
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
    Dialog(DialogArgs),
}
