use llama_cu::Session;
use regex::Regex;
use std::{path::PathBuf, sync::LazyLock};

#[derive(Args)]
pub struct DialogArgs {
    model: PathBuf,
    #[clap(long)]
    max_steps: Option<usize>,
    #[clap(long)]
    gpus: Option<String>,
}

macro_rules! print_now {
    ($($arg:tt)*) => {{
        use std::io::Write;

        print!($($arg)*);
        std::io::stdout().flush().unwrap();
    }};
}

impl DialogArgs {
    pub fn dialog(self) {
        let Self {
            model,
            max_steps,
            gpus,
        } = self;
        let max_steps = max_steps.unwrap_or(1000);
        let gpus = gpus
            .map(|devices| {
                NUM_REGEX
                    .find_iter(&devices)
                    .map(|c| c.as_str().parse().unwrap())
                    .collect()
            })
            .unwrap_or_else(|| vec![1].into());
        let (mut session, _handle) = Session::new(model, gpus, max_steps);

        let mut line = String::new();
        loop {
            line.clear();
            while line.is_empty() {
                print_now!("user> ");
                std::io::stdin().read_line(&mut line).unwrap();
                assert_eq!(line.pop(), Some('\n'));
            }

            let busy = session.send(line.clone());
            let first = busy.receive().unwrap();
            print_now!("assistant> {first}");
            while let Some(text) = busy.receive() {
                print_now!("{text}")
            }
            println!();
            println!("=== over ===")
        }
    }
}

static NUM_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").unwrap());
