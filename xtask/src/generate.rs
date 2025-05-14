use llama_cu::Session;
use regex::Regex;
use std::{path::PathBuf, sync::LazyLock};

#[derive(Args)]
pub struct GenerateArgs {
    model: PathBuf,
    #[clap(long)]
    gpus: Option<String>,
    #[clap(long)]
    max_steps: Option<usize>,
    #[clap(short, long)]
    prompt: Option<String>,
}

macro_rules! print_now {
    ($($arg:tt)*) => {{
        use std::io::Write;

        print!($($arg)*);
        std::io::stdout().flush().unwrap();
    }};
}

impl GenerateArgs {
    pub fn generate(self) {
        let Self {
            model,
            prompt,
            max_steps,
            gpus,
        } = self;
        let prompt = prompt.unwrap_or("Once upon a time,".into());
        let max_steps = max_steps.unwrap_or(1000);
        let gpus = gpus
            .map(|devices| {
                NUM_REGEX
                    .find_iter(&devices)
                    .map(|c| c.as_str().parse().unwrap())
                    .collect()
            })
            .unwrap_or_else(|| vec![1].into());
        let (mut session, handle) = Session::new(model, gpus, max_steps);
        let busy = session.send(prompt.clone());
        let first = busy.receive().unwrap();
        print_now!("{prompt}{first}");
        while let Some(text) = busy.receive() {
            print_now!("{text}")
        }
        drop(session);
        handle.join()
    }
}

static NUM_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").unwrap());
