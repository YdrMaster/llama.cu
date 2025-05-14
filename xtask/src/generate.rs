use regex::Regex;
use std::{path::PathBuf, sync::LazyLock};

#[derive(Args)]
pub struct GenerateArgs {
    model: PathBuf,
    #[clap(short, long)]
    prompt: Option<String>,
    #[clap(long)]
    max_steps: Option<usize>,
    #[clap(long)]
    gpus: Option<String>,
}

impl GenerateArgs {
    pub fn generate(self) {
        let Self {
            model,
            prompt,
            max_steps,
            gpus,
        } = self;
        let prompt = prompt.as_deref().unwrap_or("Once upon a time,");
        let max_steps = max_steps.unwrap_or(1000);
        let gpus = gpus
            .map(|devices| {
                NUM_REGEX
                    .find_iter(&devices)
                    .map(|c| c.as_str().parse().unwrap())
                    .collect()
            })
            .unwrap_or_else(|| vec![1]);
        llama_cu::infer(model, &gpus, prompt, max_steps)
    }
}

static NUM_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").unwrap());
