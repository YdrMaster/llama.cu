use crate::BaseArgs;
use llama_cu::Session;

#[derive(Args)]
pub struct GenerateArgs {
    #[clap(flatten)]
    base: BaseArgs,
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
        let Self { base, prompt } = self;
        let gpus = base.gpus();
        let max_steps = base.max_steps();
        let prompt = prompt.unwrap_or("Once upon a time,".into());
        let (mut session, _handle) = Session::new(base.model, gpus, max_steps);
        let busy = session.send(prompt.clone());
        let first = busy.receive().unwrap();
        print_now!("{prompt}{first}");
        while let Some(text) = busy.receive() {
            print_now!("{text}")
        }
        drop(session);
    }
}
