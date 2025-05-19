use crate::{BaseArgs, macros::print_now};
use llama_cu::Session;

#[derive(Args)]
pub struct GenerateArgs {
    #[clap(flatten)]
    base: BaseArgs,
    #[clap(short, long)]
    prompt: Option<String>,
    #[clap(short = 't', long)]
    use_template: bool,
}

impl GenerateArgs {
    pub fn generate(self) {
        let Self {
            base,
            prompt,
            use_template,
        } = self;
        let gpus = base.gpus();
        let max_steps = base.max_steps();
        let prompt = prompt.unwrap_or("Once upon a time,".into());
        let (mut session, _handle) = Session::new(base.model, gpus, max_steps, !base.no_cuda_graph);
        let busy = session.send(prompt.clone(), use_template);
        let first = busy.receive().unwrap();
        print_now!("{prompt}{first}");
        while let Some(text) = busy.receive() {
            print_now!("{text}")
        }
        println!();
        drop(session)
    }
}
