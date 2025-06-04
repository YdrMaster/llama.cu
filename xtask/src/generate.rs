use crate::{BaseArgs, macros::print_now};
use llama_cu::{Message, Received, Service, Session, SessionId, TextBuf};
use log::info;
use std::time::{Duration, Instant};

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
        let mut prompt = prompt.unwrap_or("Once upon a time,".into());

        let service = Service::new(base.model, &gpus, !base.no_cuda_graph);
        let session = Session {
            id: SessionId(0),
            sample_args: Default::default(),
            cache: service.terminal().new_cache(),
        };
        print_now!("{prompt}");
        if use_template {
            prompt = service.terminal().render(&[Message::user(&prompt)])
        }
        service
            .terminal()
            .start(session, &service.terminal().tokenize(&prompt), max_steps);

        let mut prefill = Duration::ZERO;
        let mut decode = Duration::ZERO;
        let mut steps = 0;
        let mut buf = TextBuf::new();
        loop {
            let time = Instant::now();
            let Received { sessions, outputs } = service.recv(Duration::MAX);
            if prefill.is_zero() {
                prefill = time.elapsed()
            } else {
                decode += time.elapsed()
            }
            steps += 1;

            for (_, tokens) in outputs {
                print_now!("{}", service.terminal().decode(&tokens, &mut buf))
            }
            if !sessions.is_empty() {
                break;
            }
        }
        println!();
        info!("prefill = {prefill:?}, decode = {decode:?}");
        let time = decode / steps as _;
        info!(
            "steps = {steps}, perf: {time:?}/tok, {}tok/s",
            Duration::from_secs(1).div_duration_f32(time),
        )
    }
}
