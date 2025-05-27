use crate::{BaseArgs, macros::print_now};
use llama_cu::{Received, Service, Session, SessionId};

#[derive(Args)]
pub struct ChatArgs {
    #[clap(flatten)]
    base: BaseArgs,
}

impl ChatArgs {
    pub fn chat(self) {
        let Self { base } = self;
        let gpus = base.gpus();
        let max_steps = base.max_steps();

        let service = Service::new(base.model, &gpus, !base.no_cuda_graph);
        let mut session = Some(Session {
            id: SessionId(0),
            sample_args: Default::default(),
            cache: service.terminal().new_cache(),
        });

        let mut line = String::new();
        loop {
            line.clear();
            while line.is_empty() {
                print_now!("user> ");
                std::io::stdin().read_line(&mut line).unwrap();
                assert_eq!(line.pop(), Some('\n'));
            }

            service
                .terminal()
                .start(session.take().unwrap(), line.clone(), true);

            print_now!("assistant> ");
            for _ in 0..max_steps {
                let Received { sessions, outputs } = service.recv();

                for (_, (_, piece)) in outputs {
                    let str = unsafe { std::str::from_utf8_unchecked(&piece) };
                    print_now!("{str}");
                }
                if let Some((s, _)) = sessions.into_iter().next() {
                    session = Some(s);
                    break;
                }
            }
            println!();
            println!("=== over ===")
        }
    }
}
