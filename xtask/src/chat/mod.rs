mod app_session;
mod tui;

use crate::{BaseArgs, macros::print_now};
use llama_cu::{Received, Service, Session, SessionId, TextBuf};

#[derive(Args)]
pub struct ChatArgs {
    #[clap(flatten)]
    base: BaseArgs,
    #[clap(long)]
    tui: bool,
}

impl ChatArgs {
    pub fn chat(self) {
        let Self {
            base,
            tui: advanced,
        } = self;
        let gpus = base.gpus();
        let max_steps = base.max_steps();

        let service = Service::new(base.model, &gpus, !base.no_cuda_graph);
        if !advanced {
            simple(service, max_steps)
        } else {
            let terminal = ratatui::init();
            let result = tui::App::new(service).run(terminal);
            ratatui::restore();
            result.unwrap()
        }
    }
}

fn simple(service: Service, max_steps: usize) {
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

        let mut buf = TextBuf::new();
        print_now!("assistant> ");
        for _ in 0..max_steps {
            let Received { sessions, outputs } = service.recv();

            for (_, tokens) in outputs {
                print_now!("{}", service.terminal().decode(&tokens, &mut buf))
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
