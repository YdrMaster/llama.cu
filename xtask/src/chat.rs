use crate::BaseArgs;
use llama_cu::Session;

#[derive(Args)]
pub struct ChatArgs {
    #[clap(flatten)]
    base: BaseArgs,
}

macro_rules! print_now {
    ($($arg:tt)*) => {{
        use std::io::Write;

        print!($($arg)*);
        std::io::stdout().flush().unwrap();
    }};
}

impl ChatArgs {
    pub fn dialog(self) {
        let Self { base } = self;
        let gpus = base.gpus();
        let max_steps = base.max_steps();
        let (mut session, _handle) = Session::new(base.model, gpus, max_steps);

        let mut line = String::new();
        loop {
            line.clear();
            while line.is_empty() {
                print_now!("user> ");
                std::io::stdin().read_line(&mut line).unwrap();
                assert_eq!(line.pop(), Some('\n'));
            }

            let busy = session.send(line.clone(), true);
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
