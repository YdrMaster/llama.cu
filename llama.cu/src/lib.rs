mod exec;
mod handle;
mod load;
mod memory;
mod model;
mod op;
mod utils;

use crate::{
    exec::{Command, Output, engine},
    model::{ChatTemplate, GGufModel, map_files},
    utils::meta,
};
use exec::Request;
use ggus::GGufMetaMapExt;
use log::info;
use nn::Tensor;
use operators::cuda::{self, Device};
use std::{
    collections::BTreeMap,
    ffi::c_int,
    path::Path,
    sync::{
        Arc, OnceLock,
        mpsc::{self, Receiver, Sender, TryRecvError},
    },
};
use tokeneer::{Bpe, Tokeneer, utok};

pub use exec::{DistKVCache, Session, SessionId};
pub use model::Message;
pub use tokeneer::TextBuf;

pub struct Service {
    handle: Option<(Receiver<Output>, std::thread::JoinHandle<()>)>,
    terminal: Terminal,
}

#[derive(Clone)]
pub struct Terminal {
    sender: Sender<Command>,
    cache_parts: Box<[(Device, usize)]>,
    components: Arc<OnceLock<ModelComponents>>,
}

pub enum ReturnReason {
    Finish,
    Overflow,
    NoDecode,
}

#[derive(Default)]
pub struct Received {
    pub sessions: Vec<(Session, ReturnReason)>,
    pub outputs: BTreeMap<SessionId, Vec<utok>>,
}

struct ModelComponents {
    tokenizer: Tokeneer<Bpe>,
    chat_template: Option<ChatTemplate>,
    cache_template: Tensor<usize, 2>,
    eos: utok,
}

impl Service {
    pub fn new(model: impl AsRef<Path>, gpus: &[c_int], use_cuda_grpah: bool) -> Self {
        info!("start inference @gpu{gpus:?}");
        // 创建调度通道
        let (outputs, receiver) = mpsc::channel();
        let (sender, commands) = mpsc::channel();
        // 从文件加载权重
        let maps = map_files(model);
        let gpus = gpus.to_vec();
        let gpus_ = gpus.clone();
        // 启动推理引擎
        assert!(cuda::init().is_ok());
        let once = Arc::new(OnceLock::new());
        let once_ = once.clone();
        let handle = std::thread::spawn(move || {
            let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
            gguf.insert_sin_cos();

            let tokenizer = Bpe::from_gguf(&gguf);
            let chat_template = gguf.chat_template(&tokenizer);
            let cache_template = gguf.kv_cache();
            let eos = meta![gguf => tokenizer_ggml_eos_token_id];

            once_.get_or_init(|| ModelComponents {
                tokenizer,
                chat_template,
                cache_template,
                eos,
            });
            drop(once_);

            let llama = gguf.llama();
            engine(llama, &gpus_, commands, outputs, use_cuda_grpah)
        });
        once.wait();
        assert!(matches!(receiver.recv().unwrap(), Output::Ready));
        info!("ready for inference");
        Self {
            handle: Some((receiver, handle)),
            terminal: Terminal {
                sender,
                cache_parts: gpus.iter().map(|&i| (Device::new(i), 1)).collect(),
                components: once,
            },
        }
    }

    pub const fn terminal(&self) -> &Terminal {
        &self.terminal
    }

    pub fn recv(&self) -> Received {
        let mut received = Received::default();
        match self.handle.as_ref().unwrap().0.recv() {
            Ok(output) => self.handle_output(output, &mut received),
            Err(_) => unreachable!(),
        }
        received
    }

    pub fn try_recv(&self) -> Received {
        let mut received = Received::default();
        loop {
            match self.handle.as_ref().unwrap().0.try_recv() {
                Ok(output) => self.handle_output(output, &mut received),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => unreachable!(),
            }
        }
        received
    }

    fn handle_output(&self, output: Output, received: &mut Received) {
        match output {
            Output::Overflow(sessions) => received
                .sessions
                .extend(sessions.into_iter().map(|s| (s, ReturnReason::Overflow))),
            Output::Removed(session) => received.sessions.push((session, ReturnReason::Finish)),
            Output::Complete {
                output,
                kv_pair,
                event,
                no_decode,
            } => {
                let device = self.terminal.cache_parts[0].0;
                let mut outputs = device
                    .retain_primary()
                    .apply(|ctx| exec::decode(output, kv_pair, event, &ctx.stream()));
                let components = self.terminal.components.wait();
                for (&id, toks) in &mut outputs {
                    if toks.contains(&components.eos) {
                        self.terminal.sender.send(Command::Remove(id)).unwrap()
                    }
                }
                received
                    .sessions
                    .extend(no_decode.into_iter().map(|s| (s, ReturnReason::NoDecode)));
                for (id, mut tokens) in outputs {
                    if let Some((len, _)) = tokens
                        .iter()
                        .enumerate()
                        .find(|(_, t)| **t == components.eos)
                    {
                        tokens.truncate(len)
                    }
                    received.outputs.entry(id).or_default().extend(tokens)
                }
            }
            Output::Ready => unreachable!(),
        }
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        let (receiver, handle) = self.handle.take().unwrap();
        let Terminal {
            sender,
            cache_parts,
            ..
        } = &self.terminal;
        sender.send(Command::ShutDown).unwrap();
        handle.join().unwrap();
        cache_parts[0]
            .0
            .retain_primary()
            .apply(|ctx| receiver.into_iter().for_each(|output| output.drop_on(ctx)))
    }
}

impl Terminal {
    pub fn new_cache(&self) -> DistKVCache {
        DistKVCache::new(&self.components.wait().cache_template, &self.cache_parts)
    }

    pub fn render(&self, msgs: &[Message]) -> String {
        self.components
            .wait()
            .chat_template
            .as_ref()
            .unwrap()
            .render(msgs, true)
            .unwrap()
    }

    pub fn tokenize<'s>(&self, text: &'s str) -> Vec<utok> {
        self.components.wait().tokenizer.encode(text)
    }

    pub fn start(&self, session: Session, tokens: &[utok]) -> bool {
        self.sender
            .send(Command::Insert(Request {
                session,
                prompt: tokens.to_vec().into(),
                out: 1,
            }))
            .is_ok()
    }

    pub fn stop(&self, id: SessionId) -> bool {
        self.sender.send(Command::Remove(id)).is_ok()
    }

    pub fn decode(&self, tokens: &[utok], buf: &mut TextBuf) -> String {
        self.components.wait().tokenizer.decode(tokens, buf)
    }
}
