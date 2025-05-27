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
use model::Message;
use nn::Tensor;
use operators::cuda::{self, Device};
use std::{
    collections::BTreeMap,
    ffi::c_int,
    path::Path,
    sync::{
        Arc, OnceLock,
        mpsc::{self, Receiver, Sender},
    },
};
use tokeneer::{Bpe, Tokeneer, utok};

pub use exec::{DistKVCache, Session, SessionId};

pub struct Service {
    handle: Option<(Receiver<Output>, std::thread::JoinHandle<()>)>,
    terminal: Terminal,
}

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

pub struct Received {
    pub sessions: Vec<(Session, ReturnReason)>,
    pub outputs: BTreeMap<SessionId, (Vec<utok>, Vec<u8>)>,
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
        match self.handle.as_ref().unwrap().0.recv() {
            Ok(Output::Overflow(sessions)) => Received {
                sessions: sessions
                    .into_iter()
                    .map(|s| (s, ReturnReason::Overflow))
                    .collect(),
                outputs: Default::default(),
            },
            Ok(Output::Removed(session)) => Received {
                sessions: vec![(session, ReturnReason::Finish)],
                outputs: Default::default(),
            },
            Ok(Output::Complete {
                output,
                kv_pair,
                event,
                no_decode,
            }) => {
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
                Received {
                    sessions: no_decode
                        .into_iter()
                        .map(|s| (s, ReturnReason::NoDecode))
                        .collect(),
                    outputs: outputs
                        .into_iter()
                        .map(|(id, mut tokens)| {
                            if let Some((len, _)) = tokens
                                .iter()
                                .enumerate()
                                .find(|(_, t)| **t == components.eos)
                            {
                                tokens.truncate(len)
                            }
                            let piece = components.tokenizer.decode(&tokens).into_bytes();
                            (id, (tokens, piece))
                        })
                        .collect(),
                }
            }
            Ok(Output::Ready) | Err(_) => unreachable!(),
        }
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        let (_, handle) = self.handle.take().unwrap();
        let _ = self.terminal.sender.send(Command::Notify);
        handle.join().unwrap()
    }
}

impl Terminal {
    pub fn new_cache(&self) -> DistKVCache {
        DistKVCache::new(&self.components.wait().cache_template, &self.cache_parts)
    }

    pub fn start(&self, session: Session, mut prompt: String, use_template: bool) {
        let ModelComponents {
            tokenizer,
            chat_template,
            ..
        } = self.components.wait();

        if use_template {
            if let Some(chat_template) = &chat_template {
                prompt = chat_template
                    .render(
                        &[Message {
                            role: "user",
                            content: &prompt,
                        }],
                        true,
                    )
                    .unwrap()
            }
        }

        self.sender
            .send(Command::Insert(Request {
                session,
                prompt: tokenizer.encode(&prompt).into(),
                out: 1,
            }))
            .unwrap()
    }
}
