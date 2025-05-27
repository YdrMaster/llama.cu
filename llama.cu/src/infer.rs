use crate::{
    Task,
    exec::{Command, DistKVCache, Output, Request, Session, SessionId, engine},
    model::{GGufModel, Message, map_files},
    utils::meta,
};
use ggus::GGufMetaMapExt;
use log::{info, warn};
use operators::cuda::{self, ContextSpore, Device};
use std::{
    collections::BTreeMap,
    ffi::c_int,
    path::Path,
    sync::mpsc::{self, Receiver, Sender},
    time::{Duration, Instant},
};
use tokeneer::{Bpe, utok};

pub fn infer(
    model: impl AsRef<Path>,
    gpus: &[c_int],
    max_steps: usize,
    requests: Receiver<Task>,
    session: Sender<Receiver<String>>,
    use_cuda_grpah: bool,
) -> (Duration, usize) {
    // 从文件加载权重
    let maps = map_files(model);
    let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
    gguf.insert_sin_cos();
    let llama = gguf.llama();
    // 创建调度通道
    let (outputs, receiver) = mpsc::channel();
    let (sender, commands) = mpsc::channel();
    // 启动推理引擎
    assert!(cuda::init().is_ok());
    std::thread::scope(|s| {
        let _worker = s.spawn(move || engine(llama, gpus, commands, outputs, use_cuda_grpah));
        info!("start inference @gpu{gpus:?}");
        let devices = gpus.iter().cloned().map(Device::new).collect::<Vec<_>>();
        service(
            requests, session, &devices, sender, receiver, &gguf, max_steps,
        )
    })
}

fn service(
    requests: Receiver<Task>,
    session: Sender<Receiver<String>>,
    devices: &[Device],
    sender: Sender<Command>,
    receiver: Receiver<Output>,
    gguf: &GGufModel,
    max_steps: usize,
) -> (Duration, usize) {
    let tokeneer = Bpe::from_gguf(gguf);
    let chat_template = gguf.chat_template(&tokeneer);
    let cache_template = gguf.kv_cache();
    let eos = meta![gguf => tokenizer_ggml_eos_token_id];
    let mut duration = Duration::ZERO;
    let mut n_decode = 1;

    let mut cache = Some(DistKVCache::new(
        &cache_template,
        &devices.iter().map(|dev| (*dev, 1)).collect::<Vec<_>>(),
    ));

    macro_rules! send {
        ($prompt:expr) => {
            sender
                .send(Command::Insert(Request {
                    session: Session {
                        id: SessionId(0),
                        sample_args: Default::default(),
                        cache: cache.take().unwrap(),
                    },
                    prompt: $prompt.into(),
                    out: 1,
                }))
                .unwrap()
        };
    }

    macro_rules! receive {
        () => {{
            let output = receiver.recv().unwrap();
            let (output, no_decode) = process(devices[0], output);
            assert!(no_decode.is_empty());
            output[&SessionId(0)][0]
        }};
    }

    for Task {
        mut prompt,
        use_template,
    } in requests
    {
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
        // 回复接收通道
        let (response, busy_session) = mpsc::channel();
        session.send(busy_session).unwrap();
        // 发送提示词
        let prompt_tokens = tokeneer.encode(&prompt);
        send!(prompt_tokens);
        // 首轮不计时
        let next = receive!();
        response.send(tokeneer.decode(&[next])).unwrap();
        // 循环接收输出
        let mut time = Instant::now();
        for _ in 0..max_steps - 1 {
            let next = receive!();
            duration += time.elapsed();
            // 收到休止符
            if next == eos {
                break;
            }
            time = Instant::now();
            n_decode += 1;
            // 会话拒绝输出
            if response.send(tokeneer.decode(&[next])).is_err() {
                break;
            }
        }
    }

    drop(sender);
    devices[0].retain_primary().apply(|ctx| {
        for output in receiver {
            match output {
                Output::Complete { kv_pair, event, .. } => {
                    drop((kv_pair.sprout(ctx), event.sprout(ctx)))
                }
                Output::Overflow(_) => todo!("overflow"),
                Output::Removed(_) => warn!("TODO: removed"),
            }
        }
    });
    (duration, n_decode)
}

fn process(device: Device, output: Output) -> (BTreeMap<SessionId, Box<[utok]>>, Box<[Session]>) {
    match output {
        Output::Complete {
            output,
            kv_pair,
            event,
            no_decode,
        } => {
            let output = device
                .retain_primary()
                .apply(|ctx| crate::exec::decode(output, kv_pair, event, &ctx.stream()));
            (output, no_decode)
        }
        Output::Overflow(_) => todo!("overflow"),
        Output::Removed(_) => todo!("removed"),
    }
}
