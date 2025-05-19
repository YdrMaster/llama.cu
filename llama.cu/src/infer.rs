use crate::{
    Task,
    exec::{Command, KVCache, Output, Request, Session, SessionId},
    handle::Handle,
    memory::MemPages,
    model::{GGufModel, Message, map_files},
    utils::{destruct, meta},
};
use ggus::GGufMetaMapExt;
use log::info;
use nn::{Distribution, LLaMA, Tensor};
use operators::cuda::{self, ContextSpore, Device};
use std::{
    ffi::c_int,
    path::Path,
    sync::{
        Arc, Barrier,
        mpsc::{self, Receiver, Sender},
    },
    time::{Duration, Instant},
};
use tokeneer::{Bpe, utok};

#[cfg(nccl)]
use operators::nccl::{Communicator, CommunicatorGroup};

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
    info!("start inference @gpu{gpus:?}");

    assert!(cuda::init().is_ok());

    std::thread::scope(|s| {
        let devices = gpus.iter().cloned().map(Device::new).collect();
        let mut senders = Vec::new();
        let (outputs, receiver) = mpsc::channel();

        match gpus {
            &[] => unreachable!(),
            &[index] => {
                let (sender, commands) = mpsc::channel();
                senders.push(sender);

                let dev = Device::new(index);
                let _worker =
                    s.spawn(move || launch_mono(llama, dev, commands, outputs, use_cuda_grpah));
            }
            #[cfg(not(nccl))]
            [..] => panic!("nccl not found"),
            #[cfg(nccl)]
            gpus => {
                let comms = CommunicatorGroup::new(gpus);
                let barrier = Arc::new(Barrier::new(comms.len()));

                let _workers = comms
                    .into_vec()
                    .into_iter()
                    .map(|comm| {
                        let rank = comm.rank();
                        let total = comm.count();
                        let dist = Distribution::new(rank, 1, total);

                        let (sender, commands) = mpsc::channel();
                        senders.push(sender);

                        let llama = llama.clone();
                        let barrier = barrier.clone();
                        let outputs = outputs.clone();
                        s.spawn(move || {
                            launch_partial(
                                llama,
                                comm,
                                dist,
                                barrier,
                                commands,
                                outputs,
                                use_cuda_grpah,
                            )
                        })
                    })
                    .collect::<Box<_>>();
            }
        }

        service(
            requests,
            session,
            devices,
            senders.into(),
            receiver,
            &gguf,
            max_steps,
        )
    })
}

fn launch_mono(
    llama: LLaMA<Tensor<&[u8], 2>>,
    dev: Device,
    commands: Receiver<Command>,
    outputs: Sender<Output>,
    use_cuda_grpah: bool,
) {
    crate::exec::engine(
        llama,
        dev,
        Distribution::MONO,
        None,
        commands,
        outputs,
        use_cuda_grpah,
        |ctx| Handle::new(ctx),
    )
}

#[cfg(nccl)]
fn launch_partial(
    llama: LLaMA<Tensor<&[u8], 2>>,
    comm: Communicator,
    dist: Distribution,
    barrier: Arc<Barrier>,
    commands: Receiver<Command>,
    outputs: Sender<Output>,
    use_cuda_grpah: bool,
) {
    crate::exec::engine(
        llama,
        comm.device(),
        dist,
        Some(barrier),
        commands,
        outputs,
        use_cuda_grpah,
        |ctx| Handle::with_comm(ctx, comm),
    )
}

fn service(
    requests: Receiver<Task>,
    session: Sender<Receiver<String>>,
    devices: Box<[Device]>,
    senders: Box<[Sender<Command>]>,
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

    let mut cache = devices
        .iter()
        .map(|dev| {
            Some(KVCache::new(
                &cache_template,
                1,
                senders.len(),
                &MemPages::new(Device::new(dev.index())),
            ))
        })
        .collect::<Box<_>>();

    macro_rules! send_all {
        ($prompt:expr) => {
            for (i, sender) in senders.iter().enumerate() {
                sender
                    .send(Command::Insert(Request {
                        session: Session {
                            id: SessionId(i),
                            sample_args: Default::default(),
                            cache: cache[i].take().unwrap(),
                        },
                        prompt: $prompt.to_vec().into(),
                        out: 1,
                    }))
                    .unwrap()
            }
        };
    }

    macro_rules! receive_all {
        () => {{
            let mut next = u32::MAX;
            while cache.iter().any(Option::is_none) {
                let output = receiver.recv().unwrap();
                let (sess, next_) = process(&devices, output, next == u32::MAX);
                if next == u32::MAX {
                    next = next_;
                }
                cache[sess.id.0] = Some(sess.cache);
            }
            next
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
        send_all!(&prompt_tokens);
        // 首轮不计时
        let next = receive_all!();
        send_all!(&[next]);
        response.send(tokeneer.decode(&[next])).unwrap();
        // 循环接收输出
        let mut time = Instant::now();
        for _ in 0..max_steps - 1 {
            let next = receive_all!();
            duration += time.elapsed();
            // 收到休止符
            if next == eos {
                break;
            }
            send_all!(&[next]);
            time = Instant::now();
            n_decode += 1;
            // 会话拒绝输出
            if response.send(tokeneer.decode(&[next])).is_err() {
                break;
            }
        }
    }

    drop(senders);
    let _ = receive_all!();

    (duration, n_decode)
}

fn process(devices: &[Device], output: Output, decode: bool) -> (Session, utok) {
    match output {
        Output::Complete {
            output,
            kv_pair,
            event,
            no_decode,
        } => {
            destruct!([sess] = no_decode);
            let next = if decode {
                devices[sess.id.0].retain_primary().apply(|ctx| {
                    let output = crate::exec::decode(output, kv_pair, event, &ctx.stream());
                    destruct!([pair] = output);
                    let (id, next) = pair;
                    assert_eq!(id, sess.id);
                    destruct!([next] = next);
                    next
                })
            } else {
                devices[sess.id.0].retain_primary().apply(|ctx| {
                    let _ = kv_pair.sprout(ctx);
                    let _ = event.sprout(ctx);
                    0
                })
            };
            (sess, next)
        }
        Output::Overflow(_) => todo!("overflow"),
        Output::Removed(_) => todo!("removed"),
    }
}
