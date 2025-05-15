use crate::{
    Task,
    exec::{KVCache, ModelGroup, Request},
    handle::Handle,
    memory::MemPages,
    model::{GGufModel, Message, map_files},
    upos,
    utils::meta,
};
use ggus::{GGufMetaMapExt, ggml_quants::digit_layout::types};
use nn::{Dim, Distribution, Graph, GraphBuilder, LLaMA, NNGraph, Tensor, TensorMeta, op as nn_op};
use operators::{
    Operator,
    attention_kv_cached::cuda::Operator as Attn,
    cuda::{self, Device, Gpu},
    random_sample::{KVPair, SampleArgs, cuda::Operator as Sample},
};
use smallvec::{SmallVec, smallvec};
use std::{
    ffi::c_int,
    path::Path,
    sync::mpsc::{self, Receiver, SendError, Sender},
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
) -> (Duration, usize) {
    // 从文件加载权重
    let maps = map_files(model);
    let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
    gguf.insert_sin_cos();
    // 调取重要配置
    let llama = gguf.llama();
    let kv_cache = gguf.kv_cache();

    assert!(cuda::init().is_ok());
    let (next, receiver) = mpsc::channel();
    match gpus {
        &[] => unreachable!(),
        &[index] => {
            let (sender, tokens) = mpsc::channel();
            let channel = MonoChannel {
                dev: Device::new(index),
                tokens,
                next,
            };

            std::thread::scope(|s| {
                let _thread = s.spawn(move || launc_mono(llama, &kv_cache, channel));

                service(
                    requests,
                    session,
                    [sender].into(),
                    receiver,
                    &gguf,
                    max_steps,
                )
            })
        }
        #[cfg(not(nccl))]
        [..] => panic!("nccl not found"),
        #[cfg(nccl)]
        gpus => {
            let mut senders = Vec::new();
            let comms = CommunicatorGroup::new(gpus);
            let channels = comms
                .into_vec()
                .into_iter()
                .map(|comm| Channel {
                    tokens: {
                        let (sender, receiver) = mpsc::channel();
                        senders.push(sender);
                        receiver
                    },
                    next: if comm.rank() == 0 {
                        Some(next.clone())
                    } else {
                        None
                    },
                    comm,
                })
                .collect::<Box<_>>();

            std::thread::scope(|s| {
                let _threads = channels
                    .into_iter()
                    .enumerate()
                    .map(|(i, channel)| {
                        let llama = llama.clone();
                        let kv_cache = kv_cache.clone();
                        let dist = Distribution::new(i, 1, gpus.len());
                        s.spawn(move || launch_partial(llama, dist, &kv_cache, channel))
                    })
                    .collect::<Box<_>>();

                service(
                    requests,
                    session,
                    senders.into(),
                    receiver,
                    &gguf,
                    max_steps,
                )
            })
        }
    }
}

struct MonoChannel {
    dev: Device,
    tokens: Receiver<SmallVec<[utok; 1]>>,
    next: Sender<utok>,
}

#[cfg(nccl)]
struct Channel {
    comm: Communicator,
    tokens: Receiver<SmallVec<[utok; 1]>>,
    next: Option<Sender<utok>>,
}

fn builder() -> GraphBuilder {
    let mut ans = GraphBuilder::default();
    ans.register_op("embedding", nn_op::embedding::Embedding)
        .register_op("rms-norm", nn_op::normalization::RmsNorm)
        .register_op("linear", nn_op::linear::Linear)
        .register_op("rope", nn_op::rope::Rope)
        .register_op("attention", nn_op::attention::Attention)
        .register_op("swiglu", nn_op::activation::SwiGLU)
        .register_op("concat", nn_op::concat::Concat)
        .register_op("split", nn_op::split::Split)
        .register_op("all-reduce", nn_op::all_reduce::AllReduce);
    ans
}

/// 单卡启动
fn launc_mono(
    mut llama: LLaMA<Tensor<&[u8], 2>>,
    template: &Tensor<usize, 2>,
    channel: MonoChannel,
) {
    let MonoChannel { dev, tokens, next } = channel;
    let dist = Distribution::MONO;

    let output_head = llama.output_head.take().unwrap();
    let NNGraph(Graph { topo, nodes, edges }) = builder()
        .build(
            llama.tensor_parallel(dist),
            [
                TensorMeta::new(types::U32, [Dim::var("n_tok")]),
                TensorMeta::new(types::U32, [Dim::var("n_tok")]),
            ],
        )
        .unwrap();

    // 权重加载
    let mut pages = MemPages::new(dev);
    let (_weight, edges) = pages.load_weight(edges);

    // 推理
    let graph = NNGraph(Graph { topo, nodes, edges });
    let gpu = Gpu::new(pages.dev().context(), Default::default());
    let attn = Attn::new(&gpu);
    let sample = Sample::new(&gpu);
    gpu.apply(|ctx| {
        let mut handle = Handle::new(ctx);
        let mut models = ModelGroup::new(
            [1, 8, 16, 32, 64],
            &graph,
            output_head,
            attn,
            sample,
            &mut handle,
            &mut pages,
        );

        let mut kv_pair_host = ctx.malloc_host::<KVPair<()>>(1);

        // 创建 kv cache
        let mut kv_cache = KVCache::new(template, Distribution::MONO, &mut pages);

        let stream = ctx.stream();
        let mut pos = 0;
        for tokens in tokens {
            let len = tokens.len();
            kv_cache.prepare(pos as usize + len, &mut pages);

            let request = Request {
                sample_args: SampleArgs::new(1.2, 0.5, 1000).unwrap(),
                tokens,
                kv_cache: kv_cache.as_tensor().clone(),
                pos,
                out: 1,
            };

            let kv_pair = models.launch([request].into(), &mut handle, &mut pages, &stream);
            pos += len as upos;
            stream
                .memcpy_d2h(&mut kv_pair_host, &kv_pair)
                .synchronize()
                .free(kv_pair);
            let pair = unsafe { kv_pair_host.as_mut_ptr().cast::<KVPair<()>>().read() };
            if let Err(SendError(_)) = next.send(pair.idx() as _) {
                break;
            }
        }
    })
}

#[cfg(nccl)]
fn launch_partial(
    mut llama: LLaMA<Tensor<&[u8], 2>>,
    dist: Distribution,
    template: &Tensor<usize, 2>,
    channel: Channel,
) {
    let Channel { comm, tokens, next } = channel;
    let dev = comm.device();

    let output_head = llama.output_head.take().unwrap();
    let NNGraph(Graph { topo, nodes, edges }) = builder()
        .build(
            llama.tensor_parallel(dist),
            [
                TensorMeta::new(types::U32, [Dim::var("n_tok")]),
                TensorMeta::new(types::U32, [Dim::var("n_tok")]),
            ],
        )
        .unwrap();

    // 权重加载
    let mut pages = MemPages::new(dev);
    let (_weight, edges) = pages.load_weight(edges);

    // 推理
    let graph = NNGraph(Graph { topo, nodes, edges });
    let gpu = Gpu::new(pages.dev().retain_primary(), Default::default());
    let attn = Attn::new(&gpu);
    let sample = Sample::new(&gpu);
    gpu.apply(|ctx| {
        let mut handle = Handle::with_comm(ctx, comm);
        let mut models = ModelGroup::new(
            [1, 8, 16, 32, 64],
            &graph,
            output_head,
            attn,
            sample,
            &mut handle,
            &mut pages,
        );

        let mut kv_pair_host = ctx.malloc_host::<KVPair<()>>(1);

        // 创建 kv cache
        let mut kv_cache = KVCache::new(template, dist, &mut pages);

        let stream = ctx.stream();
        let mut pos = 0;
        for tokens in tokens {
            let len = tokens.len();
            kv_cache.prepare(pos as usize + len, &mut pages);

            let request = Request {
                sample_args: SampleArgs::new(1.2, 0.5, 1000).unwrap(),
                tokens,
                kv_cache: kv_cache.as_tensor().clone(),
                pos,
                out: 1,
            };

            let kv_pair = models.launch([request].into(), &mut handle, &mut pages, &stream);
            pos += len as upos;
            if let Some(next) = &next {
                stream
                    .memcpy_d2h(&mut kv_pair_host, &kv_pair)
                    .synchronize()
                    .free(kv_pair);
                let pair = unsafe { kv_pair_host.as_mut_ptr().cast::<KVPair<()>>().read() };
                if let Err(SendError(_)) = next.send(pair.idx() as _) {
                    break;
                }
            }
        }
    })
}

fn service(
    requests: Receiver<Task>,
    session: Sender<Receiver<String>>,
    senders: Box<[Sender<SmallVec<[utok; 1]>>]>,
    receiver: Receiver<utok>,
    gguf: &GGufModel,
    max_steps: usize,
) -> (Duration, usize) {
    let tokeneer = Bpe::from_gguf(gguf);
    let chat_template = gguf.chat_template(&tokeneer);
    let eos = meta![gguf => tokenizer_ggml_eos_token_id];
    let mut duration = Duration::ZERO;
    let mut n_decode = 1;

    let send_all = move |next: SmallVec<[utok; 1]>| {
        for sender in &senders {
            sender.send(next.clone()).unwrap()
        }
    };

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
        let prompt_tokens = SmallVec::from_vec(tokeneer.encode(&prompt));
        send_all(prompt_tokens);
        // 首轮不计时
        let next = receiver.recv().unwrap();
        send_all(smallvec![next]);
        response.send(tokeneer.decode(&[next])).unwrap();
        // 循环接收输出
        let mut time = Instant::now();
        for next in receiver.iter().take(max_steps - 1) {
            duration += time.elapsed();
            // 收到休止符
            if next == eos {
                break;
            }
            send_all(smallvec![next]);
            time = Instant::now();
            n_decode += 1;
            // 会话拒绝输出
            if let Err(SendError(_)) = response.send(tokeneer.decode(&[next])) {
                break;
            }
        }
    }

    (duration, n_decode)
}
