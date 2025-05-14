use crate::{
    exec::{ModelExec, OutputHead},
    gguf::{GGufModel, map_files},
    handle::Handle,
    loader::WeightLoader,
    memory::{KVCache, MemPages},
    model::{self, insert_sin_cos},
    utils::{RangeCollector, meta},
};
use ggus::{GGufMetaMapExt, ggml_quants::digit_layout::types};
use nn::{
    Dim, Distribution, GraphBuilder, LLaMA, TPAction, TPTensor, Tensor, TensorMeta, op as nn_op,
};
use operators::{
    Operator,
    attention_kv_cached::cuda::Operator as Attn,
    cuda::{self, Device, Gpu},
    nccl::{Communicator, CommunicatorGroup},
    random_sample::{SampleArgs, cuda::Operator as Sample},
};
use smallvec::{SmallVec, smallvec};
use std::{
    collections::HashSet,
    ffi::c_int,
    path::Path,
    sync::mpsc::{self, Receiver, SendError, Sender},
    time::{Duration, Instant},
};
use tokeneer::{Bpe, Tokeneer};

#[allow(non_camel_case_types)]
type utok = u32;

pub fn infer(
    model: impl AsRef<Path>,
    gpus: &[c_int],
    max_steps: usize,
    requests: Receiver<String>,
    session: Sender<Receiver<String>>,
) -> (Duration, usize) {
    // 从文件加载权重
    let maps = map_files(model);
    let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
    insert_sin_cos(&mut gguf);
    // 调取重要配置
    let nvoc = meta![gguf => tokenizer_ggml_tokens].len();
    let eos = meta![gguf => tokenizer_ggml_eos_token_id];
    let llama = model::init(&gguf);
    let tokeneer = Bpe::from_gguf(&gguf);
    let kv_cache = model::kv_cache(&gguf);

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
                let _thread = s.spawn(move || launc_mono(llama, nvoc, &kv_cache, channel));

                service(
                    requests,
                    session,
                    [sender].into(),
                    receiver,
                    tokeneer,
                    max_steps,
                    eos,
                )
            })
        }
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
                        s.spawn(move || launch_partial(llama, dist, nvoc, &kv_cache, channel))
                    })
                    .collect::<Box<_>>();

                service(
                    requests,
                    session,
                    senders.into(),
                    receiver,
                    tokeneer,
                    max_steps,
                    eos,
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
        .register_op("split", nn_op::split::Split);
    ans
}

/// 单卡启动
fn launc_mono(
    llama: LLaMA<Tensor<&[u8], 2>>,
    nvoc: usize,
    template: &Tensor<usize, 2>,
    channel: MonoChannel,
) {
    let MonoChannel { dev, tokens, next } = channel;
    // 构造图表示
    let nn::Graph(graph::Graph { topo, nodes, edges }) = builder()
        .build(
            llama,
            [
                TensorMeta::new(types::U32, [Dim::var("n_tok")]),
                TensorMeta::new(types::U32, [Dim::var("n_tok")]),
                TensorMeta::new(types::U32, [Dim::var("n_out")]),
            ],
        )
        .unwrap();
    // 排布权重存储
    let align = Some(dev.alignment()).filter(|&n| n > 0).unwrap_or(512);
    let mut ranges = RangeCollector::new(align);
    for nn::Edge { external, .. } in &edges {
        if let Some(nn::External { item, .. }) = external {
            ranges.insert(item.get().as_ptr(), item.get().len())
        }
    }
    // 权重加载
    let mut pages = MemPages::new(&dev);
    let mut weight = pages.reserve_vir(ranges.size());
    let mapped = weight.map(0, pages.prop().create(weight.len()));
    let edges = dev.context().apply(|ctx| {
        let mut loader = WeightLoader::new(
            ranges
                .sizes()
                .filter(|&(_, times)| times < 4)
                .map(|(size, _)| size),
        );

        let stream = ctx.stream();
        let mut copied = HashSet::new();
        edges
            .into_iter()
            .map(|nn::Edge { meta, external }| nn::Edge {
                meta,
                external: external.map(|nn::External { name, item }| {
                    let range = &ranges[&item.get().as_ptr()];
                    let dev = &mut mapped[range.clone()];
                    if copied.insert(range.clone()) {
                        loader.load(dev, &stream, |dst| dst.copy_from_slice(item.get()))
                    }
                    nn::External {
                        name,
                        item: item.map(|_| dev.as_ptr().cast()),
                    }
                }),
            })
            .collect::<Box<_>>()
    });
    // 创建 kv cache
    let mut kv_cache = KVCache::new(template, Distribution::MONO, &mut pages);
    // 推理
    let graph = nn::Graph(graph::Graph { topo, nodes, edges });
    let gpu = Gpu::new(dev.context(), Default::default());
    let attn = Attn::new(&gpu);
    let sample = Sample::new(&gpu);
    gpu.apply(|ctx| {
        let mut handle = Handle::new(ctx);
        let mut models = (0..=6)
            .map(|i| ModelExec::new(&mut handle, graph.clone(), 1 << i, 1, &mut pages))
            .collect::<Box<_>>();

        let stream = ctx.stream();
        let mut output_head = OutputHead::new(sample, ctx, nvoc);
        let mut pos = 0;
        for tokens in tokens {
            let to_cache = tokens.len().next_power_of_two();
            kv_cache.prepare(pos + to_cache, &mut pages);

            let model_idx = to_cache.trailing_zeros() as usize;
            let next_ = models[model_idx].launch(
                &tokens,
                pos,
                kv_cache.as_tensor(),
                &mut pages,
                &attn,
                &mut output_head,
                SampleArgs::new(1.2, 0.5, 1000).unwrap(),
                &stream,
            );
            pos += tokens.len();
            if let Err(SendError(_)) = next.send(next_) {
                break;
            }
        }
    })
}

fn launch_partial(
    llama: LLaMA<Tensor<&[u8], 2>>,
    dist: Distribution,
    nvoc: usize,
    template: &Tensor<usize, 2>,
    channel: Channel,
) {
    let Channel { comm, tokens, next } = channel;
    let dev = comm.device();
    // 构造图表示
    let nn::Graph(graph::Graph { topo, nodes, edges }) = builder()
        .register_op("all-reduce", nn_op::all_reduce::AllReduce)
        .build(
            llama.tensor_parallel(dist),
            [
                TensorMeta::new(types::U32, [Dim::var("n_tok")]),
                TensorMeta::new(types::U32, [Dim::var("n_tok")]),
                TensorMeta::new(types::U32, [Dim::var("n_out")]),
            ],
        )
        .unwrap();
    // 排布权重存储
    let align = Some(dev.alignment()).filter(|&n| n > 0).unwrap_or(512);
    let mut ranges = RangeCollector::new(align);
    for nn::Edge { external, .. } in &edges {
        if let Some(nn::External { item, .. }) = external {
            let TPTensor { act, val } = item;
            let len = match act {
                Some(act) => val.get().len() / act.dist.total * act.dist.len,
                None => val.get().len(),
            };
            ranges.insert((act.clone(), val.get().as_ptr()), len)
        }
    }
    // 权重加载
    let mut pages = MemPages::new(&dev);
    let mut weight = pages.reserve_vir(ranges.size());
    let mapped = weight.map(0, pages.prop().create(weight.len()));
    let edges = dev.context().apply(|ctx| {
        let mut loader = WeightLoader::new(
            ranges
                .sizes()
                .filter(|&(_, times)| times < 4)
                .map(|(size, _)| size),
        );

        let stream = ctx.stream();
        let mut copied = HashSet::new();
        edges
            .into_iter()
            .map(|nn::Edge { meta, external }| nn::Edge {
                meta,
                external: external.map(|nn::External { name, item }| {
                    let TPTensor { act, val } = item;
                    let range = &ranges[&(act.clone(), val.get().as_ptr())];
                    let dev = &mut mapped[range.clone()];
                    let ptr = dev.as_ptr().cast();
                    nn::External {
                        name,
                        item: match act.clone() {
                            Some(TPAction { wt, dist }) => {
                                if copied.insert(range.clone()) {
                                    loader.load(dev, &stream, |dst| wt.move_data(dist, dst, &val))
                                }
                                let shape = wt.split_shape(dist, val.shape());
                                Tensor::from_dim_slice(val.dt(), &shape).map(|_| ptr)
                            }
                            None => {
                                if copied.insert(range.clone()) {
                                    loader.load(dev, &stream, |dst| dst.copy_from_slice(val.get()))
                                }
                                val.map(|_| ptr)
                            }
                        },
                    }
                }),
            })
            .collect::<Box<_>>()
    });
    // 创建 kv cache
    let mut kv_cache = KVCache::new(template, dist, &mut pages);
    // 推理
    let graph = nn::Graph(graph::Graph { topo, nodes, edges });
    let gpu = Gpu::new(dev.retain_primary(), Default::default());
    let attn = Attn::new(&gpu);
    let sample = Sample::new(&gpu);
    gpu.apply(|ctx| {
        let mut handle = Handle::with_comm(ctx, comm);
        let mut models = (0..=4)
            .map(|i| ModelExec::new(&mut handle, graph.clone(), 1 << i, 1, &mut pages))
            .collect::<Box<_>>();

        let stream = ctx.stream();
        let mut output_head = OutputHead::new(sample, ctx, nvoc);
        let mut pos = 0;
        for tokens in tokens {
            let to_cache = tokens.len().next_power_of_two();
            kv_cache.prepare(pos + to_cache, &mut pages);

            let model_idx = to_cache.trailing_zeros() as usize;
            let next_ = models[model_idx].launch(
                &tokens,
                pos,
                kv_cache.as_tensor(),
                &mut pages,
                &attn,
                &mut output_head,
                SampleArgs::new(1.2, 0.5, 1000).unwrap(),
                &stream,
            );
            pos += tokens.len();
            if let Some(next) = &next {
                if let Err(SendError(_)) = next.send(next_) {
                    break;
                }
            }
        }
    })
}

fn service(
    requests: Receiver<String>,
    session: Sender<Receiver<String>>,
    senders: Box<[Sender<SmallVec<[utok; 1]>>]>,
    receiver: Receiver<utok>,
    tokeneer: Tokeneer<Bpe>,
    max_steps: usize,
    eos: utok,
) -> (Duration, usize) {
    let mut duration = Duration::ZERO;
    let mut steps = 1;

    let send_all = move |next: SmallVec<[utok; 1]>| {
        for sender in &senders {
            sender.send(next.clone()).unwrap()
        }
    };

    for prompt in requests {
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
            steps += 1;
            // 会话拒绝输出
            if let Err(SendError(_)) = response.send(tokeneer.decode(&[next])) {
                break;
            }
        }
    }

    (duration, steps)
}
