use crate::{
    exec::{ModelExec, OutputHead},
    gguf::{GGufModel, map_files},
    handle::Handle,
    loader::WeightLoader,
    memory::{AddrRegion, MemPages},
    model::{self, insert_sin_cos},
    utils::{RangeCollector, meta, print_now},
};
use ggus::{GGufMetaMapExt, ggml_quants::digit_layout::types};
use nn::{
    Dim, Distribution, GraphBuilder, LLaMA, TPAction, TPTensor, Tensor, TensorMeta, op as nn_op,
};
use operators::{
    Operator,
    attention_kv_cached::cuda::Operator as Attn,
    cuda::{self, DevMem, Device, Gpu, Stream, VirByte, VirMem},
    nccl::{Communicator, CommunicatorGroup},
    random_sample::{Indices, KVPair, RandomSample, cuda::Operator as Sample},
};
use std::{
    collections::HashSet,
    path::Path,
    sync::mpsc::{Receiver, SendError, Sender, channel},
    time::{Duration, Instant},
};
use tokeneer::Bpe;

pub fn infer(model: impl AsRef<Path>, gpus: &[i32], prompt: &str, max_steps: usize) {
    // 从文件加载权重
    let maps = map_files(model);
    let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
    insert_sin_cos(&mut gguf);
    let nvoc = meta![gguf => tokenizer_ggml_tokens].len();
    let kv_cache = model::kv_cache::<2>(&gguf, 1024);
    let llama = model::init(&gguf);
    let tokeneer = Bpe::from_gguf(&gguf);
    let eos = meta![gguf => tokenizer_ggml_eos_token_id];

    assert!(cuda::init().is_ok());

    let (duration, steps) = match gpus {
        &[] => unreachable!(),
        &[index] => {
            let (sender, tokens) = channel();
            let (next, receiver) = channel();
            let channel = MonoChannel {
                dev: Device::new(index),
                tokens,
                next,
            };

            std::thread::scope(|s| {
                let _thread = s.spawn(move || launc_mono(llama, nvoc, &kv_cache, channel));

                sender.send(tokeneer.encode(prompt)).unwrap();

                let mut steps = 1;
                let next = receiver.recv().unwrap();
                sender.send(vec![next]).unwrap();
                print_now!("{prompt}{}", tokeneer.decode(&[next]));

                let time = Instant::now();
                for next in receiver.into_iter().take(max_steps) {
                    if next == eos {
                        break;
                    }
                    sender.send(vec![next]).unwrap();
                    steps += 1;
                    print_now!("{}", tokeneer.decode(&[next]))
                }
                let time = time.elapsed();
                drop(sender);
                (time, steps)
            })
        }
        gpus => {
            let mut senders = Vec::new();
            let mut receiver = None;
            let comms = CommunicatorGroup::new(&gpus);
            let channels = comms
                .into_vec()
                .into_iter()
                .map(|comm| Channel {
                    tokens: {
                        let (sender, receiver) = channel();
                        senders.push(sender);
                        receiver
                    },
                    next: if comm.rank() == 0 {
                        let (sender, receiver_) = channel();
                        assert!(receiver.replace(receiver_).is_none());
                        Some(sender)
                    } else {
                        None
                    },
                    comm,
                })
                .collect::<Box<_>>();
            let receiver = receiver.unwrap();

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
                    .collect::<Vec<_>>();

                let tokens = tokeneer.encode(prompt);
                for sender in &senders {
                    sender.send(tokens.clone()).unwrap()
                }
                let send_all = move |next: u32| {
                    for sender in &senders {
                        sender.send(vec![next]).unwrap()
                    }
                };

                let mut steps = 1;
                let next = receiver.recv().unwrap();
                send_all(next);
                print_now!("{prompt}{}", tokeneer.decode(&[next]));

                let time = Instant::now();
                for next in receiver.into_iter().take(max_steps) {
                    if next == eos {
                        break;
                    }
                    send_all(next);
                    steps += 1;
                    print_now!("{}", tokeneer.decode(&[next]))
                }
                let time = time.elapsed();
                drop(send_all);
                (time, steps)
            })
        }
    };
    let time = duration.div_f32(steps as _);
    println!();
    println!();
    println!(
        "steps = {steps}, perf: {time:?}/tok, {}tok/s",
        Duration::from_secs(1).div_duration_f32(time),
    )
}

struct KVCache {
    tensor_vir: Tensor<*const VirByte, 2>,
    mem_region: AddrRegion,
}

impl KVCache {
    pub fn new(template: &Tensor<usize, 2>, dist: Distribution, pages: &mut MemPages) -> Self {
        let mut shape = template.shape().to_vec();
        shape[3] = shape[3] / dist.total * dist.len;
        let template = Tensor::from_dim_slice(template.dt(), &shape);

        let _each = template.get() / template.shape()[0]; // kv cache 每个 token 的尺寸
        let mut mem_region = pages.reserve_vir(*template.get()); // 为 kv cache 分配虚页
        let tensor_vir = template
            .map(|_| mem_region.as_ptr()) // 存入张量
            .transform(|layout| layout.transpose(&[3, 1, 2, 0])); // 转置 [nkvh, nblk, 2, nctx, dh]
        pages.map(&mut mem_region);

        Self {
            tensor_vir,
            mem_region,
        }
    }
}

struct MonoChannel {
    pub dev: Device,
    pub tokens: Receiver<Vec<u32>>,
    pub next: Sender<u32>,
}

struct Channel {
    pub comm: Communicator,
    pub tokens: Receiver<Vec<u32>>,
    pub next: Option<Sender<u32>>,
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
    let mut ranges = RangeCollector::new(512);
    for nn::Edge { external, .. } in &edges {
        if let Some(nn::External { item, .. }) = external {
            ranges.insert(item.get().as_ptr(), item.get().len())
        }
    }
    // 权重加载
    let MonoChannel { dev, tokens, next } = channel;
    let mut pages = MemPages::new(&dev);
    let page_size = pages.page_size();
    let mut weight = VirMem::new(ranges.size().div_ceil(page_size) * page_size, 0).map_on(&dev);
    let edges = dev.context().apply(|ctx| {
        let mut loader = WeightLoader::new(
            ranges
                .sizes()
                .filter(|&(_, times)| times < 4)
                .map(|(size, _)| size),
        );

        let stream = ctx.stream();
        let mut mapped = HashSet::new();
        edges
            .into_iter()
            .map(|nn::Edge { meta, external }| nn::Edge {
                meta,
                external: external.map(|nn::External { name, item }| {
                    let range = &ranges[&item.get().as_ptr()];
                    let dev = &mut weight[range.clone()];
                    if mapped.insert(range.clone()) {
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
    let kv_cache = KVCache::new(template, Distribution::MONO, &mut pages);
    // 推理
    let graph = nn::Graph(graph::Graph { topo, nodes, edges });
    let gpu = Gpu::new(dev.context(), Default::default());
    let attn = Attn::new(&gpu);
    let sample = Sample::new(&gpu);
    gpu.apply(|ctx| {
        let stream = ctx.stream();
        let mut output_head = OutputHead {
            sample,
            indices: sample_indices::<2>(nvoc, &stream),
            kv_pair: ctx.malloc::<u8>(size_of::<KVPair<()>>()),
            kv_pair_host: ctx.malloc_host::<u8>(size_of::<KVPair<()>>()),
        };

        let mut handle = Handle::new(ctx);

        let mut models = (0..=6)
            .map(|i| ModelExec::new(&mut handle, graph.clone(), 1 << i, 1, &mut pages))
            .collect::<Box<_>>();

        let mut pos = 0;
        for tokens in tokens {
            let model_idx = tokens.len().next_power_of_two().trailing_zeros() as usize;
            let next_ = models[model_idx].launch(
                &tokens,
                pos,
                &kv_cache.tensor_vir,
                &mut pages,
                &attn,
                &mut output_head,
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
    let mut ranges = RangeCollector::new(512);
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
    let Channel { comm, tokens, next } = channel;
    let dev = comm.device();
    let mut pages = MemPages::new(&dev);
    let page_size = pages.page_size();
    let mut weight = VirMem::new(ranges.size().div_ceil(page_size) * page_size, 0).map_on(&dev);
    let edges = dev.context().apply(|ctx| {
        let mut loader = WeightLoader::new(
            ranges
                .sizes()
                .filter(|&(_, times)| times < 4)
                .map(|(size, _)| size),
        );

        let stream = ctx.stream();
        let mut mapped = HashSet::new();
        edges
            .into_iter()
            .map(|nn::Edge { meta, external }| nn::Edge {
                meta,
                external: external.map(|nn::External { name, item }| {
                    let TPTensor { act, val } = item;
                    let range = &ranges[&(act.clone(), val.get().as_ptr())];
                    let dev = &mut weight[range.clone()];
                    let ptr = dev.as_ptr().cast();
                    nn::External {
                        name,
                        item: match act.clone() {
                            Some(TPAction { wt, dist }) => {
                                if mapped.insert(range.clone()) {
                                    loader.load(dev, &stream, |dst| wt.move_data(dist, dst, &val))
                                }
                                let shape = wt.split_shape(dist, val.shape());
                                Tensor::from_dim_slice(val.dt(), &shape).map(|_| ptr)
                            }
                            None => {
                                if mapped.insert(range.clone()) {
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
    let kv_cache = KVCache::new(template, dist, &mut pages);
    // 推理
    let graph = nn::Graph(graph::Graph { topo, nodes, edges });
    let gpu = Gpu::new(dev.retain_primary(), Default::default());
    let attn = Attn::new(&gpu);
    let sample = Sample::new(&gpu);
    gpu.apply(|ctx| {
        let stream = ctx.stream();
        let mut output_head = OutputHead {
            sample,
            indices: sample_indices::<2>(nvoc, &stream),
            kv_pair: ctx.malloc::<u8>(size_of::<KVPair<()>>()),
            kv_pair_host: ctx.malloc_host::<u8>(size_of::<KVPair<()>>()),
        };

        let mut handle = Handle::with_comm(ctx, comm);

        let mut models = (0..=5)
            .map(|i| ModelExec::new(&mut handle, graph.clone(), 1 << i, 1, &mut pages))
            .collect::<Box<_>>();

        let mut pos = 0;
        for tokens in tokens {
            let model_idx = tokens.len().next_power_of_two().trailing_zeros() as usize;
            let next_ = models[model_idx].launch(
                &tokens,
                pos,
                &kv_cache.tensor_vir,
                &mut pages,
                &attn,
                &mut output_head,
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

/// 采样序号表
fn sample_indices<'ctx, const N: usize>(
    nvoc: usize,
    stream: &Stream<'ctx>,
) -> Tensor<DevMem<'ctx>, N> {
    let Indices { n, mem } = Sample::build_indices(nvoc, stream);
    Tensor::from_dim_slice(types::U32, [n]).map(|_| mem)
}
