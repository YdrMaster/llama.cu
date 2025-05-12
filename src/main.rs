mod blob;
mod gguf;
mod handle;
mod loader;
mod macros;
mod memory;
mod model;
mod op;
mod range_collector;
mod utils;

use blob::Blob;
use gguf::{GGufModel, map_files};
use ggus::ggml_quants::{digit_layout::types, f16};
use handle::{Attention, Exec, Handle};
use loader::WeightLoader;
use macros::destruct;
use memory::{AddrRegion, MemPages};
use model::sample_indices;
use nn::{Dim, GraphBuilder, TPAction, TPTensor, Tensor, TensorMeta, op as nn_op};
use operators::{
    Operator, TensorLayout,
    attention_kv_cached::{Args as AttnArgs, cuda::Operator as Attn},
    cuda::{self, DevMem, Device, Gpu, HostMem, Stream, VirByte, VirMem, memcpy_h2d},
    random_sample::{Args as SampleArgs, KVPair, cuda::Operator as Sample},
};
use range_collector::RangeCollector;
use std::{collections::HashSet, io::Write, iter::zip};
use tensor::ndarray_layout::ArrayLayout;
use tokeneer::Bpe;

fn main() {
    let mut args = std::env::args();
    let _ = args.next();
    let path = args.next().unwrap();
    let prompt = args.next().unwrap_or("Once upon a time,".into());

    let mut timer = utils::Timer::new();
    // 从文件加载权重
    let maps = map_files(path);
    let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
    let llama = model::init(&mut gguf);
    timer.push("load");
    // 构造图表示
    let nn::Graph(graph::Graph { topo, nodes, edges }) = GraphBuilder::default()
        .register_op("embedding", nn_op::embedding::Embedding)
        .register_op("rms-norm", nn_op::normalization::RmsNorm)
        .register_op("layer-norm", nn_op::normalization::LayerNorm)
        .register_op("attention", nn_op::attention::Attention)
        .register_op("split", nn_op::split::Split)
        .register_op("swiglu", nn_op::activation::SwiGLU)
        .register_op("gelu", nn_op::activation::GeLU)
        .register_op("linear", nn_op::linear::Linear)
        .register_op("rope", nn_op::rope::Rope)
        .register_op("concat", nn_op::concat::Concat)
        .build(
            llama,
            [
                TensorMeta::new(types::U32, [Dim::var("n_tok")]),
                TensorMeta::new(types::U32, [Dim::var("n_tok")]),
                TensorMeta::new(types::U32, [Dim::var("n_out")]),
            ],
        )
        .unwrap();
    timer.push("build");
    // 排布权重存储
    let mut ranges = RangeCollector::new(512);
    let edges = edges
        .into_iter()
        .map(|nn::Edge { meta, external }| nn::Edge {
            meta,
            external: external.map(|nn::External { name, item }| {
                let TPTensor { act, val } = item;
                let tensor = gguf.tensors[&*val].as_deref();
                ranges.insert((act.clone(), tensor.get().as_ptr()), tensor.get().len());
                nn::External {
                    name,
                    item: TPTensor { act, val: tensor },
                }
            }),
        })
        .collect::<Box<_>>();
    // 权重加载
    assert!(cuda::init().is_ok());
    let dev = Device::new(0);
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
                    if mapped.insert(range.clone()) {
                        match act {
                            Some(TPAction { wt, dist }) => {
                                loader.load(dev, &stream, |dst| wt.move_data(dist, dst, &val));
                            }
                            None => {
                                loader.load(dev, &stream, |dst| dst.copy_from_slice(val.get()));
                            }
                        }
                    }
                    nn::External {
                        name,
                        item: val.map(|_| dev.as_ptr().cast()),
                    }
                }),
            })
            .collect::<Box<_>>()
    });
    timer.push("cuda");
    // 创建 kv cache
    let kv_cache = KVCache::new(&gguf, &mut pages);
    // 推理
    let graph = nn::Graph(graph::Graph { topo, nodes, edges });
    let gpu = Gpu::new(dev.context(), Default::default());
    let attn = Attn::new(&gpu);
    let sample = Sample::new(&gpu);
    gpu.apply(|ctx| {
        let stream = ctx.stream();
        let mut output_head = OutputHead {
            sample,
            indices: sample_indices::<2>(&gguf, &stream),
            kv_pair: ctx.malloc::<u8>(size_of::<KVPair<()>>()),
            kv_pair_host: ctx.malloc_host::<u8>(size_of::<KVPair<()>>()),
        };

        let mut handle = Handle::new(ctx);
        timer.push("prepare");

        let tokeneer = Bpe::from_gguf(&gguf);

        let mut models = (0..=6)
            .map(|i| {
                let ans = ModelExec::new(&mut handle, graph.clone(), 1 << i, 1, &mut pages);
                timer.push(format!("build model n_tok = {}", 1 << i));
                ans
            })
            .collect::<Box<_>>();

        println!("{timer}");

        print!("{prompt}");
        std::io::stdout().flush().unwrap();
        let mut tokens = tokeneer.encode(&prompt);
        let mut pos = 0;
        loop {
            let model_idx = tokens.len().next_power_of_two().trailing_zeros() as usize;
            let next = models[model_idx].launch(
                &tokens,
                pos,
                &kv_cache.tensor_vir,
                &mut pages,
                &attn,
                &mut output_head,
                &stream,
            );
            print!("{}", tokeneer.decode(&[next]));
            std::io::stdout().flush().unwrap();
            pos += tokens.len();
            tokens = vec![next];
        }
    })
}

struct KVCache {
    tensor_vir: Tensor<*const VirByte, 2>,
    mem_region: AddrRegion,
}

impl KVCache {
    pub fn new(gguf: &GGufModel, pages: &mut MemPages) -> Self {
        let kv_cache = model::kv_cache::<2>(gguf); // kv cache 的最大容量
        let _each = kv_cache.get() / kv_cache.shape()[0]; // kv cache 每个 token 的尺寸
        let mut mem_region = pages.reserve_vir(*kv_cache.get()); // 为 kv cache 分配虚页
        let tensor_vir = kv_cache
            .map(|_| mem_region.as_ptr()) // 存入张量
            .transform(|layout| layout.transpose(&[3, 1, 2, 0])); // 转置 [nkvh, nblk, 2, nctx, dh]
        pages.map(&mut mem_region);
        Self {
            tensor_vir,
            mem_region,
        }
    }
}

struct ModelExec<'ctx> {
    n_tok: usize,
    execs: Box<[Exec<'ctx>]>,
    workspace: AddrRegion,
    inputs: Box<[Tensor<*const VirByte, 2>]>,
    outputs: Box<[Tensor<*const VirByte, 2>]>,
}

struct OutputHead<'ctx> {
    sample: Sample,
    indices: Tensor<DevMem<'ctx>, 2>,
    kv_pair: DevMem<'ctx>,
    kv_pair_host: HostMem<'ctx>,
}

impl<'ctx> ModelExec<'ctx> {
    fn new(
        handle: &mut Handle<'ctx>,
        graph: nn::Graph<Tensor<*const VirByte, 2>>,
        n_tok: usize,
        n_out: usize,
        pages: &mut MemPages,
    ) -> Self {
        let graph = graph.lower(&[("n_tok", n_tok), ("n_out", n_out)].into(), |t| t);

        let mem_range_map = graph.mem_range_map(8 << 30, 512);

        let mut workspace = pages.reserve_vir(mem_range_map.range.len());
        let ptr = workspace.as_ptr();
        let graph = graph.lower(
            |key| unsafe { ptr.byte_add(mem_range_map.map[&key].start) },
            |&data| data,
        );
        let inputs: Box<[Tensor<*const VirByte, 2>]> = graph
            .0
            .topo
            .global_inputs()
            .map(|i| graph.0.edges[i].clone())
            .collect::<Box<_>>();
        let outputs = graph
            .0
            .topo
            .global_outputs()
            .iter()
            .map(|&i| graph.0.edges[i].clone())
            .collect::<Box<_>>();
        let exec = graph.into_exec();

        // memcpy node 要求当时虚地址有对应的物理页
        pages.map(&mut workspace);

        // 构造 cuda graph
        let execs = handle.merge_cuda_graph(exec);

        // 解除映射回收物理页
        pages.unmap(&mut workspace);

        Self {
            n_tok,
            execs,
            workspace,
            inputs,
            outputs,
        }
    }

    fn launch(
        &mut self,
        tokens: &[u32],
        attn_pos: usize,
        kv_cache: &Tensor<*const VirByte, 2>,
        pages: &mut MemPages,
        attn: &Attn,
        output_head: &mut OutputHead,
        stream: &Stream,
    ) -> u32 {
        pages.map(&mut self.workspace);

        let mut padding = vec![0; self.n_tok];
        padding[..tokens.len()].copy_from_slice(tokens);
        let pos = (attn_pos as u32..).take(self.n_tok).collect::<Vec<_>>();
        let input_data = [
            Blob::from_slice(&padding),
            Blob::from_slice(&pos),
            Blob::from_slice(&[tokens.len() as u32 - 1]),
        ];

        for (input, data) in zip(&self.inputs, input_data.clone()) {
            let ptr = input.get().cast_mut();
            memcpy_h2d(
                unsafe { std::slice::from_raw_parts_mut(ptr.cast(), data.len()) },
                &data,
            )
        }

        for exec in &self.execs {
            match exec {
                Exec::Graph(graph) => {
                    stream.launch_graph(graph);
                }
                Exec::Attention(box_) => {
                    let Attention { iblk, q, k, v, o } = &**box_;

                    // [nkvh, 2, nctx, dh]
                    let kv_cache = kv_cache.clone().transform(|layout| layout.index(1, *iblk));
                    let k_cache = kv_cache.clone().transform(|layout| layout.index(1, 0));
                    let v_cache = kv_cache.clone().transform(|layout| layout.index(1, 1));

                    attn.launch(
                        &AttnArgs {
                            q_layout: layout(q),
                            q_base: offset_ptr(q).cast_mut().cast(),
                            k_layout: layout(k),
                            k_base: offset_ptr(k).cast(),
                            v_layout: layout(v),
                            v_base: offset_ptr(v).cast(),
                            o_layout: layout(o),
                            o_base: offset_ptr(o).cast_mut().cast(),
                            k_cache_layout: layout(&k_cache),
                            k_cache_base: offset_ptr(&k_cache).cast_mut().cast(),
                            v_cache_layout: layout(&v_cache),
                            v_cache_base: offset_ptr(&v_cache).cast_mut().cast(),
                            mask: operators::fuesd_softmax::AttnMask::Causal,
                            pos: attn_pos,
                        },
                        &mut [],
                        stream,
                    )
                    .unwrap()
                }
            }
        }
        destruct!([logits] = self.outputs.clone());
        let logits = logits.transform(|layout| layout.merge_be(0, 2).unwrap());
        let OutputHead {
            sample,
            indices,
            kv_pair,
            kv_pair_host,
        } = output_head;
        sample
            .launch(
                &SampleArgs {
                    kv_pair: TensorLayout {
                        dt: KVPair::<()>::LAYOUT,
                        layout: ArrayLayout::new(&[], &[], 0),
                    },
                    kv_pair_base: kv_pair.as_mut_ptr(),
                    logits: layout(&logits),
                    logits_base: offset_ptr(&logits).cast(),
                    indices: layout(indices),
                    indices_base: indices.get().as_ptr(),
                    config: Default::default(),
                    seed: 0.4,
                },
                &mut [],
                stream,
            )
            .unwrap();
        stream.memcpy_d2h(kv_pair_host, kv_pair).synchronize();
        unsafe { kv_pair_host.as_ptr().cast::<KVPair<f16>>().read() }.idx() as _
    }
}

fn layout<T, const N: usize>(t: &Tensor<T, N>) -> TensorLayout {
    TensorLayout {
        dt: t.dt(),
        layout: t.layout().to_inline_size(),
    }
}

#[inline(always)]
fn offset_ptr<T, const N: usize>(t: &Tensor<*const T, N>) -> *const T {
    unsafe { t.get().byte_offset(t.layout().offset()) }
}
