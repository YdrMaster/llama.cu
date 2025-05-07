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
use ggus::ggml_quants::digit_layout::types;
use handle::{Attention, Exec, Handle};
use loader::WeightLoader;
use macros::destruct;
use memory::{AddrRegion, MemPages};
use nn::{Dim, GraphBuilder, Tensor, TensorMeta, op as nn_op};
use operators::{
    Operator, TensorLayout,
    attention_kv_cached::Args,
    cuda::{self, Device, Gpu, VirByte, VirMem, memcpy_h2d},
};
use range_collector::RangeCollector;
use std::iter::zip;

fn main() {
    let mut timer = utils::Timer::new();

    let maps = map_files(std::env::args().nth(1).unwrap());
    let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
    let llama = model::init(&mut gguf);
    timer.push("load");

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
                TensorMeta::new(types::U32, [Dim::var("n")]),
                TensorMeta::new(types::U32, [Dim::var("n")]),
                TensorMeta::new(types::U32, [1.into()]),
            ],
        )
        .unwrap();
    timer.push("build");

    // for cuda
    let mut ranges = RangeCollector::new(512);
    let edges = edges
        .into_iter()
        .map(|nn::Edge { meta, external }| nn::Edge {
            meta,
            external: external.map(|nn::External { name, item }| nn::External {
                name,
                item: {
                    let ans = gguf.tensors[&*item].as_ref();
                    ranges.insert(ans.get());
                    ans
                },
            }),
        })
        .collect::<Box<_>>();

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
        edges
            .into_iter()
            .map(|nn::Edge { meta, external }| nn::Edge {
                meta,
                external: external.map(|nn::External { name, item }| nn::External {
                    name,
                    item: item.map(|data| {
                        let range = ranges.get(data.as_ptr()).unwrap().clone();
                        let dst = &mut weight[range];
                        loader.load(dst, data, &stream);
                        dst.as_ptr().cast::<VirByte>()
                    }),
                }),
            })
            .collect::<Box<_>>()
    });
    timer.push("cuda");

    // 创建 kv cache
    let kv_cache = model::kv_cache::<2>(&gguf); // kv cache 的最大容量
    let _each = kv_cache.get() / kv_cache.shape()[0]; // kv cache 每个 token 的尺寸
    let mut kv_cache_region = pages.reserve_vir(*kv_cache.get()); // 为 kv cache 分配虚页
    let kv_cache = kv_cache
        .map(|_| kv_cache_region.as_ptr()) // 存入张量
        .transform(|layout| layout.transpose(&[3, 1, 2, 0])); // 转置 [nkvh, nblk, 2, nctx, dh]
    pages.map(&mut kv_cache_region);

    let graph = nn::Graph(graph::Graph { topo, nodes, edges });
    let gpu = Gpu::new(dev.context(), Default::default());
    let attn = operators::attention_kv_cached::cuda::Operator::new(&gpu);
    gpu.apply(|ctx| {
        let mut handle = Handle::new(ctx);
        let tokens = [9038u32, 2501, 263, 931, 29892];
        timer.push("prepare");

        let mut model = ModelExec::new(&mut handle, graph, tokens.len(), &mut pages);
        timer.push(format!("build model n_tok = {}", tokens.len()));

        pages.map(&mut model.workspace);
        timer.push("map workspace");

        let n_tok = tokens.len();
        let tokens = Blob::from_slice(&tokens);
        let pos = Blob::from_slice(&(0..n_tok as u32).collect::<Vec<_>>());
        let out_idx = Blob::from_slice(&[n_tok as u32 - 1]);
        let input_data = [tokens, pos, out_idx];
        let attn_pos = 0;

        for (input, data) in zip(&model.inputs, input_data.clone()) {
            let ptr = input.get().cast_mut();
            memcpy_h2d(
                unsafe { std::slice::from_raw_parts_mut(ptr.cast(), data.len()) },
                &data,
            )
        }
        timer.push("copy input");

        let stream = ctx.stream();
        for exec in &model.execs {
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
                        &Args {
                            q_layout: layout(q),
                            q_base: offset_ptr(q).cast_mut().cast(),
                            k_layout: layout(k),
                            k_base: offset_ptr(k).cast_mut().cast(),
                            v_layout: layout(v),
                            v_base: offset_ptr(v).cast_mut().cast(),
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
                        &stream,
                    )
                    .unwrap()
                }
            }
        }

        stream.synchronize();
        timer.push("exec");
        println!("{timer}");

        destruct!([x] = model.outputs);
        utils::fmt(&x, ctx);
    });
}

struct ModelExec<'ctx> {
    execs: Box<[Exec<'ctx>]>,
    workspace: AddrRegion,
    inputs: Box<[Tensor<*const VirByte, 2>]>,
    outputs: Box<[Tensor<*const VirByte, 2>]>,
}

impl<'ctx> ModelExec<'ctx> {
    fn new(
        handle: &mut Handle<'ctx>,
        graph: nn::Graph<Tensor<*const VirByte, 2>>,
        n_tok: usize,
        pages: &mut MemPages,
    ) -> Self {
        let graph = graph.lower(&[("n", n_tok)].into(), |t| t);

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
            execs,
            workspace,
            inputs,
            outputs,
        }
    }
}

fn layout<T, const N: usize>(t: &Tensor<*const T, N>) -> TensorLayout {
    TensorLayout {
        dt: t.dt(),
        layout: t.layout().to_inline_size(),
    }
}

#[inline(always)]
fn offset_ptr<T, const N: usize>(t: &Tensor<*const T, N>) -> *const T {
    unsafe { t.get().byte_offset(t.layout().offset()) }
}
