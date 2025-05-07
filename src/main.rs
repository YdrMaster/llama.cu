mod blob;
mod exec;
mod fmt;
mod gguf;
mod llama;
mod loader;
mod macros;
mod op;
mod range_collector;

use blob::Blob;
use exec::{Exec, merge_cuda_graph};
use gguf::{GGufModel, map_files};
use ggus::ggml_quants::{digit_layout::types, f16};
use loader::WeightLoader;
use macros::destruct;
use nn::{Dim, GraphBuilder, Tensor, TensorMeta, op as nn_op};
use operators::{
    Operator, TensorLayout,
    attention_kv_cached::Args,
    cuda::{self, Device, Gpu, VirByte, VirMem, memcpy_h2d},
};
use range_collector::RangeCollector;
use std::{iter::zip, time::Instant};
use tensor::digit_layout::DigitLayout;

// cargo run --release -- ../TinyStory-5M-v0.0-F32.gguf
fn main() {
    let mut times = TimeCollector::default();
    let maps = map_files(std::env::args().nth(1).unwrap());
    let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
    let llama = llama::init(&mut gguf);
    times.push("load");

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
    times.push("build");

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
    let page_size = dev.mem_prop().granularity_minimum();
    let mut mapped = VirMem::new(ranges.size().div_ceil(page_size) * page_size, 0).map_on(&dev);
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
                        let dst = &mut mapped[range];
                        loader.load(dst, data, &stream);
                        dst.as_ptr().cast::<VirByte>()
                    }),
                }),
            })
            .collect::<Box<_>>()
    });
    let (weight_vir, weight_phy) = mapped.unmap();
    times.push("cuda");

    let tokens = [9038u32, 2501, 263, 931, 29892];
    let n_tok = tokens.len();
    let graph = nn::Graph(graph::Graph { topo, nodes, edges }).lower(&[("n", n_tok)].into(), |t| t);
    times.push("fix shape");

    let mem_range_map = graph.mem_range_map(8 << 30, 512);

    let workspace_vir = reserve_pages(mem_range_map.range.len(), page_size);
    let ptr = workspace_vir[0].as_ptr();
    let graph = graph.lower(
        |key| unsafe { ptr.byte_add(mem_range_map.map[&key].start) },
        |&data| data,
    );
    let global_inputs = graph
        .0
        .topo
        .global_inputs()
        .map(|i| graph.0.edges[i].clone())
        .collect::<Box<_>>();
    let global_outputs = graph
        .0
        .topo
        .global_outputs()
        .iter()
        .map(|&i| graph.0.edges[i].clone())
        .collect::<Box<_>>();
    let exec = graph.into_exec();
    times.push("into exec");
    // 创建 kv cache
    let kv_cache = llama::kv_cache::<2>(&gguf); // kv cache 的最大容量
    let each = kv_cache.get() / kv_cache.shape()[0]; // kv cache 每个 token 的尺寸
    let kv_cache_vir = reserve_pages(*kv_cache.get(), page_size); // 为 kv cache 分配虚页
    let kv_cache = kv_cache
        .map(|_| kv_cache_vir[0].as_ptr()) // 存入张量
        .transform(|layout| layout.transpose(&[3, 1, 2, 0])); // 转置 [nkvh, nblk, 2, nctx, dh]

    // memcpy node 要求当时虚地址有对应的物理页
    let _weight = weight_vir.map(weight_phy);
    let _workspace = workspace_vir
        .into_iter()
        .map(|vir| vir.map_on(&dev))
        .collect::<Box<_>>();
    let _kv_cache = kv_cache_vir
        .into_iter()
        .map(|vir| vir.map_on(&dev))
        .collect::<Box<_>>();

    let tokens = Blob::from_slice(&tokens);
    let pos = Blob::from_slice(&(0..n_tok as u32).collect::<Vec<_>>());
    let out_idx = Blob::from_slice(&[n_tok as u32 - 1]);
    let input_data = [tokens, pos, out_idx];
    let attn_pos = 0;

    let context = dev.context();
    let gpu = Gpu::new(context, Default::default());
    let attn = operators::attention_kv_cached::cuda::Operator::new(&gpu);
    gpu.apply(|ctx| {
        let (_handle, exec) = merge_cuda_graph(ctx, exec);
        times.push("build cuda graph");

        for (input, data) in zip(&global_inputs, input_data.clone()) {
            let ptr = input.get().cast_mut();
            memcpy_h2d(
                unsafe { std::slice::from_raw_parts_mut(ptr.cast(), data.len()) },
                &data,
            )
        }

        let stream = ctx.stream();
        for exec in &exec {
            match exec {
                Exec::Graph(graph, outputs) => {
                    graph.launch(&stream);
                    if !outputs.is_empty() {
                        destruct!([y] = outputs);
                        fmt::fmt(&y, ctx);
                        std::process::exit(0)
                    }
                }
                Exec::Attention(box_) => {
                    let exec::Attention { iblk, q, k, v, o } = &**box_;

                    // [nkvh, 2, nctx, dh]
                    let kv_cache = kv_cache.clone().transform(|layout| layout.index(1, *iblk));
                    let k_cache = kv_cache.clone().transform(|layout| layout.index(1, 0));
                    let v_cache = kv_cache.clone().transform(|layout| layout.index(1, 1));

                    attn.launch(
                        &Args {
                            q_layout: layout(&q),
                            q_base: offset_ptr(&q).cast_mut().cast(),
                            k_layout: layout(&k),
                            k_base: offset_ptr(&k).cast_mut().cast(),
                            v_layout: layout(&v),
                            v_base: offset_ptr(&v).cast_mut().cast(),
                            o_layout: layout(&o),
                            o_base: offset_ptr(&o).cast_mut().cast(),
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
        times.push("launch");
        println!("{times}");

        destruct!([x] = global_outputs);
        fmt::fmt(&x, ctx);
    });
}

#[derive(Default)]
#[repr(transparent)]
struct TimeCollector(Vec<(String, Instant)>);

impl TimeCollector {
    pub fn push(&mut self, name: impl std::fmt::Display) {
        self.0.push((name.to_string(), Instant::now()))
    }
}

impl std::fmt::Display for TimeCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name_width = self.0.iter().map(|(name, _)| name.len()).max().unwrap_or(0) + 2;
        for i in 1..self.0.len() {
            writeln!(
                f,
                "{:·<name_width$}{:?}",
                self.0[i].0,
                self.0[i].1 - self.0[i - 1].1
            )?
        }
        Ok(())
    }
}

fn reserve_pages(size: usize, page_size: usize) -> Box<[VirMem]> {
    let n_pages = size.div_ceil(page_size);
    if n_pages == 0 {
        return Box::new([]);
    }

    let mut ans = Vec::with_capacity(n_pages);
    let first = VirMem::new(page_size, 0);
    let mut end = first.as_ptr_range().end;
    ans.push(first);
    while ans.len() < n_pages {
        let next = VirMem::new(page_size, end as _);
        let ptr = next.as_ptr();
        if ptr != end {
            ans.clear()
        }
        end = next.as_ptr_range().end;
        ans.push(next)
    }
    ans.into()
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
