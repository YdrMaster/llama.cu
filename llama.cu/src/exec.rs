use crate::{
    handle::{Attention, Exec, Handle},
    memory::{AddrRegion, MemPages},
    utils::{self, Blob, destruct, layout, offset_ptr},
};
use ggus::ggml_quants::f16;
use nn::Tensor;
use operators::{
    Operator, TensorLayout,
    attention_kv_cached::{Args as AttnArgs, cuda::Operator as Attn},
    cuda::{CurrentCtx, DevMem, HostMem, Stream, VirByte, memcpy_h2d},
    random_sample::{
        Args as SampleArgs, Indices, KVPair, RandomSample, SampleArgs as Config,
        cuda::Operator as Sample,
    },
};
use std::iter::zip;
use tensor::{digit_layout::types, ndarray_layout::ArrayLayout};

pub struct ModelExec<'ctx> {
    n_tok: usize,
    execs: Box<[Exec<'ctx>]>,
    workspace: AddrRegion,
    inputs: Box<[Tensor<*const VirByte, 2>]>,
    outputs: Box<[Tensor<*const VirByte, 2>]>,
}

impl<'ctx> ModelExec<'ctx> {
    pub fn new(
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
        pages.map(workspace.pages(..));

        // 构造 cuda graph
        let execs = handle.merge_cuda_graph(exec);

        // 解除映射回收物理页
        pages.unmap(workspace.pages(..));

        Self {
            n_tok,
            execs,
            workspace,
            inputs,
            outputs,
        }
    }

    pub fn launch(
        &mut self,
        tokens: &[u32],
        attn_pos: usize,
        kv_cache: &Tensor<*const VirByte, 2>,
        pages: &mut MemPages,
        attn: &Attn,
        output_head: &mut OutputHead,
        config: Config,
        stream: &Stream,
    ) -> u32 {
        pages.map(self.workspace.pages(..));

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
                Exec::Graph(graph, stub) => {
                    stream.launch_graph(graph);
                    if !stub.is_empty() {
                        for t in stub {
                            utils::fmt(t, stream.ctx())
                        }
                        std::process::exit(0);
                    }
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
                    seed: if config.is_argmax() {
                        1.
                    } else {
                        rand::random()
                    },
                    config,
                },
                &mut [],
                stream,
            )
            .unwrap();
        stream.memcpy_d2h(kv_pair_host, kv_pair).synchronize();
        unsafe { kv_pair_host.as_ptr().cast::<KVPair<f16>>().read() }.idx() as _
    }
}

pub struct OutputHead<'ctx> {
    sample: Sample,
    indices: Tensor<DevMem<'ctx>, 2>,
    kv_pair: DevMem<'ctx>,
    kv_pair_host: HostMem<'ctx>,
}

impl<'ctx> OutputHead<'ctx> {
    pub fn new(sample: Sample, ctx: &'ctx CurrentCtx, nvoc: usize) -> Self {
        let stream = ctx.stream();
        Self {
            sample,
            indices: {
                let Indices { n, mem } = Sample::build_indices(nvoc, &stream);
                Tensor::from_dim_slice(types::U32, [n]).map(|_| mem)
            },
            kv_pair: stream.malloc::<u8>(size_of::<KVPair<()>>()),
            kv_pair_host: ctx.malloc_host::<u8>(size_of::<KVPair<()>>()),
        }
    }
}
