use crate::{
    handle::{Attention, Exec, Handle},
    load::WeightLoader,
    memory::MemPages,
    op::{self, Operator as _},
    upos,
    utils::{self, destruct, dims, layout, offset_ptr},
};
use nn::{
    Arg, Linear, NNGraph, NormType, Normalization, Tensor, digit_layout::types,
    ndarray_layout::ArrayLayout,
};
use operators::{
    Operator as _, TensorLayout,
    attention_kv_cached::{Args as AttnArgs, cuda::Operator as Attn},
    cuda::{CurrentCtx, DevMem, HostMem, Stream, VirByte, VirMem, memcpy_h2d},
    random_sample::{
        Args as SampleArgs, Indices, KVPair, RandomSample, SampleArgs as Config,
        cuda::Operator as Sample,
    },
};
use smallvec::SmallVec;
use std::iter::zip;
use tokeneer::utok;

pub struct ModelExec<'ctx> {
    n_tok: usize,
    execs: Box<[Exec<'ctx>]>,
    workspace: VirMem,
    workspace_mapped: bool,
    inputs: Box<[Tensor<*const VirByte, 2>]>,
    outputs: Box<[Tensor<*const VirByte, 2>]>,
}

pub struct Request {
    pub config: Config,
    pub tokens: SmallVec<[utok; 1]>,
    pub kv_cache: Tensor<*const VirByte, 2>,
    pub pos: upos,
    pub out: usize,
}

impl<'ctx> ModelExec<'ctx> {
    pub fn new(
        graph: NNGraph<Tensor<*const VirByte, 2>>,
        n_tok: usize,
        handle: &mut Handle<'ctx>,
        pages: &mut MemPages,
    ) -> Self {
        let graph = graph.lower(&[("n_tok", n_tok)].into(), |t| t);

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
        pages.map(&mut workspace, ..);

        // 构造 cuda graph
        let execs = handle.merge_cuda_graph(exec);

        // 解除映射回收物理页
        pages.unmap(&mut workspace, ..);

        Self {
            n_tok,
            execs,
            workspace,
            workspace_mapped: false,
            inputs,
            outputs,
        }
    }
}

impl ModelExec<'_> {
    pub fn launch<'ctx>(
        &mut self,
        attn: &Attn,
        handle: &mut Handle,
        pages: &mut MemPages,
        output_head: &mut OutputHead,
        requests: Box<[Request]>,
        stream: &Stream<'ctx>,
    ) -> DevMem<'ctx> {
        if !self.workspace_mapped {
            self.workspace_mapped = true;
            pages.map(&mut self.workspace, ..)
        }
        // 初始化输入
        let (n_out, [padding, pos, out_idx]) = fill_inputs(self.n_tok, &requests, stream.ctx());

        // 拷贝到硬件
        for (input, data) in zip(&self.inputs, [padding, pos]) {
            let ptr = input.get().cast_mut();
            memcpy_h2d(
                unsafe { std::slice::from_raw_parts_mut(ptr.cast(), data.len()) },
                &data,
            )
        }
        // 执行
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
                    let mut start = 0;
                    for req in &requests {
                        // [nkvh, 2, nctx, dh]
                        let cache = req.kv_cache.clone();
                        let cache = cache.transform(|layout| layout.index(1, *iblk));
                        let k_cache = cache.clone().transform(|layout| layout.index(1, 0));
                        let v_cache = cache.clone().transform(|layout| layout.index(1, 1));
                        // [nh, n, dh]
                        let len = req.tokens.len();
                        let q = q.clone().transform(|layout| layout.slice(1, start, 1, len));
                        let k = k.clone().transform(|layout| layout.slice(1, start, 1, len));
                        let v = v.clone().transform(|layout| layout.slice(1, start, 1, len));
                        let o = o.clone().transform(|layout| layout.slice(1, start, 1, len));
                        start += len;
                        attn.launch(
                            &AttnArgs {
                                q_layout: layout(&q),
                                q_base: offset_ptr(&q).cast_mut().cast(),
                                k_layout: layout(&k),
                                k_base: offset_ptr(&k).cast(),
                                v_layout: layout(&v),
                                v_base: offset_ptr(&v).cast(),
                                o_layout: layout(&o),
                                o_base: offset_ptr(&o).cast_mut().cast(),
                                k_cache_layout: layout(&k_cache),
                                k_cache_base: offset_ptr(&k_cache).cast_mut().cast(),
                                v_cache_layout: layout(&v_cache),
                                v_cache_base: offset_ptr(&v_cache).cast_mut().cast(),
                                mask: operators::fuesd_softmax::AttnMask::Causal,
                                pos: req.pos as _,
                            },
                            &mut [],
                            stream,
                        )
                        .unwrap()
                    }
                }
            }
        }
        destruct!([x] = self.outputs.clone());
        output_head.launch(
            x,
            out_idx,
            n_out,
            requests
                .iter()
                .flat_map(|req| std::iter::repeat_n(req.config, req.out)),
            handle,
            stream,
        )
    }
}

pub struct OutputHead<'ctx> {
    norm: Tensor<DevMem<'ctx>, 2>,
    linear: Tensor<DevMem<'ctx>, 2>,
    epsilon: Option<Arg>,

    sample: Sample,
    indices: Tensor<DevMem<'ctx>, 2>,
}

impl<'ctx> OutputHead<'ctx> {
    pub fn new(
        nn: nn::OutputHead<Tensor<&[u8], 2>>,
        sample: Sample,
        ctx: &'ctx CurrentCtx,
        nvoc: usize,
    ) -> Self {
        let nn::OutputHead {
            out_norm: Normalization { items, epsilon, .. },
            lm_head: Linear { weight, .. },
        } = nn;
        let norm = match items {
            NormType::RmsNorm { scale, .. } => scale,
            NormType::LayerNorm { .. } => todo!(),
        };
        let linear = weight;

        let stream = ctx.stream();
        let mut loader = WeightLoader::new([]);
        let mut load = |t: Tensor<&[u8], 2>| {
            let dst = stream.malloc::<u8>(t.get().len());
            let (host, mut ans) = t.replace(dst);
            loader.load(ans.get_mut(), &stream, |inter| {
                inter.copy_from_slice(host);
            });
            ans
        };

        Self {
            norm: load(norm),
            linear: load(linear),
            epsilon: Some(epsilon.into()),
            sample,
            indices: {
                let Indices { n, mem } = Sample::build_indices(nvoc, &stream);
                Tensor::from_dim_slice(types::U32, [n]).map(|_| mem)
            },
        }
    }
}

impl OutputHead<'_> {
    pub fn launch<'ctx>(
        &self,
        x: Tensor<*const VirByte, 2>,
        out_idx: HostMem,
        n_out: usize,
        config: impl IntoIterator<Item = Config>,
        handle: &mut Handle,
        stream: &Stream<'ctx>,
    ) -> DevMem<'ctx> {
        let Self {
            norm,
            linear,
            epsilon,
            sample,
            indices,
        } = self;
        dims!([_, d] = x);
        let out_idx = stream.from_host::<u8>(&out_idx);
        let out_idx = Tensor::from_dim_slice(types::U32, [n_out]).map(|_| out_idx.as_ptr().cast());
        // gather
        let mut out = Tensor::new(x.dt(), [n_out, d]).map(|len| stream.malloc::<u8>(len));
        let out = out.as_mut().map(|mem| mem.as_ptr().cast());
        op::Embedding::launch(handle, None, [x, out_idx], [out.clone()], stream);
        // norm
        let scale = norm.as_ref().map(|mem| mem.as_ptr().cast());
        op::RmsNorm::launch(
            handle,
            epsilon.clone(),
            [out.clone(), scale],
            [out.clone()],
            stream,
        );
        // linear
        dims!([nvoc, _] = linear);
        let mut logits = Tensor::new(out.dt(), [n_out, nvoc]).map(|len| stream.malloc::<u8>(len));
        let logits = logits.as_mut().map(|mem| mem.as_ptr().cast());
        let lm_head = linear.as_ref().map(|mem| mem.as_ptr().cast());
        op::Linear::launch(
            handle,
            Some(false.into()),
            [out, lm_head],
            [logits.clone()],
            stream,
        );
        let mut kv_pair = stream.malloc::<KVPair<()>>(n_out);
        for (i, config) in config.into_iter().enumerate() {
            let logit = logits.clone().transform(|layout| layout.index(0, i));
            let kv_pair = &mut kv_pair[i * size_of::<KVPair<()>>()..];
            sample
                .launch(
                    &SampleArgs {
                        kv_pair: TensorLayout {
                            dt: KVPair::<()>::LAYOUT,
                            layout: ArrayLayout::new(&[], &[], 0),
                        },
                        kv_pair_base: kv_pair.as_mut_ptr(),
                        logits: layout(&logit),
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
                .unwrap()
        }
        kv_pair
    }
}

/// 构造输入数据
fn fill_inputs<'ctx>(
    n_tok: usize,
    requests: &[Request],
    ctx: &'ctx CurrentCtx,
) -> (usize, [HostMem<'ctx>; 3]) {
    let mut ans0 = ctx.malloc_host::<utok>(n_tok);
    let mut ans1 = ctx.malloc_host::<upos>(n_tok);
    let mut ans2 = ctx.malloc_host::<utok>(n_tok);

    let tokens: &mut [utok] = cast_slice_mut(&mut ans0);
    let pos: &mut [upos] = cast_slice_mut(&mut ans1);
    let out_idx: &mut [utok] = cast_slice_mut(&mut ans2);
    let mut itok = 0;
    let mut iout = 0;
    for req in requests {
        let seq_len = req.tokens.len();
        for (j, &tok) in req.tokens.iter().enumerate() {
            if j >= seq_len - req.out {
                out_idx[iout] = itok as _;
                iout += 1
            }

            tokens[itok] = tok;
            pos[itok] = req.pos + j as upos;
            itok += 1;
        }
    }
    tokens[itok..].fill(0);
    pos[itok..].fill(0);

    (iout, [ans0, ans1, ans2])
}

fn cast_slice_mut<T: Copy, U: Copy>(slice: &mut [T]) -> &mut [U] {
    let ([], ans, []) = (unsafe { slice.align_to_mut() }) else {
        panic!()
    };
    ans
}
