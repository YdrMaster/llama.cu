use super::{group::Request, output_head::OutputHead, step::Step};
use crate::{
    handle::Handle,
    memory::MemPages,
    upos,
    utils::{self, cast_slice_mut, destruct},
};
use bytesize::ByteSize;
use log::trace;
use nn::{NNGraph, Tensor};
use operators::{
    attention_kv_cached::cuda::Operator as Attn,
    cuda::{CurrentCtx, DevMem, HostMem, Stream, VirByte, VirMem, memcpy_h2d},
};
use std::{
    iter::zip,
    sync::{Arc, Barrier},
    time::Instant,
};
use tokeneer::utok;

pub(super) struct ModelExec<'ctx> {
    n_tok: usize,
    execs: Box<[Step<'ctx>]>,
    workspace: VirMem,
    inputs: Box<[Tensor<*const VirByte, 2>]>,
    outputs: Box<[Tensor<*const VirByte, 2>]>,
    barrier: Option<Arc<Barrier>>,
}

impl<'ctx> ModelExec<'ctx> {
    pub fn new(
        graph: NNGraph<Tensor<*const VirByte, 2>>,
        n_tok: usize,
        handle: &mut Handle<'ctx>,
        pages: &mut MemPages,
        barrier: Option<Arc<Barrier>>,
        use_cuda_graph: bool,
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
        let time = Instant::now();
        if let Some(barrier) = &barrier {
            barrier.wait();
        }
        let execs = handle.build_steps(exec, use_cuda_graph);
        trace!(
            "model compiled @{} in {:.2?}, seq len = {n_tok}, workspace = {}",
            pages.dev().index(),
            time.elapsed(),
            ByteSize::b(workspace.len() as _).display(),
        );

        // 解除映射回收物理页
        pages.unmap(&mut workspace, ..);

        Self {
            n_tok,
            execs,
            workspace,
            inputs,
            outputs,
            barrier,
        }
    }
}

impl ModelExec<'_> {
    /// 映射虚页
    pub fn map(&mut self, pages: &mut MemPages) {
        pages.map(&mut self.workspace, ..)
    }

    /// 解映射虚页
    pub fn unmap(&mut self, pages: &mut MemPages) {
        pages.unmap(&mut self.workspace, ..)
    }

    pub fn launch<'ctx>(
        &mut self,
        attn: &Attn,
        handle: &mut Handle,
        output_head: &mut OutputHead,
        toks: &[utok],
        reqs: &[Request],
        stream: &Stream<'ctx>,
    ) -> DevMem<'ctx> {
        // 初始化输入
        let (n_out, [padding, pos, out_idx]) = fill_inputs(self.n_tok, toks, reqs, stream.ctx());

        // 拷贝到硬件
        for (input, data) in zip(&self.inputs, [padding, pos]) {
            let ptr = input.get().cast_mut();
            memcpy_h2d(
                unsafe { std::slice::from_raw_parts_mut(ptr.cast(), data.len()) },
                &data,
            )
        }
        // 执行
        if let Some(barrier) = &self.barrier {
            barrier.wait();
        }
        for exec in &self.execs {
            match exec {
                Step::Graph(graph, stub) => {
                    stream.launch_graph(graph);
                    if !stub.is_empty() {
                        for t in stub {
                            utils::fmt(t, stream.ctx())
                        }
                        std::process::exit(0);
                    }
                }
                Step::Attention(box_) => handle.launch_attn(attn, box_, reqs, stream),
                Step::Exec(exec) => handle.launch_nn_exec(exec, stream),
            }
        }
        destruct!([x] = self.outputs.clone());
        output_head.launch(
            x,
            out_idx,
            n_out,
            reqs.iter()
                .flat_map(|req| std::iter::repeat_n(req.sample_args, req.out)),
            handle,
            stream,
        )
    }
}

/// 构造输入数据
fn fill_inputs<'ctx>(
    padding: usize,
    toks: &[utok],
    reqs: &[Request],
    ctx: &'ctx CurrentCtx,
) -> (usize, [HostMem<'ctx>; 3]) {
    let mut ans0 = ctx.malloc_host::<utok>(padding);
    let mut ans1 = ctx.malloc_host::<upos>(padding);
    let mut ans2 = ctx.malloc_host::<utok>(padding);

    let tokens: &mut [utok] = cast_slice_mut(&mut ans0);
    let pos: &mut [upos] = cast_slice_mut(&mut ans1);
    let out_idx: &mut [utok] = cast_slice_mut(&mut ans2);
    let mut itok = 0;
    let mut iout = 0;
    for req in reqs {
        for i in 0..req.seq {
            if i >= req.seq - req.out {
                out_idx[iout] = itok as _;
                iout += 1
            }

            pos[itok] = (req.pos + i) as _;
            itok += 1
        }
    }
    tokens[..itok].copy_from_slice(toks);
    tokens[itok..].fill(0);
    pos[itok..].fill(0);

    (iout, [ans0, ans1, ans2])
}
