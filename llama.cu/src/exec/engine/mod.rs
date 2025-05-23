mod engine_manager;

use super::{
    Command, Output, Request, Session,
    group::{ModelGroup, Req},
    output_head::OutputHead,
};
use crate::{handle::Handle, op::FastEmbedding};
use engine_manager::{CommandReceiveError, EngineManager, Round};
use log::warn;
use nn::{Distribution, LLaMA, Tensor};
use operators::{
    Operator,
    attention_kv_cached::cuda::Operator as Attn,
    cuda::{ContextResource, CurrentCtx, Device, Gpu, HostMem},
    random_sample::{KVPair, cuda::Operator as Sample},
};
use std::{
    iter::zip,
    sync::{
        Arc, Barrier,
        mpsc::{Receiver, Sender},
    },
};
use tokeneer::utok;

struct SessionStub {
    session: Session,
    state: State,
    prompt: Option<Box<[utok]>>,
}

#[derive(Clone, Copy)]
struct State {
    seq: usize,
    out: usize,
}

impl Request {
    fn into_stub(self) -> SessionStub {
        let Request {
            session,
            prompt,
            out,
        } = self;
        SessionStub {
            session,
            state: State {
                seq: prompt.len(),
                out,
            },
            prompt: Some(prompt),
        }
    }
}

pub(crate) fn engine(
    mut llama: LLaMA<Tensor<&[u8], 2>>,
    dev: Device,
    dist: Distribution,
    barrier: Option<Arc<Barrier>>,
    commands: Receiver<Command>,
    outputs: Sender<Output>,
    use_cuda_graph: bool,
    handle: impl FnOnce(&CurrentCtx) -> Handle,
) {
    const NTOKS: &[usize] = &[1, 8, 32, 128, 512];

    let gpu = Gpu::new(dev.retain_primary(), Default::default());
    let attn = Attn::new(&gpu);
    let sample = Sample::new(&gpu);
    gpu.apply(|ctx| {
        let output_head = llama.output_head.take().unwrap();
        let output_head = OutputHead::new(output_head, sample, ctx);

        let mut handle = handle(ctx);
        let mut models = ModelGroup::new(
            llama,
            dist,
            attn,
            NTOKS.iter().copied(),
            &mut handle,
            barrier,
            use_cuda_graph,
        );

        let mut manager = EngineManager::default();
        let max_tok = *NTOKS.last().unwrap();
        let mut fast_embd = FastEmbedding::new(max_tok, ctx);
        let mut pre_kv_pairs = ctx.malloc::<KVPair<()>>(max_tok);
        let loading = ctx.stream();
        let stream = ctx.stream();
        loop {
            // 接收指令
            match manager.receive(&commands, &outputs) {
                Ok(()) => {}
                Err(CommandReceiveError::SendError) => return,
                Err(CommandReceiveError::ReceiveError) => {
                    warn!("command sender dropped");
                    break;
                }
            }
            // 组织请求
            let Round {
                overflow,
                tokens,
                reqs,
                sample,
                output,
                fast_map,
                no_decode,
            } = manager.prepare();
            if !overflow.is_empty() && outputs.send(Output::Overflow(overflow.into())).is_err() {
                return;
            }
            let out_idx = out_idx(&reqs, output.iter().map(|(_, len)| *len), ctx);
            // 推理
            let (key, tok) = models.load_inputs(&tokens, &reqs, &loading);
            fast_embd.launch(tok, &pre_kv_pairs, fast_map, &mut handle, &loading, &stream);
            let x = models.launch(key, &reqs, &mut handle, &stream);
            let kv_pairs = output_head.launch(x, out_idx, sample, &mut handle, &stream);
            stream.memcpy_d2d(&mut pre_kv_pairs[..kv_pairs.len()], &kv_pairs);
            let output = Output::Complete {
                output: output.into(),
                kv_pair: kv_pairs.sporulate(),
                event: stream.record().sporulate(),
                no_decode: no_decode.into(),
            };
            if outputs.send(output).is_err() {
                return;
            }
        }
        for stub in manager.into_stubs() {
            if outputs.send(Output::Removed(stub.session)).is_err() {
                return;
            }
        }
    })
}

fn out_idx<'ctx, T>(
    reqs: &[Req<T>],
    outs: impl IntoIterator<Item = usize>,
    ctx: &'ctx CurrentCtx,
) -> HostMem<'ctx> {
    let mut out_idx = Vec::<utok>::new();

    let mut itok = 0;
    for (req, out) in zip(reqs, outs) {
        for i in req.seq - out..req.seq {
            out_idx.push((itok + i) as _);
        }
        itok += req.seq
    }

    let mut ans = ctx.malloc_host::<utok>(out_idx.len());
    let ptr = out_idx.as_ptr().cast();
    let len = size_of_val(out_idx.as_slice());
    ans.copy_from_slice(unsafe { std::slice::from_raw_parts(ptr, len) });
    ans
}
