mod engine_manager;

use super::{
    Command, Output, Request, Session,
    group::{ModelGroup, Req},
    output_head::OutputHead,
};
use crate::{handle::Handle, op};
use engine_manager::{CommandReceiveError, EngineManager, Round};
use log::warn;
use nn::{Distribution, LLaMA, Tensor};
use operators::{
    Operator,
    attention_kv_cached::cuda::Operator as Attn,
    cuda::{ContextResource, CurrentCtx, DevByte, DevMem, Device, Gpu, HostMem, Stream},
    random_sample::{KVPair, SampleArgs, cuda::Operator as Sample},
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
        let mut buf = TokensBuffer::new(ctx, *NTOKS.last().unwrap());
        let mut pre_kv_pairs = ctx.malloc::<KVPair<()>>(0);
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
                output,
                fast_map,
                no_decode,
            } = manager.prepare();
            if !overflow.is_empty() && outputs.send(Output::Overflow(overflow.into())).is_err() {
                return;
            }
            let toks = buf.set(&tokens, models.padding(tokens.len()), &stream);
            let out_idx = out_idx(&reqs, output.iter().map(|(_, len)| *len), ctx);
            // 推理
            op::fast_embedding(&mut handle, toks, &pre_kv_pairs, fast_map, &stream);
            let x = models.launch(toks, &reqs, &mut handle, &stream);
            let kv_pairs = output_head.launch(
                x,
                out_idx,
                output
                    .iter()
                    .flat_map(|(_, out)| std::iter::repeat_n(SampleArgs::default(), *out)),
                &mut handle,
                &stream,
            );
            pre_kv_pairs = stream
                .free(pre_kv_pairs)
                .malloc::<KVPair<()>>(kv_pairs.len() / size_of::<KVPair<()>>());
            stream.memcpy_d2d(&mut pre_kv_pairs, &kv_pairs);
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

struct TokensBuffer<'ctx> {
    host: HostMem<'ctx>,
    dev: DevMem<'ctx>,
}

impl<'ctx> TokensBuffer<'ctx> {
    pub fn new(ctx: &'ctx CurrentCtx, max_tok: usize) -> Self {
        Self {
            host: ctx.malloc_host::<utok>(max_tok),
            dev: ctx.malloc::<utok>(max_tok),
        }
    }
}

impl TokensBuffer<'_> {
    pub fn set(&mut self, tokens: &[utok], padding: usize, stream: &Stream) -> &mut [DevByte] {
        let len = size_of_val(tokens);
        let padding = padding * size_of::<utok>();

        let (host, tail) = self.host[..padding].split_at_mut(len);
        host.copy_from_slice(unsafe { std::slice::from_raw_parts(tokens.as_ptr().cast(), len) });
        tail.fill(0);

        let src = &self.host[..padding];
        let dev = &mut self.dev[..padding];
        stream.memcpy_h2d(dev, src);
        dev
    }
}
