mod engine_manager;

use super::{
    Command, Output, Request, Session,
    group::{ModelGroup, Req},
    kv_cache::KVCache,
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
    nccl::CommunicatorGroup,
    random_sample::{KVPair, cuda::Operator as Sample},
};
use std::{
    ffi::c_int,
    iter::zip,
    num::NonZeroUsize,
    sync::{
        Arc, Barrier, Mutex,
        mpsc::{self, Receiver, Sender},
    },
};
use tokeneer::utok;

#[cfg(nccl)]
use operators::nccl::Communicator;

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

const NTOKS: &[usize] = &[1, 8, 32, 128, 512];

pub(crate) fn engine(
    mut llama: LLaMA<Tensor<&[u8], 2>>,
    gpus: &[c_int],
    commands: Receiver<Command>,
    outputs: Sender<Output>,
    use_cuda_graph: bool,
) {
    if let &[dev] = gpus {
        return mono(llama, Device::new(dev), commands, outputs, use_cuda_graph);
    }

    #[cfg(not(nccl))]
    unreachable!();

    #[cfg(nccl)]
    {
        let mut comms = CommunicatorGroup::new(gpus).into_vec().into_iter();
        let first = comms.next().unwrap();

        let output_head = llama.output_head.take().unwrap();
        let worker = Worker {
            dev: first.device(),
            dist: Distribution {
                start: 0,
                len: 1,
                total: gpus.len(),
            },
            ntoks: NTOKS,
            barrier: Some(Arc::new(Barrier::new(gpus.len()))),
            use_cuda_graph,
        };

        std::thread::scope(|s| {
            let mut senders = Vec::new();
            let _threads = comms
                .into_iter()
                .map(|comm| {
                    let dist = Distribution::new(comm.rank(), 1, gpus.len());
                    let worker = Worker {
                        dev: comm.device(),
                        dist,
                        ..worker.clone()
                    };
                    let llama = llama.clone();
                    let (sender, receiver) = mpsc::channel();
                    senders.push(sender);
                    s.spawn(move || worker.work(llama, comm, receiver))
                })
                .collect::<Vec<_>>();

            worker.lead(llama, output_head, commands, outputs, &senders, |ctx| {
                Handle::with_comm(ctx, first)
            })
        })
    }
}

fn mono(
    mut llama: LLaMA<Tensor<&[u8], 2>>,
    dev: Device,
    commands: Receiver<Command>,
    outputs: Sender<Output>,
    use_cuda_graph: bool,
) {
    let output_head = llama.output_head.take().unwrap();
    Worker {
        dev,
        dist: Distribution {
            start: 0,
            len: 1,
            total: 1,
        },
        ntoks: NTOKS,
        barrier: None,
        use_cuda_graph,
    }
    .lead(llama, output_head, commands, outputs, &[], |ctx| {
        Handle::new(ctx)
    })
}

#[derive(Clone)]
struct Worker<'a> {
    dev: Device,
    dist: Distribution,
    ntoks: &'a [usize],
    barrier: Option<Arc<Barrier>>,
    use_cuda_graph: bool,
}

impl Worker<'_> {
    fn lead(
        self,
        llama: LLaMA<Tensor<&[u8], 2>>,
        output_head: nn::OutputHead<Tensor<&[u8], 2>>,
        commands: Receiver<Command>,
        outputs: Sender<Output>,
        senders: &[Sender<(NonZeroUsize, Vec<Req<Arc<[Mutex<KVCache>]>>>)>],
        handle: impl FnOnce(&CurrentCtx) -> Handle,
    ) {
        let Self {
            dev,
            dist,
            ntoks,
            barrier,
            use_cuda_graph,
        } = self;

        let gpu = Gpu::new(dev.retain_primary(), Default::default());
        let attn = Attn::new(&gpu);
        let sample = Sample::new(&gpu);
        gpu.apply(|ctx| {
            let mut handle = handle(ctx);
            let mut models = ModelGroup::new(
                llama,
                dist,
                attn,
                ntoks.iter().copied(),
                &mut handle,
                barrier.as_deref(),
                use_cuda_graph,
            );

            let output_head = OutputHead::new(output_head, sample, ctx);

            let mut manager = EngineManager::default();
            let max_tok = *ntoks.last().unwrap();
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
                if !overflow.is_empty() && outputs.send(Output::Overflow(overflow.into())).is_err()
                {
                    return;
                }
                let out_idx = out_idx(&reqs, output.iter().map(|(_, len)| *len), ctx);
                // 加载输入
                let (key, tok) = models.load_toks(&tokens, &loading);
                // 快速启动路径
                fast_embd.launch(tok, &pre_kv_pairs, fast_map, &mut handle, &loading, &stream);
                // 通知协处理单元
                for sender in senders {
                    sender.send((key.clone(), reqs.clone())).unwrap()
                }
                #[cfg(nccl)]
                models.share_toks(key, &mut handle, &stream);
                // 推理
                let x = models.launch(key, &reqs, &mut handle, &stream);
                // 输出
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

    #[cfg(nccl)]
    fn work(
        self,
        llama: LLaMA<Tensor<&[u8], 2>>,
        comm: Communicator,
        receiver: Receiver<(NonZeroUsize, Vec<Req<Arc<[Mutex<KVCache>]>>>)>,
    ) {
        let Self {
            dev,
            dist,
            ntoks,
            barrier,
            use_cuda_graph,
        } = self;

        let gpu = Gpu::new(dev.retain_primary(), Default::default());
        let attn = Attn::new(&gpu);
        gpu.apply(|ctx| {
            let mut handle = Handle::with_comm(ctx, comm);
            let mut models = ModelGroup::new(
                llama,
                dist,
                attn,
                ntoks.iter().copied(),
                &mut handle,
                barrier.as_deref(),
                use_cuda_graph,
            );

            let stream = ctx.stream();
            for (key, reqs) in receiver {
                models.share_toks(key, &mut handle, &stream);
                models.launch(key, &reqs, &mut handle, &stream);
            }
        })
    }
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
