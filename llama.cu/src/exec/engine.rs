use super::{
    Command, Output, Request, Session,
    engine_manager::{EngineManager, Round},
    group::{ModelGroup, Req},
    kv_cache::KVCache,
    output_head::OutputHead,
};
use crate::{
    handle::Handle,
    op::{FastEmbedding, random_sample::KVPair},
};
use nn::{Distribution, LLaMA, Tensor};
use operators::{
    Operator,
    attention_kv_cached::cuda::Operator as Attn,
    cuda::{ContextResource, CurrentCtx, Device, Gpu, HostMem},
};
use std::{
    ffi::c_int,
    iter::zip,
    num::NonZeroUsize,
    sync::{
        Arc, Barrier, Mutex, RwLock,
        mpsc::{Receiver, Sender},
    },
};
use tokeneer::utok;

#[cfg(nccl)]
use operators::nccl::{Communicator, CommunicatorGroup};

pub(super) struct SessionStub {
    pub session: Session,
    pub state: State,
    pub prompt: Option<Box<[utok]>>,
}

#[derive(Clone, Copy)]
pub(super) struct State {
    pub seq: usize,
    pub out: usize,
    pub remain_steps: usize,
}

impl Request {
    pub(super) fn into_stub(self) -> SessionStub {
        let Request {
            session,
            prompt,
            out,
            max_steps,
        } = self;
        SessionStub {
            session,
            state: State {
                seq: prompt.len(),
                out,
                remain_steps: max_steps,
            },
            prompt: Some(prompt),
        }
    }
}

const NTOKS: &[usize] = &[1, 8, 32, 64, 128, 256, 1024];

pub(crate) fn engine(
    llama: LLaMA<Tensor<&[u8], 2>>,
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

        let mut llama = llama;
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
            task_box: Default::default(),
            use_cuda_graph,
        };

        std::thread::scope(|s| {
            let _threads = comms
                .map(|comm| {
                    let dist = Distribution::new(comm.rank(), 1, gpus.len());
                    let worker = Worker {
                        dev: comm.device(),
                        dist,
                        ..worker.clone()
                    };
                    let llama = llama.clone();
                    s.spawn(move || worker.work(llama, comm))
                })
                .collect::<Vec<_>>();

            worker.lead(llama, output_head, commands, outputs, |ctx| {
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
        task_box: Default::default(),
        use_cuda_graph,
    }
    .lead(llama, output_head, commands, outputs, |ctx| {
        Handle::new(ctx)
    })
}

#[derive(Clone)]
struct Worker<'a> {
    dev: Device,
    dist: Distribution,
    ntoks: &'a [usize],
    barrier: Option<Arc<Barrier>>,
    task_box: TaskBox,
    use_cuda_graph: bool,
}

type TaskBox = Arc<RwLock<Option<Task>>>;

#[cfg_attr(not(nccl), allow(dead_code))]
struct Task {
    key: NonZeroUsize,
    reqs: Vec<Req<Arc<[Mutex<KVCache>]>>>,
}

impl Worker<'_> {
    fn lead(
        self,
        llama: LLaMA<Tensor<&[u8], 2>>,
        output_head: nn::OutputHead<Tensor<&[u8], 2>>,
        commands: Receiver<Command>,
        outputs: Sender<Output>,
        handle: impl FnOnce(&CurrentCtx) -> Handle,
    ) {
        let Self {
            dev,
            dist,
            ntoks,
            barrier,
            task_box,
            use_cuda_graph,
        } = self;

        let gpu = Gpu::new(dev.retain_primary(), Default::default());
        let attn = Attn::new(&gpu);
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

            let mut output_head = OutputHead::new(output_head, ctx);

            let mut manager = EngineManager::default();
            let max_tok = *ntoks.last().unwrap();
            let mut fast_embd = FastEmbedding::new(max_tok, ctx);
            let mut pre_kv_pairs = ctx.malloc::<KVPair>(max_tok);
            let loading = ctx.stream();
            let stream = ctx.stream();
            if outputs.send(Output::Ready).is_ok() {
                let mut out_idx_buf = ctx.malloc_host::<utok>(1);
                while manager.receive(&commands, &outputs).is_ok() {
                    // 组织请求
                    let Round {
                        overflow,
                        tokens,
                        reqs,
                        sample,
                        output,
                        fast_map,
                        finished,
                    } = manager.prepare();
                    if !overflow.is_empty()
                        && outputs.send(Output::Overflow(overflow.into())).is_err()
                    {
                        break;
                    }
                    if tokens.is_empty() {
                        assert!(
                            reqs.is_empty()
                                && sample.is_empty()
                                && output.is_empty()
                                && fast_map.is_empty()
                                && finished.is_empty()
                        );
                        continue;
                    }
                    stream.synchronize(); // TODO: Remove this line
                    out_idx(
                        &reqs,
                        output.iter().map(|(_, len)| *len),
                        &mut out_idx_buf,
                        ctx,
                    );
                    // 加载输入
                    let (key, tok) = models.load_toks(&tokens, &loading, &stream);
                    // 快速启动路径
                    fast_embd.launch(tok, &pre_kv_pairs, fast_map, &mut handle, &loading, &stream);
                    // 通知协处理单元
                    #[cfg(nccl)]
                    if let Some(barrier) = &barrier {
                        *task_box.write().unwrap() = Some(Task {
                            key,
                            reqs: reqs.clone(),
                        });
                        barrier.wait();
                        models.share_toks(key, &mut handle, &stream);
                    }
                    // 推理
                    let x = models.launch(key, &reqs, &mut handle, &stream);
                    // 输出
                    let kv_pairs =
                        output_head.launch(x, &out_idx_buf, sample, &mut handle, &stream);
                    stream.memcpy_d2d(&mut pre_kv_pairs[..kv_pairs.len()], &kv_pairs);
                    let output = Output::Complete {
                        output: output.into(),
                        kv_pair: kv_pairs.sporulate(),
                        event: stream.record().sporulate(),
                        finished: finished.into(),
                    };
                    if outputs.send(output).is_err() {
                        break;
                    }
                }
            }
            // 通知协处理单元退出
            if let Some(barrier) = &barrier {
                let _ = task_box.write().unwrap().take();
                barrier.wait();
            }
            // 送回存储的会话信息
            for stub in manager.into_stubs() {
                if outputs.send(Output::Removed(stub.session)).is_err() {
                    break;
                }
            }
        })
    }

    #[cfg(nccl)]
    fn work(self, llama: LLaMA<Tensor<&[u8], 2>>, comm: Communicator) {
        let Self {
            dev,
            dist,
            ntoks,
            barrier,
            task_box,
            use_cuda_graph,
        } = self;

        let gpu = Gpu::new(dev.retain_primary(), Default::default());
        let attn = Attn::new(&gpu);
        let barrier = barrier.unwrap();
        gpu.apply(|ctx| {
            let mut handle = Handle::with_comm(ctx, comm);
            let mut models = ModelGroup::new(
                llama,
                dist,
                attn,
                ntoks.iter().copied(),
                &mut handle,
                Some(&barrier),
                use_cuda_graph,
            );

            let stream = ctx.stream();
            loop {
                barrier.wait();
                match &*task_box.read().unwrap() {
                    Some(Task { key, reqs }) => {
                        models.share_toks(*key, &mut handle, &stream);
                        models.launch(*key, reqs, &mut handle, &stream);
                    }
                    None => break,
                }
            }
        })
    }
}

fn out_idx<'ctx, T>(
    reqs: &[Req<T>],
    outs: impl IntoIterator<Item = usize>,
    buf: &mut HostMem<'ctx>,
    ctx: &'ctx CurrentCtx,
) {
    let mut out_idx = Vec::<utok>::new();

    let mut itok = 0;
    for (req, out) in zip(reqs, outs) {
        for i in req.seq - out..req.seq {
            out_idx.push((itok + i) as _)
        }
        itok += req.seq
    }

    let ptr = out_idx.as_ptr().cast();
    let len = size_of_val(out_idx.as_slice());
    if buf.len() != len {
        *buf = ctx.malloc_host::<utok>(out_idx.len())
    }
    buf.copy_from_slice(unsafe { std::slice::from_raw_parts(ptr, len) })
}
