use super::{KVCache, ModelGroup};
use crate::{handle::Handle, memory::MemPages, utils::cast_slice_mut};
use log::warn;
use nn::{
    Dim, Distribution, Graph, GraphBuilder, LLaMA, NNGraph, Tensor, TensorMeta,
    digit_layout::types, op,
};
use operators::{
    Operator,
    attention_kv_cached::cuda::Operator as Attn,
    cuda::{
        ContextResource, ContextSpore, CurrentCtx, DevMemSpore, Device, EventSpore, Gpu, Stream,
    },
    random_sample::{KVPair, SampleArgs, cuda::Operator as Sample},
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Barrier,
        mpsc::{Receiver, Sender, TryRecvError},
    },
};
use tokeneer::utok;

pub(crate) enum Command {
    Insert(Request),
    Remove(SessionId),
}

pub(crate) enum Output {
    Overflow(Box<[Session]>),
    Removed(Session),
    Complete {
        output: Box<[(SessionId, usize)]>,
        kv_pair: DevMemSpore,
        event: EventSpore,
        no_decode: Box<[Session]>,
    },
}

pub(crate) struct Request {
    pub session: Session,
    pub prompt: Box<[utok]>,
    pub out: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(transparent)]
pub(crate) struct SessionId(pub usize);

pub(crate) struct Session {
    pub id: SessionId,
    pub sample_args: SampleArgs,
    pub cache: KVCache,
}

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

#[derive(Default)]
struct EngineManager(BTreeMap<SessionId, SessionStub>);

#[derive(Default)]
struct Round {
    overflow: Vec<Session>,
    tokens: Vec<utok>,
    reqs: Vec<super::group::Request>,
    output: Vec<(SessionId, usize)>,
    no_decode: Vec<Session>,
}

impl EngineManager {
    fn apply(&mut self, cmd: Command) -> Option<Session> {
        match cmd {
            Command::Insert(req) => {
                self.0.insert(req.session.id, req.into_stub());
                None
            }
            Command::Remove(id) => {
                // fmt
                self.0.remove(&id).map(|stub| stub.session)
            }
        }
    }

    /// 为所有 cache 充分扩容
    fn update(&mut self, pages: &mut MemPages) -> Round {
        let mut ans = Round::default();
        let sessions = std::mem::take(&mut self.0);
        for (id, mut stub) in sessions {
            let pos = stub.session.cache.pos();
            let seq = stub.state.seq;
            let out = stub.state.out;
            // 尝试填充缓存
            if !stub.session.cache.update(pos + seq, pages) {
                warn!("overflow {}", pos + seq);
                // 缓存溢出，不再推理
                ans.overflow.push(stub.session);
                continue;
            }
            // 填充推理信息
            ans.output.push((id, out));
            ans.reqs.push(super::group::Request {
                kv_cache: stub.session.cache.as_tensor().clone(),
                pos,
                seq,
                out,
                sample_args: stub.session.sample_args,
            });
            if let Some(prompt) = stub.prompt.take() {
                // prefill
                debug_assert_eq!(stub.state.seq, prompt.len());
                ans.tokens.extend(prompt);
                // TODO fast embd
                // if stub.state.out == 0 {
                // chunked prefill
                ans.no_decode.push(stub.session);
                continue;
                // }
            } else {
                // decode
                assert_eq!(stub.state.seq, 1);
                ans.tokens.push(0);
                todo!("fast embd")
            }
            // 回填
            // TODO fast embd
            // debug_assert!(self.0.insert(id, stub).is_none())
        }
        ans
    }
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
    let output_head = llama.output_head.take().unwrap();
    let NNGraph(Graph { topo, nodes, edges }) = builder()
        .build(
            llama.tensor_parallel(dist),
            [
                TensorMeta::new(types::U32, [Dim::var("n_tok")]),
                TensorMeta::new(types::U32, [Dim::var("n_tok")]),
            ],
        )
        .unwrap();

    // 权重加载
    let mut pages = MemPages::new(dev);
    let (_weight, edges) = pages.load_weight(edges);

    // 推理
    let graph = NNGraph(Graph { topo, nodes, edges });
    let gpu = Gpu::new(pages.dev().retain_primary(), Default::default());
    let attn = Attn::new(&gpu);
    let sample = Sample::new(&gpu);
    gpu.apply(|ctx| {
        let mut handle = handle(ctx);
        let mut models = ModelGroup::new(
            [1, 8, 32, 128, 512],
            &graph,
            output_head,
            attn,
            sample,
            &mut handle,
            &mut pages,
            barrier,
            use_cuda_graph,
        );

        let mut manager = EngineManager(Default::default());
        let stream = ctx.stream();
        'outer: loop {
            // 接收指令
            while manager.0.is_empty() {
                match commands.recv() {
                    Ok(cmd) => {
                        if let Some(session) = manager.apply(cmd) {
                            if outputs.send(Output::Removed(session)).is_err() {
                                return;
                            }
                        }
                    }
                    Err(_) => break 'outer,
                }
                loop {
                    match commands.try_recv() {
                        Ok(cmd) => {
                            if let Some(session) = manager.apply(cmd) {
                                if outputs.send(Output::Removed(session)).is_err() {
                                    return;
                                }
                            }
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => break 'outer,
                    }
                }
            }
            // 组织请求
            let Round {
                overflow,
                tokens,
                reqs,
                output,
                no_decode,
            } = manager.update(&mut pages);
            if !overflow.is_empty() && outputs.send(Output::Overflow(overflow.into())).is_err() {
                return;
            }
            // 推理
            let kv_pair = models.launch(&tokens, &reqs, &mut handle, &mut pages, &stream);
            let output = Output::Complete {
                output: output.into(),
                kv_pair: kv_pair.sporulate(),
                event: stream.record().sporulate(),
                no_decode: no_decode.into(),
            };
            if outputs.send(output).is_err() {
                return;
            }
        }
        for (_, stub) in manager.0 {
            if outputs.send(Output::Removed(stub.session)).is_err() {
                return;
            }
        }
    })
}

fn builder() -> GraphBuilder {
    let mut ans = GraphBuilder::default();
    ans.register_op("embedding", op::embedding::Embedding)
        .register_op("rms-norm", op::normalization::RmsNorm)
        .register_op("linear", op::linear::Linear)
        .register_op("rope", op::rope::Rope)
        .register_op("attention", op::attention::Attention)
        .register_op("swiglu", op::activation::SwiGLU)
        .register_op("concat", op::concat::Concat)
        .register_op("split", op::split::Split)
        .register_op("all-reduce", op::all_reduce::AllReduce);
    ans
}

pub(crate) fn decode(
    output: Box<[(SessionId, usize)]>,
    kv_pair: DevMemSpore,
    event: EventSpore,
    stream: &Stream,
) -> Box<[(SessionId, Box<[utok]>)]> {
    let ctx = stream.ctx();
    let kv_pair = kv_pair.sprout(ctx);
    let mut host = ctx.malloc_host::<KVPair<()>>(kv_pair.len() / size_of::<KVPair<()>>());
    stream.wait_for(&event.sprout(ctx));
    stream
        .memcpy_d2h(&mut host, &kv_pair)
        .free(kv_pair)
        .synchronize();
    let kv_pair: &mut [KVPair<()>] = cast_slice_mut(&mut host);
    let mut offset = 0;
    output
        .into_iter()
        .map(|(id, len)| {
            let slice = &kv_pair[offset..][..len];
            offset += len;
            (id, slice.iter().map(|kv| kv.idx() as _).collect())
        })
        .collect()
}
