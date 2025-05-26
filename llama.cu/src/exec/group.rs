use super::{KVCache, model::ModelExec};
use crate::{handle::Handle, memory::MemPages};
use log::debug;
use nn::{
    Dim, Distribution, Graph, GraphBuilder, LLaMA, NNGraph, Tensor, TensorMeta,
    digit_layout::types, op,
};
use operators::{
    attention_kv_cached::cuda::Operator as Attn,
    cuda::{DevByte, Stream, VirByte, VirMem},
};
use std::{
    collections::BTreeMap,
    num::{NonZero, NonZeroUsize},
    sync::{Arc, Barrier, Mutex},
    time::Instant,
};
use tokeneer::utok;

#[derive(Clone)]
pub(crate) struct Req<Cache> {
    pub kv_cache: Cache,
    pub pos: usize,
    pub seq: usize,
}

pub(crate) struct ModelGroup<'ctx> {
    models: BTreeMap<NonZeroUsize, ModelExec<'ctx>>,
    mapped: Option<NonZeroUsize>,
    attn: Attn,
    pages: MemPages,
    _weight: VirMem,
}

impl<'ctx> ModelGroup<'ctx> {
    pub fn new(
        llama: LLaMA<Tensor<&[u8], 2>>,
        dist: Distribution,

        attn: Attn,
        n_toks: impl IntoIterator<Item = usize>,
        handle: &mut Handle<'ctx>,
        barrier: Option<&Barrier>,
        use_cuda_graph: bool,
    ) -> Self {
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
        let dev = handle.ctx.dev();
        let mut pages = MemPages::new(dev);
        let (_weight, edges) = pages.load_weight(&dev, edges);

        // 推理
        let graph = NNGraph(Graph { topo, nodes, edges });

        let idev = dev.index();
        debug!("compiling model group @{idev}");
        let time = Instant::now();
        let models = n_toks
            .into_iter()
            .map(|n_tok| {
                if let Some(b) = barrier {
                    b.wait();
                }
                (
                    NonZeroUsize::new(n_tok).unwrap(),
                    ModelExec::new(graph.clone(), n_tok, handle, &mut pages, use_cuda_graph),
                )
            })
            .collect::<BTreeMap<_, _>>();
        debug!(
            "group ({} models) compiled @{idev} in {:.02?}",
            models.len(),
            time.elapsed(),
        );
        Self {
            models,
            mapped: None,
            attn,
            pages,
            _weight,
        }
    }

    pub fn padding(&self, len: usize) -> usize {
        let len = NonZeroUsize::new(len).unwrap();
        let (ans, _) = self.models.range(len..).next().unwrap();
        ans.get()
    }

    pub fn map_memory(&mut self, key: NonZero<usize>) {
        let Self {
            models,
            mapped,
            pages,
            ..
        } = self;

        let map = match mapped {
            Some(mapped) => {
                if *mapped != key {
                    models.get_mut(mapped).unwrap().unmap(pages);
                    true
                } else {
                    false
                }
            }
            None => true,
        };
        let model = models.get_mut(&key).unwrap();
        if map {
            *mapped = Some(key);
            model.map(pages)
        }
    }

    pub fn load_toks(&mut self, toks: &[utok], loading: &Stream) -> (NonZeroUsize, &mut [DevByte]) {
        let key = NonZeroUsize::new(self.padding(toks.len())).unwrap();
        self.map_memory(key);
        let tok = self
            .models
            .get_mut(&key)
            .unwrap()
            .load_toks_host(toks, loading);
        (key, tok)
    }

    pub fn load_toks_buf(&mut self, key: NonZeroUsize) -> &mut [DevByte] {
        self.models.get_mut(&key).unwrap().toks_buf()
    }

    pub fn launch(
        &mut self,
        key: NonZeroUsize,
        reqs: &[Req<Arc<[Mutex<KVCache>]>>],
        handle: &mut Handle,
        stream: &Stream<'ctx>,
    ) -> Tensor<*const VirByte, 2> {
        let Self {
            models,
            attn,
            pages,
            ..
        } = self;

        let mut reqs = reqs
            .iter()
            .map(|req| Req {
                kv_cache: req.kv_cache[handle.rank()].lock().unwrap(),
                pos: req.pos,
                seq: req.seq,
            })
            .collect::<Vec<_>>();
        let reqs = reqs
            .iter_mut()
            .map(|req| {
                req.kv_cache.update(req.pos, req.pos + req.seq, pages);
                Req {
                    kv_cache: req.kv_cache.as_tensor(),
                    pos: req.pos,
                    seq: req.seq,
                }
            })
            .collect::<Vec<_>>();

        let model = models.get_mut(&key).unwrap();
        model.load_pos(&reqs, stream);
        model.launch(attn, handle, &reqs, stream)
    }
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
