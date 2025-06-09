use super::{KVCache, model::ModelExec};
use crate::{exec::upos, handle::Handle, memory::MemPages};
use log::debug;
use nn::{
    Distribution, Graph, GraphBuilder, LLaMA, NNGraph, Tensor, TensorMeta, digit_layout::types, op,
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
        // 构建计算图
        let NNGraph(Graph { topo, nodes, edges }) = builder()
            .build(
                llama.tensor_parallel(dist),
                [
                    TensorMeta::new(types::U32, ["n_tok".into()]),
                    TensorMeta::new(types::U32, ["n_tok".into()]),
                ],
            )
            .unwrap();
        // 加载权重
        let dev = handle.ctx.dev();
        let mut pages = MemPages::new(dev);
        let (_weight, edges) = pages.load_weight(&dev, edges);
        // 构建 cuda graph
        let graph = NNGraph(Graph { topo, nodes, edges });
        debug!("compiling model group @{}", dev.index());
        let time = Instant::now();
        let models = n_toks
            .into_iter()
            .map(|n_tok| {
                if let Some(b) = barrier {
                    b.wait();
                }
                let key = NonZeroUsize::new(n_tok).unwrap();
                let exec = ModelExec::new(graph.clone(), n_tok, handle, &mut pages, use_cuda_graph);
                (key, exec)
            })
            .collect::<BTreeMap<_, _>>();
        debug!(
            "group ({} models) compiled @{} in {:.02?}",
            models.len(),
            dev.index(),
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

    pub fn load_inputs(
        &mut self,
        len: usize,
        tok: &[utok],
        pos: &[upos],
        stream: &Stream,
    ) -> (NonZeroUsize, &mut [DevByte]) {
        let (&key, _) = self
            .models
            .range(NonZeroUsize::new(len).unwrap()..)
            .next()
            .unwrap();
        self.map_exec(key, stream);

        let model = self.models.get_mut(&key).unwrap();
        stream.memcpy_h2d(model.tok_buf(), &tok[..key.get()]);
        stream.memcpy_h2d(model.pos_buf(), &pos[..key.get()]);
        (key, model.tok_buf())
    }

    #[cfg(nccl)]
    pub fn share_toks(&mut self, key: NonZeroUsize, handle: &mut Handle, stream: &Stream<'ctx>) {
        self.map_exec(key, stream);
        if let Some(comm) = &handle.comm {
            let toks = self.models.get_mut(&key).unwrap().tok_buf();
            comm.broadcast(toks, None, 0, stream)
        }
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
                req.kv_cache.update(req.pos + req.seq, pages);
                Req {
                    kv_cache: req.kv_cache.as_tensor(),
                    pos: req.pos,
                    seq: req.seq,
                }
            })
            .collect::<Vec<_>>();

        models
            .get_mut(&key)
            .unwrap()
            .launch(attn, handle, &reqs, stream)
    }

    /// 为组中的指定执行模型映射物理页
    fn map_exec(&mut self, key: NonZero<usize>, stream: &Stream) {
        let Self {
            models,
            mapped,
            pages,
            ..
        } = self;
        // 检查当前映射的模型
        if let Some(mapped) = mapped {
            if *mapped == key {
                return;
            }
            // 当前映射的模型不是要映射的模型，解映射
            stream.synchronize();
            models.get_mut(mapped).unwrap().unmap(pages)
        }
        // 建立映射
        models.get_mut(&key).unwrap().map(pages);
        // 更新记录
        *mapped = Some(key)
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
