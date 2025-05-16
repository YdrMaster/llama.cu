use super::{model::ModelExec, output_head::OutputHead};
use crate::{handle::Handle, memory::MemPages};
use nn::{NNGraph, Tensor};
use operators::{
    attention_kv_cached::cuda::Operator as Attn,
    cuda::{DevMem, Stream, VirByte},
    random_sample::{SampleArgs, cuda::Operator as Sample},
};
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    sync::{Arc, Barrier},
};
use tokeneer::utok;

#[derive(Clone)]
pub(crate) struct Request {
    pub kv_cache: Tensor<*const VirByte, 2>,
    pub pos: usize,
    pub seq: usize,
    pub out: usize,
    pub sample_args: SampleArgs,
}

pub(crate) struct ModelGroup<'ctx> {
    models: BTreeMap<NonZeroUsize, ModelExec<'ctx>>,
    mapped: Option<NonZeroUsize>,
    attn: Attn,
    output_head: OutputHead<'ctx>,
}

impl<'ctx> ModelGroup<'ctx> {
    pub fn new(
        n_toks: impl IntoIterator<Item = usize>,
        graph: &NNGraph<Tensor<*const VirByte, 2>>,
        output_head: nn::OutputHead<Tensor<&[u8], 2>>,
        attn: Attn,
        sample: Sample,
        handle: &mut Handle<'ctx>,
        pages: &mut MemPages,
        barrier: Option<Arc<Barrier>>,
    ) -> Self {
        Self {
            models: n_toks
                .into_iter()
                .map(|n_tok| {
                    (
                        NonZeroUsize::new(n_tok).unwrap(),
                        ModelExec::new(graph.clone(), n_tok, handle, pages, barrier.clone()),
                    )
                })
                .collect(),
            mapped: None,
            attn,
            output_head: OutputHead::new(output_head, sample, handle.ctx),
        }
    }

    pub fn launch(
        &mut self,
        toks: &[utok],
        reqs: &[Request],
        handle: &mut Handle,
        pages: &mut MemPages,
        stream: &Stream<'ctx>,
    ) -> DevMem<'ctx> {
        let Self {
            models,
            mapped,
            attn,
            output_head,
        } = self;

        let Some(len) = NonZeroUsize::new(toks.len()) else {
            return stream.malloc::<u8>(0);
        };

        let (&key, _) = models.range_mut(len..).next().unwrap();
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
        model.launch(attn, handle, output_head, toks, reqs, stream)
    }
}
