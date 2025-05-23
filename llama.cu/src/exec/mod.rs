mod engine;
mod group;
mod kv_cache;
mod model;
mod output_head;
mod step;

use crate::{memory::MemPages, utils::cast_slice_mut};
use kv_cache::KVCache;
use nn::Tensor;
use operators::{
    cuda::{ContextSpore, DevMemSpore, Device, EventSpore, Stream},
    random_sample::{KVPair, SampleArgs},
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use tokeneer::utok;

pub(crate) use engine::engine;

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
    pub cache: DistKVCache,
}

pub(crate) struct DistKVCache {
    pub parts: Arc<[Mutex<KVCache>]>,
    pub pos: usize,
    pub len: usize,
}

impl DistKVCache {
    pub fn new(template: &Tensor<usize, 2>, parts: &[(Device, usize)]) -> Self {
        let total = parts.iter().map(|(_, len)| len).sum::<usize>();
        let parts = parts
            .iter()
            .map(|(dev, len)| KVCache::new(template, *len, total, &MemPages::new(*dev)));
        Self {
            parts: parts.map(Mutex::new).collect(),
            pos: 0,
            len: template.shape()[0],
        }
    }
}

pub(crate) fn decode(
    output: Box<[(SessionId, usize)]>,
    kv_pair: DevMemSpore,
    event: EventSpore,
    stream: &Stream,
) -> BTreeMap<SessionId, Box<[utok]>> {
    let ctx = stream.ctx();
    let kv_pair = kv_pair.sprout(ctx);
    let mut host = ctx.malloc_host::<KVPair<()>>(kv_pair.len() / size_of::<KVPair<()>>());
    stream
        .wait_for(&event.sprout(ctx))
        .memcpy_d2h(&mut host, &kv_pair)
        .synchronize()
        .free(kv_pair);
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
