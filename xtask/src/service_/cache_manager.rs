use llama_cu::{DistKVCache, SampleArgs, Session, SessionId, Terminal, utok};
use std::{collections::BTreeMap, iter::zip, time::Instant};

pub(crate) struct CacheManager {
    terminal: Terminal,
    caches: BTreeMap<Instant, (Vec<utok>, DistKVCache)>,
}

impl CacheManager {
    pub fn new(terminal: Terminal) -> Self {
        Self {
            terminal,
            caches: Default::default(),
        }
    }

    pub fn send(
        &mut self,
        id: SessionId,
        tokens: Vec<utok>,
        sample_args: SampleArgs,
    ) -> (SessionId, Vec<utok>) {
        let best_cache = self
            .caches
            .iter()
            .map(|(key, (history, _))| (*key, common_len(history, &tokens)))
            .max_by_key(|&(_, len)| len);

        let cache = match best_cache {
            Some((key, pos)) => {
                let (_, mut cache) = self.caches.remove(&key).unwrap();
                cache.pos = pos;
                cache
            }
            None => self.terminal.new_cache(),
        };
        let pos = cache.pos;
        self.terminal.start(
            Session {
                id,
                sample_args,
                cache,
            },
            &tokens[pos..],
        );
        (id, tokens)
    }

    pub fn insert(&mut self, tokens: Vec<utok>, cache: DistKVCache) {
        self.caches.insert(Instant::now(), (tokens, cache));
    }
}

fn common_len<T: Eq>(a: &[T], b: &[T]) -> usize {
    zip(a, b).take_while(|(a, b)| a == b).count()
}
