﻿use crate::utils::Data;
use ggus::{GENERAL_ALIGNMENT, GGuf, GGufError, GGufMetaDataValueType, GGufMetaKV, GGufMetaMap};
use nn::Tensor;
use std::{collections::HashMap, thread};

/// GGuf 模型，可能来自多个分片文件。
pub(crate) struct GGufModel<'a> {
    /// 元数据键值对。
    pub meta_kvs: HashMap<&'a str, GGufMetaKV<'a>>,
    /// 张量。
    pub tensors: HashMap<&'a str, Tensor<Data<'a>, 2>>,
}

impl<'a> GGufModel<'a> {
    /// 从多个分片文件中读取 GGuf 模型。
    pub fn read(files: impl IntoIterator<Item = &'a [u8]> + 'a) -> Self {
        let mut ans = Self {
            meta_kvs: Default::default(),
            tensors: Default::default(),
        };
        thread::scope(|s| {
            for (i, thread) in files
                .into_iter()
                .map(|data| s.spawn(|| GGuf::new(data)))
                .collect::<Vec<_>>()
                .into_iter()
                .enumerate()
            {
                thread
                    .join()
                    .unwrap()
                    .and_then(|gguf| ans.merge(gguf))
                    .unwrap_or_else(|e| panic!("Error at file {i}: {e}"));
            }
        });
        ans
    }

    fn merge(&mut self, gguf: GGuf<'a>) -> Result<(), GGufError> {
        for (k, kv) in gguf.meta_kvs {
            if k == GENERAL_ALIGNMENT || k.starts_with("split.") {
                continue;
            }
            if self.meta_kvs.insert(k, kv).is_some() {
                return Err(GGufError::DuplicateMetaKey(k.into()));
            }
        }

        for (name, t) in gguf.tensors {
            use std::collections::hash_map::Entry::{Occupied, Vacant};
            match self.tensors.entry(name) {
                Occupied(_) => return Err(GGufError::DuplicateTensorName(name.into())),
                Vacant(vacant) => {
                    let t = t.to_info();
                    let ty = t.ty().to_digit_layout();
                    let shape = t
                        .shape()
                        .iter()
                        .rev()
                        .map(|&x| x as usize)
                        .collect::<Vec<_>>();
                    vacant.insert(Tensor::from_dim_slice(ty, &*shape).map(|len| {
                        assert_eq!(len, t.nbytes());
                        gguf.data[t.offset()..][..t.nbytes()].into()
                    }));
                }
            }
        }

        Ok(())
    }
}

impl GGufMetaMap for GGufModel<'_> {
    fn get(&self, key: &str) -> Option<(GGufMetaDataValueType, &[u8])> {
        self.meta_kvs.get(key).map(|kv| (kv.ty(), kv.value_bytes()))
    }
}
