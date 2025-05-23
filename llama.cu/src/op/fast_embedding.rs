use super::{Handle, ModuleKey, cuda_type};
use nn::digit_layout::{DigitLayout, types};
use operators::{
    cuda::{CurrentCtx, DevByte, DevMem, HostMem, Stream, params},
    random_sample::KVPair,
};
use std::{ffi::c_uint, ptr::copy_nonoverlapping};
use tokeneer::utok;

pub struct FastEmbedding<'ctx> {
    host: HostMem<'ctx>,
    dev: DevMem<'ctx>,
}

impl<'ctx> FastEmbedding<'ctx> {
    pub fn new(max_tok: usize, ctx: &'ctx CurrentCtx) -> Self {
        Self {
            host: ctx.malloc_host::<KVPair<utok>>(max_tok),
            dev: ctx.malloc::<KVPair<utok>>(max_tok),
        }
    }
}

impl FastEmbedding<'_> {
    pub fn launch(
        &mut self,
        tokens: &mut [DevByte],
        kv_pairs: &[DevByte],
        map: impl IntoIterator<Item = (usize, usize)>,
        handle: &mut Handle,
        loading: &Stream,
        stream: &Stream,
    ) {
        let map_host = map
            .into_iter()
            .map(|(k, v)| KVPair::new(k as _, v as utok))
            .collect::<Vec<_>>();
        if map_host.is_empty() {
            return;
        }
        // 拷贝参数
        let len = size_of_val(map_host.as_slice());
        unsafe { copy_nonoverlapping(map_host.as_ptr().cast(), self.host.as_mut_ptr(), len) };
        let event = loading
            .memcpy_h2d(&mut self.dev[..len], &self.host[..len])
            .record();
        // 编译内核
        let key = [ModuleKey::Text("fast-embedding")].into_iter();
        let module = handle.compile(key.collect(), || code(types::F16, types::U32));
        let kernel = module.get_kernel(c"fast_embedding");
        // 准备参数
        let params = params![tokens.as_mut_ptr(), kv_pairs.as_ptr(), self.dev.as_ptr()];
        // 启动内核
        stream.wait_for(&event).launch(
            &kernel,
            ((), map_host.len() as c_uint, 0),
            &params.to_ptrs(),
        );
    }
}

fn code(t_val: DigitLayout, t_idx: DigitLayout) -> String {
    const CODE: &str = include_str!("fast_embedding.cuh");
    let t_val = cuda_type(t_val);
    let t_idx = cuda_type(t_idx);

    format!(
        r#"{CODE}

extern "C" __global__ void fast_embedding(
    {t_idx} *__restrict__ tokens,
    KV<{t_val}> const *__restrict__ kv_pairs,
    KV<{t_idx}> const *__restrict__ map
){{
    kernel(tokens, kv_pairs, map);
}}"#
    )
}
