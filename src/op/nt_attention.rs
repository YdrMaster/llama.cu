use crate::offset_ptr;
use nn::Tensor;
use operators::cuda::{AsRaw, Stream, VirByte, bindings::CUresult::CUDA_SUCCESS};

#[allow(clippy::needless_borrow)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[allow(unused)]
pub fn nt_attention<const N: usize>(
    q: &Tensor<*const VirByte, N>,
    k: &Tensor<*const VirByte, N>,
    v: &Tensor<*const VirByte, N>,
    k_cache: &Tensor<*const VirByte, N>,
    k_cache_end: &Tensor<*const VirByte, N>,
    v_cache: &Tensor<*const VirByte, N>,
    v_cache_end: &Tensor<*const VirByte, N>,
    mask: &Tensor<*const VirByte, N>,
    o: &Tensor<*const VirByte, N>,
    stream: &Stream,
) {
    let q = NTTBuf::new(q);
    let k = NTTBuf::new(k);
    let v = NTTBuf::new(v);
    let k_cache = NTTBuf::new(k_cache);
    let k_cache_end = NTTBuf::new(k_cache_end);
    let v_cache = NTTBuf::new(v_cache);
    let v_cache_end = NTTBuf::new(v_cache_end);
    let mask = NTTBuf::new(mask);
    let o = NTTBuf::new(o);
    let result = unsafe {
        bindings::launch_attention_kv_nh_64(
            stream.as_raw().cast(),
            q.to_ntt(),
            k.to_ntt(),
            v.to_ntt(),
            k_cache.to_ntt(),
            k_cache_end.to_ntt(),
            v_cache.to_ntt(),
            v_cache_end.to_ntt(),
            mask.to_ntt(),
            o.to_ntt(),
        )
    };
    if result != CUDA_SUCCESS as i32 {
        panic!("launch_attention_kv failed: {result}")
    }
}

struct NTTBuf {
    data: *const VirByte,
    shape: Box<[u64]>,
    strides: Box<[i64]>,
}

impl NTTBuf {
    fn new<const N: usize>(t: &Tensor<*const VirByte, N>) -> Self {
        let unit = t.dt().nbytes() as isize;
        Self {
            data: offset_ptr(t),
            shape: t.shape().iter().map(|&d| d as _).collect(),
            strides: t.strides().iter().map(|&s| (s / unit) as _).collect(),
        }
    }
}

impl NTTBuf {
    fn to_ntt(&self) -> bindings::NineToothedTensor {
        bindings::NineToothedTensor {
            data: self.data.cast_mut().cast(),
            shape: self.shape.as_ptr().cast_mut(),
            strides: self.strides.as_ptr().cast_mut(),
        }
    }
}
