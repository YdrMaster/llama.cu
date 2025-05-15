pub mod bindings {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(unused)]
    #![allow(warnings)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

mod kv_pair;
pub use kv_pair::KVPair;

pub use bindings as cuda_bindings;

use crate::utils::{dims, offset_ptr, strides};
use nn::{Tensor, digit_layout::types as ty};
use operators::cuda::{AsRaw, Stream, VirByte};

#[derive(Clone, Copy, Debug)]
pub struct SampleArgs {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

pub fn calculate_workspace_size(n: usize) -> (usize, usize) {
    let mut argmax_size = 0;
    let mut sample_size = 0;
    let result = unsafe {
        cuda_bindings::calculate_workspace_size_half(&mut argmax_size, &mut sample_size, n)
    };
    if result != cuda_bindings::cudaError::cudaSuccess {
        panic!("calculate_workspace_size failed");
    }
    (argmax_size, sample_size)
}

pub fn argmax<const N: usize>(
    kv_pair: Tensor<*const VirByte, N>,
    logits: Tensor<*const VirByte, N>,
    stream: &Stream,
) {
    assert_eq!(kv_pair.dt(), KVPair::<()>::LAYOUT);
    assert_eq!(logits.dt(), ty::F16);

    dims!([] = kv_pair);
    dims!([n] = logits);

    strides!([sl] = logits);

    assert_eq!(sl, logits.dt().nbytes() as isize);

    let (argmax_size, _sample_size) = calculate_workspace_size(n);
    let mut workspace = stream.malloc::<u8>(argmax_size);
    unsafe {
        cuda_bindings::argmax_half(
            offset_ptr(&kv_pair) as *mut _,
            offset_ptr(&logits) as *const _,
            n,
            workspace.as_mut_ptr() as *mut _,
            workspace.len(),
            stream.as_raw() as *mut _,
        );
    }
}

pub fn sample<const N: usize>(
    kv_pair: Tensor<*const VirByte, N>,
    logits: Tensor<*const VirByte, N>,
    indices: Tensor<*const VirByte, N>,
    args: SampleArgs,
    seed: f32,
    stream: &Stream,
) {
    assert_eq!(kv_pair.dt(), KVPair::<()>::LAYOUT);
    assert_eq!(logits.dt(), ty::F16);
    assert_eq!(indices.dt(), ty::U32);

    dims!([] = kv_pair);
    dims!([n] = logits);
    dims!([n2] = indices);

    assert_eq!(n, n2);

    strides!([sl] = logits);
    strides!([si] = indices);
    assert_eq!(sl, logits.dt().nbytes() as isize);
    assert_eq!(si, indices.dt().nbytes() as isize);
    let (_argmax_size, sample_size) = calculate_workspace_size(n);
    let mut workspace = stream.malloc::<u8>(sample_size);
    unsafe {
        cuda_bindings::sample_half(
            offset_ptr(&kv_pair) as *mut _,
            offset_ptr(&logits) as *const _,
            offset_ptr(&indices) as *const _,
            n,
            seed,
            args.temperature,
            args.top_p,
            args.top_k,
            workspace.as_mut_ptr() as *mut _,
            workspace.len(),
            stream.as_raw() as *mut _,
        );
    }
}
