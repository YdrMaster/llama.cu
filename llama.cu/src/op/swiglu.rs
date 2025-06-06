use super::{Handle, ModuleKey, Operator, cuda_type, gcd};
use crate::utils::{destruct, dims, offset_ptr, strides};
use nn::{Tensor, digit_layout::DigitLayout};
use operators::cuda::{Stream, VirByte, params};
use std::ffi::{c_int, c_uint};

pub struct Swiglu;

impl Operator for Swiglu {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    ) {
        assert!(arg.is_none());

        destruct!([gate, up] = inputs);
        destruct!([out] = outputs);

        // 检查维度
        dims!([n, d] = gate);
        dims!([n2, d2] = up);
        dims!([n3, d3] = out);

        assert_eq!(n, n2);
        assert_eq!(n, n3);
        assert_eq!(d, d2);
        assert_eq!(d, d3);

        // 检查类型
        let dt = gate.dt();
        assert_eq!(up.dt(), dt);
        assert_eq!(out.dt(), dt);

        // 获取stride
        strides!([s_n_gate, s_d_gate] = gate);
        strides!([s_n_up, s_d_up] = up);
        strides!([s_n_out, s_d_out] = out);

        // 确保stride符合期望
        let unit = dt.nbytes() as isize;
        assert_eq!(s_d_gate, unit);
        assert_eq!(s_d_up, unit);
        assert_eq!(s_d_out, unit);

        // 获取最大线程数
        let max_threads_block = handle.ctx.dev().block_limit().max_threads;

        // 编译内核
        let key = [ModuleKey::Text("swiglu"), ModuleKey::Type(dt)].into_iter();
        let module = handle.compile(key.collect(), || code(dt));
        let kernel = module.get_kernel(c"swiglu");

        // 准备参数
        let params = params![
            offset_ptr(&out),
            (s_n_out / unit) as c_int,
            offset_ptr(&gate),
            (s_n_gate / unit) as c_int,
            offset_ptr(&up),
            (s_n_up / unit) as c_int
        ];

        // 计算线程块配置
        let block = gcd(max_threads_block, d);

        // 启动内核
        stream.launch(
            &kernel,
            ((n as c_uint, (d / block) as c_uint), block as c_uint, 0),
            &params.to_ptrs(),
        );
    }
}

fn code(dt: DigitLayout) -> String {
    const CODE: &str = include_str!("swiglu.cuh");
    let dt = cuda_type(dt);

    format!(
        r#"{CODE}

extern "C" __global__ void swiglu(
    {dt} *__restrict__ out,
    int const stride_out,
    {dt} const *__restrict__ gate,
    int const stride_gate,
    {dt} const *__restrict__ up,
    int const stride_up
){{
    kernel(out, stride_out, gate, stride_gate, up, stride_up);
}}"#
    )
}
