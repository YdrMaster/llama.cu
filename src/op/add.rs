use super::{Handle, ModuleKey, Operator, cuda_type, gcd};
use crate::{macros::*, offset_ptr};
use nn::Arg;
use operators::cuda::{Stream, VirByte, params};
use std::ffi::{c_int, c_uint};
use tensor::{Tensor, digit_layout::DigitLayout};

pub struct Add;

impl Operator for Add {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    ) {
        let Some(Arg::Float(k)) = arg else { panic!() };
        destruct!([y] = outputs);
        destruct!([x, b] = inputs);
        // 检查维度
        dims!([n, d] = y);
        dims!([n2, d2] = x);
        dims!([n3, d3] = b);

        assert_eq!(n, n2);
        assert_eq!(n, n3);
        assert_eq!(d, d2);
        assert_eq!(d, d3);
        // 检查类型
        let dt = y.dt();
        assert_eq!(x.dt(), dt);
        assert_eq!(b.dt(), dt);
        // 获取 stride
        strides!([sny, sdy] = y);
        strides!([snx, sdx] = x);
        strides!([snb, sdb] = b);
        // 获取最大线程数
        let max_threads_block = handle.ctx.dev().block_limit().max_threads;
        // 编译内核
        let key = [ModuleKey::Text("add2d"), ModuleKey::Type(dt)].into_iter();
        let module = handle.compile(key.collect(), || code(dt));
        let kernel = module.get_kernel(c"add");
        // 准备参数
        let unit = dt.nbytes() as isize;
        let params = params![
            offset_ptr(&y),
            (sny / unit) as c_int,
            (sdy / unit) as c_int,
            offset_ptr(&x),
            (snx / unit) as c_int,
            (sdx / unit) as c_int,
            offset_ptr(&b),
            (snb / unit) as c_int,
            (sdb / unit) as c_int,
            k as f32
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
    const CODE: &str = include_str!("add.cuh");
    let dt = cuda_type(dt);

    format!(
        r#"{CODE}

extern "C" __global__ void add(
    {dt} *__restrict__ y,
    int const sny,
    int const sdy,
    {dt} const *__restrict__ x,
    int const snx,
    int const sdx,
    {dt} const *__restrict__ b,
    int const snb,
    int const sdb,
    float k
){{
    kernel(y, sny, sdy,
           x, snx, sdx,
           b, snb, sdb,
           k);
}}"#
    )
}
