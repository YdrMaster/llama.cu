﻿use super::{Handle, Operator, add::Add};
use crate::utils::{destruct, dims, offset_ptr};
use ggus::ggml_quants::f16;
use nn::{
    Arg, Tensor,
    digit_layout::{DigitLayout, types},
    ndarray_layout::ArrayLayout,
};
use operators::{
    cublas::GemmScheme,
    cuda::{Stream, VirByte},
};
use std::mem::swap;

pub struct Linear;

impl Operator for Linear {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    ) {
        destruct!([y] = outputs);
        let mut inputs = inputs.into_iter();
        let x = inputs.next().unwrap();
        let Some(Arg::Bool(residual)) = arg else {
            panic!()
        };

        let dt = y.dt();
        assert_eq!(x.dt(), dt);

        let (w, beta) = if residual {
            // 残差连接
            let residual = inputs.next().unwrap();
            {
                assert!(y.is_contiguous());
                assert!(residual.is_contiguous());
                let len = residual.shape().iter().fold(dt.nbytes(), |acc, d| acc * d);
                stream.memcpy_d2d(
                    unsafe { std::slice::from_raw_parts_mut(*y.get() as _, len) },
                    unsafe { std::slice::from_raw_parts(*residual.get() as _, len) },
                );
            }

            let w = inputs.next().unwrap();
            dims!([n, _] = y);
            dims!([d, _] = w);
            if let Some(b) = inputs.next() {
                let b = b.transform(|layout| layout.tile_be(0, &[1, d]).broadcast(0, n));
                // y = y + b
                Add::launch(
                    handle,
                    Some(Arg::Float(1.)),
                    [y.clone(), b],
                    [y.clone()],
                    stream,
                );
            }
            (w, 1.)
        } else {
            let w = inputs.next().unwrap();
            dims!([n, _] = y);
            dims!([d, _] = w);
            if let Some(b) = inputs.next() {
                let b = b.transform(|layout| layout.tile_be(0, &[1, d]).broadcast(0, n));
                // y = y + b
                Add::launch(
                    handle,
                    Some(Arg::Float(0.)),
                    [y.clone(), b],
                    [y.clone()],
                    stream,
                );
                (w, 1.)
            } else {
                (w, 0.)
            }
        };

        assert!(inputs.next().is_none());
        assert_eq!(w.dt(), dt);
        let layout = GemmLayout::new(
            &(dt, y.layout().clone()),
            &(dt, x.layout().clone()),
            &(dt, w.layout().transpose(&[1, 0]).clone()),
        );

        handle.cublas.set_stream(stream);
        let scalar = match dt {
            types::F16 => GemmScheme::<f16, f32>::new(1., beta).to_value(),
            types::F32 => GemmScheme::<f32, f32>::new(1., beta).to_value(),
            types::F64 => GemmScheme::<f64, f64>::new(1., beta).to_value(),
            _ => todo!(),
        };
        let (a, b) = if layout.ab_swap { (w, x) } else { (x, w) };
        match layout.batch {
            0 => unreachable!(),
            1 => unsafe {
                handle.cublas.gemm(
                    layout.m,
                    layout.n,
                    layout.k,
                    scalar,
                    offset_ptr(&a).cast(),
                    layout.a_trans,
                    layout.a_ld,
                    offset_ptr(&b).cast(),
                    layout.b_trans,
                    layout.b_ld,
                    offset_ptr(&y).cast_mut().cast(),
                    layout.c_ld,
                )
            },
            _n => todo!(),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct GemmLayout {
    pub ab_swap: bool,
    pub a_trans: bool,
    pub b_trans: bool,

    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,

    pub c_stride: isize,
    pub c_ld: isize,

    pub a_stride: isize,
    pub a_ld: isize,

    pub b_stride: isize,
    pub b_ld: isize,
}

impl GemmLayout {
    pub fn new<const N: usize>(
        c: &(DigitLayout, ArrayLayout<N>),
        a: &(DigitLayout, ArrayLayout<N>),
        b: &(DigitLayout, ArrayLayout<N>),
    ) -> Self {
        // 确认矩阵结构匹配
        let mut c = Matrix::from(c);
        let mut a = Matrix::from(a);
        let mut b = Matrix::from(b);
        if c.r != a.r || c.c != b.c || a.c != b.r {
            panic!()
        }
        // 确认批处理结构匹配
        let batch = c.batch;
        if !a.match_batch(batch) || !b.match_batch(batch) {
            panic!()
        }
        // 确认 c 列优先
        let ab_swap = if c.rs == 1 && c.cs != 1 {
            // Nothing to do
            false
        } else if c.cs == 1 {
            // cT = bT.aT
            c.transpose();
            a.transpose();
            b.transpose();
            swap(&mut a, &mut b);
            true
        } else {
            panic!()
        };

        let (a_ld, a_trans) = a.ld_trans();
        let (b_ld, b_trans) = b.ld_trans();
        Self {
            ab_swap,
            a_trans,
            b_trans,

            batch,
            m: c.r,
            n: c.c,
            k: a.c,

            c_stride: c.stride,
            c_ld: c.cs,

            a_stride: a.stride,
            a_ld,

            b_stride: b.stride,
            b_ld,
        }
    }
}

struct Matrix {
    batch: usize,
    stride: isize,
    r: usize,
    c: usize,
    rs: isize,
    cs: isize,
}

impl<const N: usize> From<&(DigitLayout, ArrayLayout<N>)> for Matrix {
    fn from((dt, layout): &(DigitLayout, ArrayLayout<N>)) -> Self {
        let [batch @ .., r, c] = layout.shape() else {
            unreachable!()
        };
        let [stride @ .., rs, cs] = layout.strides() else {
            unreachable!();
        };
        let unit = dt.nbytes() as isize;
        let (batch, stride) = match batch {
            [] | [1] => {
                assert!(matches!(stride, [] | [_]));
                (1, 0)
            }
            &[batch] => {
                let &[stride] = stride else { unreachable!() };
                (batch, stride / unit)
            }
            _ => panic!(),
        };
        Self {
            batch,
            stride,
            r: *r,
            c: *c,
            rs: rs / unit,
            cs: cs / unit,
        }
    }
}

impl Matrix {
    #[inline(always)]
    fn match_batch(&self, batch: usize) -> bool {
        self.batch == 1 || self.batch == batch
    }
    #[inline(always)]
    fn ld_trans(&mut self) -> (isize, bool) {
        match (self.rs, self.cs) {
            (1, cs) => (cs, false),
            (rs, 1) => (rs, true),
            (_, _) => panic!(),
        }
    }
    #[inline(always)]
    fn transpose(&mut self) {
        swap(&mut self.r, &mut self.c);
        swap(&mut self.rs, &mut self.cs);
    }
}
