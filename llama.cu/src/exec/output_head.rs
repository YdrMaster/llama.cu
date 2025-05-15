use crate::{
    handle::Handle,
    load::WeightLoader,
    op::{self, Operator as _},
    utils::{dims, layout, offset_ptr},
};
use nn::{
    Arg, Linear, NormType, Normalization, Tensor, digit_layout::types, ndarray_layout::ArrayLayout,
};
use operators::{
    Operator as _, TensorLayout,
    cuda::{CurrentCtx, DevMem, HostMem, Stream, VirByte},
    random_sample::{
        Args as SampleArgs, Indices, KVPair, RandomSample, SampleArgs as Config,
        cuda::Operator as Sample,
    },
};

pub(super) struct OutputHead<'ctx> {
    norm: Tensor<DevMem<'ctx>, 2>,
    linear: Tensor<DevMem<'ctx>, 2>,
    epsilon: Option<Arg>,

    sample: Sample,
    indices: Tensor<DevMem<'ctx>, 2>,
}

impl<'ctx> OutputHead<'ctx> {
    pub fn new(
        nn: nn::OutputHead<Tensor<&[u8], 2>>,
        sample: Sample,
        ctx: &'ctx CurrentCtx,
    ) -> Self {
        let nn::OutputHead {
            out_norm: Normalization { items, epsilon, .. },
            lm_head: Linear { weight, .. },
        } = nn;
        let norm = match items {
            NormType::RmsNorm { scale, .. } => scale,
            NormType::LayerNorm { .. } => todo!(),
        };
        let linear = weight;
        dims!([nvoc, _] = linear);

        let stream = ctx.stream();
        let mut loader = WeightLoader::new([]);
        let mut load = |t: Tensor<&[u8], 2>| {
            let dst = stream.malloc::<u8>(t.get().len());
            let (host, mut ans) = t.replace(dst);
            loader.load(ans.get_mut(), &stream, |inter| {
                inter.copy_from_slice(host);
            });
            ans
        };

        Self {
            norm: load(norm),
            linear: load(linear),
            epsilon: Some(epsilon.into()),
            sample,
            indices: {
                let Indices { n, mem } = Sample::build_indices(nvoc, &stream);
                Tensor::from_dim_slice(types::U32, [n]).map(|_| mem)
            },
        }
    }
}

impl OutputHead<'_> {
    pub fn launch<'ctx>(
        &self,
        x: Tensor<*const VirByte, 2>,
        out_idx: HostMem,
        n_out: usize,
        config: impl IntoIterator<Item = Config>,
        handle: &mut Handle,
        stream: &Stream<'ctx>,
    ) -> DevMem<'ctx> {
        let Self {
            norm,
            linear,
            epsilon,
            sample,
            indices,
        } = self;
        dims!([_, d] = x);
        let out_idx = stream.from_host::<u8>(&out_idx);
        let out_idx = Tensor::from_dim_slice(types::U32, [n_out]).map(|_| out_idx.as_ptr().cast());
        // gather
        let mut out = Tensor::new(x.dt(), [n_out, d]).map(|len| stream.malloc::<u8>(len));
        let out = out.as_mut().map(|mem| mem.as_ptr().cast());
        op::Embedding::launch(handle, None, [x, out_idx], [out.clone()], stream);
        // norm
        let scale = norm.as_ref().map(|mem| mem.as_ptr().cast());
        op::RmsNorm::launch(
            handle,
            epsilon.clone(),
            [out.clone(), scale],
            [out.clone()],
            stream,
        );
        // linear
        dims!([nvoc, _] = linear);
        let mut logits = Tensor::new(out.dt(), [n_out, nvoc]).map(|len| stream.malloc::<u8>(len));
        let logits = logits.as_mut().map(|mem| mem.as_ptr().cast());
        let lm_head = linear.as_ref().map(|mem| mem.as_ptr().cast());
        op::Linear::launch(
            handle,
            Some(false.into()),
            [out, lm_head],
            [logits.clone()],
            stream,
        );
        let mut kv_pair = stream.malloc::<KVPair<()>>(n_out);
        for (i, config) in config.into_iter().enumerate() {
            let logit = logits.clone().transform(|layout| layout.index(0, i));
            let kv_pair = &mut kv_pair[i * size_of::<KVPair<()>>()..];
            sample
                .launch(
                    &SampleArgs {
                        kv_pair: TensorLayout {
                            dt: KVPair::<()>::LAYOUT,
                            layout: ArrayLayout::new(&[], &[], 0),
                        },
                        kv_pair_base: kv_pair.as_mut_ptr(),
                        logits: layout(&logit),
                        logits_base: offset_ptr(&logits).cast(),
                        indices: layout(indices),
                        indices_base: indices.get().as_ptr(),
                        seed: if config.is_argmax() {
                            1.
                        } else {
                            rand::random()
                        },
                        config,
                    },
                    &mut [],
                    stream,
                )
                .unwrap()
        }
        kv_pair
    }
}
