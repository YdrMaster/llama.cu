use crate::{
    handle::Handle,
    load::WeightLoader,
    op::{
        self, Operator as _,
        random_sample::{KV_PAIR, KVPair, RandomSample, SampleArgs},
    },
    utils::dims,
};
use nn::{Arg, Linear, NormType, Normalization, Tensor, digit_layout::types};
use operators::cuda::{CurrentCtx, DevMem, HostMem, Stream, VirByte};
use tokeneer::utok;

pub(super) struct OutputHead<'ctx> {
    norm: Tensor<DevMem<'ctx>, 2>,
    linear: Tensor<DevMem<'ctx>, 2>,
    epsilon: Option<Arg>,
    sample: RandomSample<'ctx>,
}

impl<'ctx> OutputHead<'ctx> {
    pub fn new(nn: nn::OutputHead<Tensor<&[u8], 2>>, ctx: &'ctx CurrentCtx) -> Self {
        let nn::OutputHead {
            out_norm: Normalization { items, epsilon, .. },
            lm_head: Linear { weight, .. },
        } = nn;
        let norm = match items {
            NormType::RmsNorm { scale, .. } => scale,
            NormType::LayerNorm { .. } => todo!(),
        };

        dims!([_, n] = weight);

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
            linear: load(weight),
            epsilon: Some(epsilon.into()),
            sample: RandomSample::new(n, ctx),
        }
    }
}

impl OutputHead<'_> {
    pub fn launch<'ctx>(
        &mut self,
        x: Tensor<*const VirByte, 2>,
        out_idx: HostMem,
        config: impl IntoIterator<Item = SampleArgs>,
        handle: &mut Handle,
        stream: &Stream<'ctx>,
    ) -> DevMem<'ctx> {
        let Self {
            norm,
            linear,
            epsilon,
            sample,
        } = self;
        dims!([_, d] = x);
        let out_idx = stream.from_host::<u8>(&out_idx);
        let out_len = out_idx.len() / size_of::<utok>();
        let out_idx =
            Tensor::from_dim_slice(types::U32, [out_len]).map(|_| out_idx.as_ptr().cast());
        // gather
        let mut out = Tensor::new(x.dt(), [out_len, d]).map(|len| stream.malloc::<u8>(len));
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
        let mut logits = Tensor::new(out.dt(), [out_len, nvoc]).map(|len| stream.malloc::<u8>(len));
        let logits = logits.as_mut().map(|mem| mem.as_ptr().cast());
        let lm_head = linear.as_ref().map(|mem| mem.as_ptr().cast());
        op::Linear::launch(
            handle,
            Some(false.into()),
            [out, lm_head],
            [logits.clone()],
            stream,
        );
        let kv_pair = stream.malloc::<KVPair>(out_len);
        for (i, config) in config.into_iter().enumerate() {
            let logits = logits.clone().transform(|layout| layout.index(0, i));
            let kv_pair = Tensor::from_dim_slice(KV_PAIR, &[])
                .map(|_| kv_pair[i * size_of::<KVPair>()..].as_ptr().cast());
            if config.is_argmax() {
                sample.argmax(kv_pair, logits, stream)
            } else {
                sample.sample(kv_pair, logits, config, rand::random(), stream)
            }
        }
        kv_pair
    }
}
