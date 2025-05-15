mod loader;
mod range_collector;

use crate::memory::MemPages;
use loader::WeightLoader;
use nn::{Edge, TPAction, TPTensor, Tensor};
use operators::cuda::{VirByte, VirMem};
use range_collector::RangeCollector;
use std::collections::HashSet;

type HostTPTensor<'a> = TPTensor<Tensor<&'a [u8], 2>>;
type VirTensor = Tensor<*const VirByte, 2>;

impl MemPages {
    pub fn load_weight(
        &mut self,
        edges: Box<[Edge<HostTPTensor>]>,
    ) -> (VirMem, Box<[Edge<VirTensor>]>) {
        // 排布权重存储
        let align = Some(self.dev().alignment())
            .filter(|&n| n > 0)
            .unwrap_or(512);
        let mut ranges = RangeCollector::new(align);
        for nn::Edge { external, .. } in &edges {
            if let Some(nn::External { item, .. }) = external {
                let TPTensor { act, val } = item;
                let len = match act {
                    Some(act) => val.get().len() / act.dist.total * act.dist.len,
                    None => val.get().len(),
                };
                ranges.insert((act.clone(), val.get().as_ptr()), len)
            }
        }
        // 权重加载
        let mut weight = self.reserve_vir(ranges.size());
        let mapped = weight.map(0, self.prop().create(weight.len()));
        let edges = self.dev().context().apply(|ctx| {
            let mut loader = WeightLoader::new(
                ranges
                    .sizes()
                    .filter(|&(_, times)| times < 4)
                    .map(|(size, _)| size),
            );

            let stream = ctx.stream();
            let mut copied = HashSet::new();
            edges
                .into_iter()
                .map(|nn::Edge { meta, external }| nn::Edge {
                    meta,
                    external: external.map(|nn::External { name, item }| {
                        let TPTensor { act, val } = item;
                        let range = &ranges[&(act.clone(), val.get().as_ptr())];
                        let dev = &mut mapped[range.clone()];
                        let ptr = dev.as_ptr().cast();
                        nn::External {
                            name,
                            item: match act.clone() {
                                Some(TPAction { wt, dist }) => {
                                    if copied.insert(range.clone()) {
                                        loader
                                            .load(dev, &stream, |dst| wt.move_data(dist, dst, &val))
                                    }
                                    let shape = wt.split_shape(dist, val.shape());
                                    Tensor::from_dim_slice(val.dt(), &shape).map(|_| ptr)
                                }
                                None => {
                                    if copied.insert(range.clone()) {
                                        loader.load(dev, &stream, |dst| {
                                            dst.copy_from_slice(val.get())
                                        })
                                    }
                                    val.map(|_| ptr)
                                }
                            },
                        }
                    }),
                })
                .collect::<Box<_>>()
        });
        (weight, edges)
    }
}
