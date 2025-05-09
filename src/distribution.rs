use crate::{gguf::GGufModel, meta, range_collector::RangeCollector};
use ggus::GGufMetaMapExt;
use nn::{Edge, Tensor};
use std::{any::Any, hash::Hash, rc::Rc};

/// 分布式切分方式
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Distribution {
    start: usize,
    len: usize,
    total: usize,
}

impl Distribution {
    pub const MONO: Self = Self {
        start: 0,
        len: 1,
        total: 1,
    };

    pub fn new(start: usize, len: usize, total: usize) -> Self {
        assert!(0 < len && start + len <= total);
        Self { start, len, total }
    }

    #[inline]
    pub const fn is_mono(&self) -> bool {
        self.len == self.total
    }
}

pub trait WeightType: Any {
    fn move_data(&self, dist: Distribution, dst: &mut [u8], src: &Tensor<&[u8], 2>);
    fn check_eq(&self, other: &dyn Any) -> bool;
}

#[derive(Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct AttnQKV(usize);

#[derive(Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct FfnGateUp;

#[derive(Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct ColumnWeight;

pub fn weight_type(key: &str, gqa: usize) -> Option<Rc<dyn WeightType>> {
    macro_rules! map {
        ($( $tail:expr => $self:expr )+) => {
            $( if key.contains($tail) { return Some(Rc::new($self)); } )+
        };
    }
    map! {
        ".attn_qkv"    => AttnQKV(gqa)
        ".ffn_gate_up" => FfnGateUp
        ".attn_output" => ColumnWeight
        ".ffn_down"    => ColumnWeight
    }
    None
}

macro_rules! impl_wt_eq {
    () => {
        fn check_eq(&self, other: &dyn Any) -> bool {
            match other.downcast_ref::<Self>() {
                Some(other) => self.eq(other),
                _ => false,
            }
        }
    };
}

impl WeightType for AttnQKV {
    impl_wt_eq!();
    fn move_data(&self, dist: Distribution, dst: &mut [u8], src: &Tensor<&[u8], 2>) {
        match src.layout().ndim() {
            1 => todo!(),
            2 => todo!(),
            _ => unreachable!(),
        }
    }
}

impl WeightType for FfnGateUp {
    impl_wt_eq!();
    fn move_data(&self, dist: Distribution, dst: &mut [u8], src: &Tensor<&[u8], 2>) {
        match src.layout().ndim() {
            1 => todo!(),
            2 => todo!(),
            _ => unreachable!(),
        }
    }
}

impl WeightType for ColumnWeight {
    impl_wt_eq!();
    fn move_data(&self, dist: Distribution, dst: &mut [u8], src: &Tensor<&[u8], 2>) {
        match src.layout().ndim() {
            1 => todo!(),
            2 => todo!(),
            _ => unreachable!(),
        }
    }
}

#[derive(Clone)]
pub struct TPAction {
    pub wt: Rc<dyn WeightType>,
    pub dist: Distribution,
}

impl PartialEq for TPAction {
    fn eq(&self, other: &Self) -> bool {
        self.wt.check_eq(other) && self.dist == other.dist
    }
}

impl Eq for TPAction {}

impl Hash for TPAction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.wt.type_id().hash(state);
        self.dist.hash(state);
    }
}

pub struct TPWeight<'a> {
    pub act: Option<TPAction>,
    pub host: Tensor<&'a [u8], 2>,
}

pub fn map_edge<'w>(
    gguf: &'w GGufModel,
    dist: Distribution,
    edges: impl IntoIterator<Item = Edge<String>>,
) -> (
    RangeCollector<(Option<TPAction>, *const u8)>,
    Box<[Edge<TPWeight<'w>>]>,
) {
    let nh = meta![gguf => llm_attention_head_count];
    let nkvh = meta![gguf => llm_attention_head_count_kv; nh];
    let gqa = nh / nkvh;

    let mut ranges = RangeCollector::new(512);
    let edges = edges
        .into_iter()
        .map(|nn::Edge { meta, external }| nn::Edge {
            meta,
            external: external.map(|nn::External { name, item }| {
                let tensor = gguf.tensors[&*item].as_deref();
                let act = if !dist.is_mono() {
                    weight_type(&item, gqa).map(|wt| TPAction { wt, dist })
                } else {
                    None
                };
                ranges.insert((act.clone(), tensor.get().as_ptr()), tensor.get().len());
                nn::External {
                    name,
                    item: TPWeight { act, host: tensor },
                }
            }),
        })
        .collect::<Box<_>>();
    (ranges, edges)
}
