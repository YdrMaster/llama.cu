use std::{
    borrow::Borrow,
    collections::HashMap,
    hash::Hash,
    ops::{Index, Range},
};

pub(super) struct RangeCollector<K> {
    calculator: RangeCalculator,
    ranges: HashMap<K, Range<usize>>,
    sizes: NumCollector<usize>,
}

impl<K> RangeCollector<K> {
    pub fn new(alignment: usize) -> Self {
        Self {
            calculator: RangeCalculator {
                align: alignment,
                size: 0,
            },
            ranges: Default::default(),
            sizes: Default::default(),
        }
    }

    #[inline]
    pub const fn size(&self) -> usize {
        self.calculator.size
    }

    #[inline]
    pub fn sizes(&self) -> impl Iterator<Item = (usize, usize)> {
        self.sizes.0.iter().map(|(&a, &b)| (a, b))
    }
}

impl<K: Eq + Hash> RangeCollector<K> {
    pub fn insert(&mut self, key: K, len: usize) {
        use std::collections::hash_map::Entry::{Occupied, Vacant};
        match self.ranges.entry(key) {
            Occupied(entry) => {
                assert_eq!(entry.get().len(), len)
            }
            Vacant(entry) => {
                entry.insert(self.calculator.push(len));
                self.sizes.insert(len)
            }
        }
    }
}

impl<K, Q> Index<&Q> for RangeCollector<K>
where
    K: Borrow<Q> + Eq + Hash,
    Q: Hash + Eq,
{
    type Output = Range<usize>;

    fn index(&self, index: &Q) -> &Self::Output {
        &self.ranges[index]
    }
}

struct RangeCalculator {
    align: usize,
    size: usize,
}

impl RangeCalculator {
    #[inline]
    pub fn push(&mut self, size: usize) -> Range<usize> {
        let start = self.size.div_ceil(self.align) * self.align;
        self.size = start + size;
        start..self.size
    }
}

/// 统计指定类型的参数出现的次数。
#[derive(Debug)]
#[repr(transparent)]
struct NumCollector<T>(HashMap<T, usize>);

impl<T> Default for NumCollector<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T: Eq + Hash> NumCollector<T> {
    pub fn insert(&mut self, t: T) {
        *self.0.entry(t).or_insert(0) += 1
    }
}
