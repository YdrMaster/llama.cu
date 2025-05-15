use nn::{Distribution, Tensor};
use operators::cuda::{Device, MemProp, PhyMem, VirByte, VirMem};
use std::{
    ops::{Range, RangeBounds},
    sync::Arc,
};

pub(crate) struct KVCache {
    /// 基于虚地址的 cache 张量
    tensor: Tensor<*const VirByte, 2>,
    /// cache 占用的地址区域
    vir: VirMem,
    /// cache 中每个 token 的尺寸
    size_per_token: usize,
    /// 每个页的大小
    page_size: usize,
    /// 已映射的页数
    mapped: usize,
}

impl KVCache {
    pub fn new(template: &Tensor<usize, 2>, dist: Distribution, pages: &mut MemPages) -> Self {
        let mut shape = template.shape().to_vec();
        shape[3] = shape[3] / dist.total * dist.len;
        let template = Tensor::from_dim_slice(template.dt(), &shape);

        let size_per_token = template.get() / template.shape()[0];
        let vir = pages.reserve_vir(*template.get());
        let tensor = template
            .map(|_| vir.as_ptr()) // 存入张量
            .transform(|layout| layout.transpose(&[3, 1, 2, 0])); // 转置 [nkvh, nblk, 2, nctx, dh]

        Self {
            tensor,
            vir,
            size_per_token,
            page_size: pages.page_size(),
            mapped: 0,
        }
    }

    pub fn prepare(&mut self, ntok: usize, pages: &mut MemPages) {
        let n_page = (ntok * self.size_per_token).div_ceil(self.page_size);
        if self.mapped < n_page {
            pages.map(&mut self.vir, self.mapped..n_page);
            self.mapped = n_page
        }
    }

    pub const fn as_tensor(&self) -> &Tensor<*const VirByte, 2> {
        &self.tensor
    }
}

pub(crate) struct MemPages {
    dev: Device,
    prop: MemProp,
    size: usize,
    pool: Vec<Arc<PhyMem>>,
}

impl MemPages {
    pub fn new(dev: Device) -> Self {
        let prop = dev.mem_prop();
        let size = prop.granularity_minimum();
        let pool = Vec::new();
        Self {
            dev,
            prop,
            size,
            pool,
        }
    }

    #[inline(always)]
    pub const fn dev(&self) -> &Device {
        &self.dev
    }

    #[inline(always)]
    pub const fn prop(&self) -> MemProp {
        self.prop
    }

    #[inline(always)]
    pub const fn page_size(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn reserve_vir(&self, len: usize) -> VirMem {
        VirMem::new(len.div_ceil(self.size) * self.size, 0)
    }

    pub fn map(&mut self, mem: &mut VirMem, range: impl RangeBounds<usize>) {
        for i in self.page_range(mem, range) {
            mem.map(i * self.size, self.take());
        }
    }

    pub fn unmap(&mut self, mem: &mut VirMem, range: impl RangeBounds<usize>) {
        for i in self.page_range(mem, range) {
            self.pool.push(mem.unmap(i * self.size))
        }
    }

    #[inline]
    fn take(&mut self) -> Arc<PhyMem> {
        self.pool
            .pop()
            .unwrap_or_else(|| self.prop.create(self.size))
    }

    fn page_range(&self, mem: &VirMem, range: impl RangeBounds<usize>) -> Range<usize> {
        use std::ops::Bound::{Excluded, Included, Unbounded};
        let start = match range.start_bound() {
            Included(i) => *i,
            Excluded(i) => *i + 1,
            Unbounded => 0,
        };
        let end = match range.end_bound() {
            Included(i) => *i + 1,
            Excluded(i) => *i,
            Unbounded => mem.len() / self.size,
        };
        start..end
    }
}
