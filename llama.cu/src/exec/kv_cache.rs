use crate::memory::MemPages;
use nn::{Distribution, Tensor};
use operators::cuda::{VirByte, VirMem};

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
        // 转置 [nctx, nblk, 2, nkvh, dh] -> [nkvh, nblk, 2, nctx, dh]
        let tensor = template
            .map(|_| vir.as_ptr())
            .transform(|layout| layout.transpose(&[3, 0]));

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
