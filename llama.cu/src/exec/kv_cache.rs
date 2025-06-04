use crate::memory::MemPages;
use nn::Tensor;
use operators::cuda::{VirByte, VirMem};

pub(crate) struct KVCache {
    /// 基于虚地址的 cache 张量
    tensor: Tensor<VirMem, 2>,
    /// cache 中每个 token 的尺寸
    size_per_token: usize,
    /// cache 当前映射的页数
    mapped: usize,
}

impl KVCache {
    pub fn new(template: &Tensor<usize, 2>, len: usize, total: usize, pages: &MemPages) -> Self {
        let mut shape = template.shape().to_vec();
        shape[3] = shape[3] / total * len;
        let template = Tensor::from_dim_slice(template.dt(), &shape);

        let size_per_token = template.get() / template.shape()[0];
        // 转置 [nctx, nblk, 2, nkvh, dh] -> [nkvh, nblk, 2, nctx, dh]
        let tensor = template
            .map(|len| pages.reserve_vir(len))
            .transform(|layout| layout.transpose(&[3, 0]));

        Self {
            tensor,
            size_per_token,
            mapped: 0,
        }
    }

    /// 更新 kv cache，使之可容纳 len 个 token
    pub fn update(&mut self, len: usize, pages: &mut MemPages) {
        assert!(len <= self.buf_len());

        let size_per_token = self.size_per_token;
        let page_size = pages.page_size();
        // 计算页数
        let target = (len * size_per_token).div_ceil(page_size);
        // 映射物理页，多退少补
        use std::cmp::Ordering::{Equal, Greater, Less};
        let mem = self.tensor.get_mut();
        match self.mapped.cmp(&target) {
            Less => pages.map(mem, self.mapped..target),
            Greater => pages.unmap(mem, target..self.mapped),
            Equal => {}
        }
        self.mapped = target
    }

    pub fn as_tensor(&self) -> Tensor<*const VirByte, 2> {
        self.tensor.as_ref().map(|vir| vir.as_ptr())
    }

    /// kv cache 虚地址空间容量
    pub fn buf_len(&self) -> usize {
        self.tensor.shape()[3]
    }
}
