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
    /// 已占用的容量
    pos: usize,
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
            pos: 0,
        }
    }

    pub fn update(&mut self, len: usize, pages: &mut MemPages) -> bool {
        if len > self.buf_len() {
            return false;
        }

        let &mut Self {
            ref mut vir,
            pos,
            size_per_token,
            page_size,
            ..
        } = self;
        // 计算页数
        let mapped = (pos * size_per_token).div_ceil(page_size);
        let target = (len * size_per_token).div_ceil(page_size);
        // 映射物理页，多退少补
        use std::cmp::Ordering::{Equal, Greater, Less};
        match mapped.cmp(&target) {
            Less => pages.map(vir, mapped..target),
            Greater => pages.unmap(vir, target..mapped),
            Equal => {}
        }
        // 更新位置
        self.pos = len;
        true
    }

    pub const fn as_tensor(&self) -> &Tensor<*const VirByte, 2> {
        &self.tensor
    }

    pub const fn pos(&self) -> usize {
        self.pos
    }

    /// kv cache 虚地址空间容量
    pub fn buf_len(&self) -> usize {
        self.tensor.shape()[3]
    }
}
