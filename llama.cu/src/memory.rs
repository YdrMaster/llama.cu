use nn::{Distribution, Tensor};
use operators::cuda::{Device, MappedMem, MemProp, PhyMem, VirByte, VirMem};
use std::{
    mem::replace,
    ops::{Deref, RangeBounds},
    ptr::dangling,
    sync::{Arc, LazyLock},
};

pub(crate) enum Page {
    Vir(VirMem),
    Mapped(MappedMem),
}

pub(crate) struct AddrRegion {
    len: usize,
    pages: Box<[Page]>,
}

impl Deref for AddrRegion {
    type Target = [VirByte];

    fn deref(&self) -> &Self::Target {
        let ptr = self.pages.first().map_or(dangling(), |page| match page {
            Page::Vir(mem) => mem.as_ptr(),
            Page::Mapped(mem) => mem.as_ptr().cast(),
        });

        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }
}

impl AddrRegion {
    pub fn pages(&mut self, range: impl RangeBounds<usize>) -> &mut [Page] {
        &mut self.pages[(range.start_bound().cloned(), range.end_bound().cloned())]
    }
}

pub(crate) struct KVCache {
    /// 基于虚地址的 cache 张量
    tensor_vir: Tensor<*const VirByte, 2>,
    /// cache 占用的地址区域
    mem_region: AddrRegion,
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
        let mem_region = pages.reserve_vir(*template.get()); // 为 kv cache 分配虚页
        let tensor_vir = template
            .map(|_| mem_region.as_ptr()) // 存入张量
            .transform(|layout| layout.transpose(&[3, 1, 2, 0])); // 转置 [nkvh, nblk, 2, nctx, dh]

        Self {
            tensor_vir,
            mem_region,
            size_per_token,
            page_size: pages.page_size(),
            mapped: 0,
        }
    }

    pub fn prepare(&mut self, ntok: usize, pages: &mut MemPages) {
        let n_page = (ntok * self.size_per_token).div_ceil(self.page_size);
        if self.mapped < n_page {
            pages.map(self.mem_region.pages(self.mapped..n_page));
            self.mapped = n_page
        }
    }

    pub const fn as_tensor(&self) -> &Tensor<*const VirByte, 2> {
        &self.tensor_vir
    }
}

pub(crate) struct MemPages {
    base: usize,
    prop: MemProp,
    size: usize,
    pool: Vec<Arc<PhyMem>>,
}

impl MemPages {
    pub fn new(dev: &Device) -> Self {
        static BASE: LazyLock<usize> = LazyLock::new(|| {
            let base = VirMem::new(1 << 30, 0).as_ptr() as usize;
            let step = 1 << 44;
            (base + step - 1) & !(step - 1)
        });
        let base = *BASE + dev.index() as usize * (1 << 40);
        let prop = dev.mem_prop();
        let size = prop.granularity_minimum();
        let pool = Vec::new();
        Self {
            base,
            prop,
            size,
            pool,
        }
    }

    pub const fn page_size(&self) -> usize {
        self.size
    }

    pub fn reserve_vir(&self, size: usize) -> AddrRegion {
        let n_pages = size.div_ceil(self.size);
        if n_pages == 0 {
            return AddrRegion {
                len: 0,
                pages: Box::new([]),
            };
        }

        let mut ans = Vec::with_capacity(n_pages);
        let first = VirMem::new(self.size, self.base);
        let mut end = first.as_ptr_range().end;
        ans.push(first);
        while ans.len() < n_pages {
            let next = VirMem::new(self.size, end as _);
            let ptr = next.as_ptr();
            if ptr != end {
                ans.clear()
            }
            end = next.as_ptr_range().end;
            ans.push(next)
        }
        AddrRegion {
            len: n_pages * self.size,
            pages: ans.into_iter().map(Page::Vir).collect(),
        }
    }

    pub fn put(&mut self, page: Arc<PhyMem>) {
        self.pool.push(page)
    }

    pub fn take(&mut self) -> Arc<PhyMem> {
        self.pool
            .pop()
            .unwrap_or_else(|| self.prop.create(self.size))
    }

    pub fn map(&mut self, pages: &mut [Page]) {
        // 创建一个占位用的虚页
        let mut placeholder = None;
        // 遍历页面
        for page in pages {
            match page {
                Page::Mapped(_) => {} // 已映射，什么都不做
                Page::Vir(vir) => {
                    // 取出或创建占位虚页
                    let exchange = placeholder
                        .take()
                        .unwrap_or_else(|| VirMem::new(self.size, self.base));
                    // 换出虚页并映射
                    let mapped = replace(vir, exchange).map(self.take());
                    // 换回占位的页
                    let Page::Vir(exchange) = replace(page, Page::Mapped(mapped)) else {
                        unreachable!()
                    };
                    placeholder = Some(exchange)
                }
            }
        }
    }

    pub fn unmap(&mut self, pages: &mut [Page]) {
        // 创建一个占位用的虚页
        let mut placeholder = None;
        // 遍历页面
        for page in pages {
            match page {
                Page::Vir(_) => {} // 已解除，什么都不做
                Page::Mapped(_) => {
                    // 取出或创建占位虚页
                    let exchange = placeholder
                        .take()
                        .unwrap_or_else(|| Page::Vir(VirMem::new(self.size, self.base)));
                    // 换出已映射的页
                    let Page::Mapped(mapped) = replace(page, exchange) else {
                        unreachable!()
                    };
                    // 解除映射
                    let (vir, phy) = mapped.unmap();
                    self.put(phy);
                    // 换回占位的页
                    placeholder = Some(replace(page, Page::Vir(vir)))
                }
            }
        }
    }
}
