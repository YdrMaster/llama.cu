use operators::cuda::{Device, MappedMem, MemProp, PhyMem, VirByte, VirMem};
use std::{
    ops::Deref,
    ptr::dangling,
    sync::{Arc, LazyLock},
};

pub(super) struct MemPages {
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
            return AddrRegion::Vir(Box::new([]));
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
        AddrRegion::Vir(ans.into())
    }

    pub fn put(&mut self, page: Arc<PhyMem>) {
        self.pool.push(page)
    }

    pub fn take(&mut self) -> Arc<PhyMem> {
        self.pool
            .pop()
            .unwrap_or_else(|| self.prop.create(self.size))
    }

    pub fn map(&mut self, region: &mut AddrRegion) {
        match region {
            AddrRegion::Vir(vir_mems) => {
                *region = AddrRegion::Mapped(
                    std::mem::take(vir_mems)
                        .into_iter()
                        .map(|vir| vir.map(self.take()))
                        .collect(),
                )
            }
            AddrRegion::Mapped(_) => {}
        }
    }

    pub fn unmap(&mut self, region: &mut AddrRegion) {
        match region {
            AddrRegion::Vir(_) => {}
            AddrRegion::Mapped(pages) => {
                *region = AddrRegion::Vir(
                    std::mem::take(pages)
                        .into_iter()
                        .map(|mapped| {
                            let (vir, phy) = mapped.unmap();
                            self.put(phy);
                            vir
                        })
                        .collect(),
                )
            }
        }
    }
}

pub enum AddrRegion {
    Vir(Box<[VirMem]>),
    Mapped(Box<[MappedMem]>),
}

impl Deref for AddrRegion {
    type Target = [VirByte];

    fn deref(&self) -> &Self::Target {
        let (ptr, len) = match self {
            Self::Vir(pages) => get_slice(pages),
            Self::Mapped(pages) => get_slice(pages),
        };
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

fn get_slice<T, U: Deref<Target = [T]>>(pages: &[U]) -> (*const VirByte, usize) {
    pages.first().map_or((dangling(), 0), |slice| {
        (slice.as_ptr().cast(), slice.len() * pages.len())
    })
}
