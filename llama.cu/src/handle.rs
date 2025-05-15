use crate::op::ModuleKey;
use operators::{
    cublas::Cublas,
    cuda::{CurrentCtx, Module, Ptx},
};
use std::collections::HashMap;

#[cfg(nccl)]
use operators::nccl::Communicator;

pub(crate) struct Handle<'ctx> {
    pub ctx: &'ctx CurrentCtx,
    pub cublas: Cublas<'ctx>,
    pub modules: HashMap<Box<[ModuleKey]>, Module<'ctx>>,
    #[cfg(nccl)]
    pub comm: Option<Communicator>,
}

impl<'ctx> Handle<'ctx> {
    pub fn new(ctx: &'ctx CurrentCtx) -> Self {
        Self {
            ctx,
            cublas: Cublas::new(ctx),
            modules: HashMap::new(),
            #[cfg(nccl)]
            comm: None,
        }
    }

    #[cfg(nccl)]
    pub fn with_comm(ctx: &'ctx CurrentCtx, comm: Communicator) -> Self {
        Self {
            ctx,
            cublas: Cublas::new(ctx),
            modules: HashMap::new(),
            comm: Some(comm),
        }
    }

    pub fn compile(&mut self, key: Box<[ModuleKey]>, code: impl FnOnce() -> String) -> &Module {
        self.modules.entry(key).or_insert_with(|| {
            let (ptx, log) = Ptx::compile(code(), self.ctx.dev().compute_capability());
            let Ok(ptx) = ptx else { panic!("{log}") };
            self.ctx.load(&ptx)
        })
    }
}
