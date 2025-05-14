use crate::{
    op::{self, ModuleKey, Operator},
    utils::destruct,
};
use graph::Named;
use nn::Tensor;
use operators::{
    cublas::Cublas,
    cuda::{CurrentCtx, GraphExec, Module, Ptx, VirByte},
};
use regex::Regex;
use std::{collections::HashMap, sync::LazyLock};

#[cfg(nccl)]
use operators::nccl::Communicator;

pub(crate) struct Handle<'ctx> {
    pub ctx: &'ctx CurrentCtx,
    pub cublas: Cublas<'ctx>,
    pub modules: HashMap<Box<[ModuleKey]>, Module<'ctx>>,
    #[cfg(nccl)]
    pub comm: Option<Communicator>,
}

pub(crate) enum Exec<'ctx> {
    Graph(GraphExec<'ctx>, Box<[Tensor<*const VirByte, 2>]>),
    Attention(Box<Attention>),
}

pub(crate) struct Attention {
    pub iblk: usize,
    pub q: Tensor<*const VirByte, 2>,
    pub k: Tensor<*const VirByte, 2>,
    pub v: Tensor<*const VirByte, 2>,
    pub o: Tensor<*const VirByte, 2>,
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

    pub fn merge_cuda_graph(
        &mut self,
        exec: impl IntoIterator<Item = nn::Exec<*const VirByte>>,
    ) -> Box<[Exec<'ctx>]> {
        let mut stream = None;
        let mut exec_ = Vec::new();
        for nn::Exec {
            node,
            inputs,
            outputs,
        } in exec
        {
            let Named { name, value: op } = node;
            macro_rules! add_to_graph {
                ($op:ident) => {
                    op::$op::launch(
                        self,
                        op.arg,
                        inputs,
                        outputs,
                        stream.get_or_insert_with(|| self.ctx.stream().capture()),
                    )
                };
            }
            match &*op.name {
                "embedding" => add_to_graph!(Embedding),
                "rms-norm" => add_to_graph!(RmsNorm),
                "linear" => add_to_graph!(Linear),
                "rope" => add_to_graph!(Rope),
                "swiglu" => add_to_graph!(Swiglu),
                #[cfg(nccl)]
                "all-reduce" => add_to_graph!(AllReduce),
                "empty" => {}
                "attention" => {
                    static REGEX: LazyLock<Regex> =
                        LazyLock::new(|| Regex::new(r"^Ω\.blk(\d+)\.attn:attention$").unwrap());

                    if let Some(stream) = stream.take() {
                        exec_.push(Exec::Graph(
                            self.ctx.instantiate(&stream.end()),
                            Default::default(),
                        ))
                    }

                    destruct!([q, k, v] = inputs);
                    destruct!([o] = outputs);
                    let Some(nn::Arg::Int(dh)) = op.arg else {
                        panic!()
                    };
                    let dh = dh as usize;

                    let transform = |t: Tensor<*const VirByte, 2>| {
                        t.transform(|layout| {
                            layout
                                .tile_be(1, &[layout.shape()[1] / dh, dh])
                                .transpose(&[1, 0])
                        })
                    };

                    let iblk = {
                        let (_, [iblk]) = REGEX.captures(&name).unwrap().extract();
                        iblk.parse().unwrap()
                    };
                    let q = transform(q);
                    let k = transform(k);
                    let v = transform(v);
                    let o = transform(o);

                    exec_.push(Exec::Attention(Box::new(Attention { iblk, q, k, v, o })))
                }
                _ => {
                    print!("todo! {} ({:?})", op.name, op.arg);
                    for t in inputs {
                        print!(" {}{:?}", t.dt(), t.shape())
                    }
                    print!(" ->");
                    for t in outputs {
                        print!(" {}{:?}", t.dt(), t.shape())
                    }
                    println!();
                    break;
                }
            }
        }
        if let Some(stream) = stream.take() {
            exec_.push(Exec::Graph(
                self.ctx.instantiate(&stream.end()),
                Default::default(),
            ))
        }
        exec_.into()
    }
}
