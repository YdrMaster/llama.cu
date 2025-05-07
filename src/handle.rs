use crate::{
    macros::destruct,
    op::{self, ModuleKey, Operator},
};
use nn::{Node, Tensor};
use operators::{
    cublas::Cublas,
    cuda::{CurrentCtx, GraphExec, Module, Ptx, VirByte},
};
use regex::Regex;
use std::{collections::HashMap, sync::LazyLock};

pub(crate) struct Handle<'ctx> {
    pub ctx: &'ctx CurrentCtx,
    pub cublas: Cublas<'ctx>,
    pub modules: HashMap<Box<[ModuleKey]>, Module<'ctx>>,
}

pub(crate) enum Exec<'ctx> {
    Graph(GraphExec<'ctx>),
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
            let Node { name, op, arg } = node;
            macro_rules! add_to_graph {
                ($op:ident) => {
                    op::$op::launch(
                        self,
                        arg,
                        inputs,
                        outputs,
                        stream.get_or_insert_with(|| self.ctx.stream().capture()),
                    )
                };
            }
            match &*op {
                "embedding" => add_to_graph!(Embedding),
                "rms-norm" => add_to_graph!(RmsNorm),
                "linear" => add_to_graph!(Linear),
                "rope" => add_to_graph!(Rope),
                "swiglu" => add_to_graph!(Swiglu),
                "empty" => {}
                "attention" => {
                    static REGEX: LazyLock<Regex> =
                        LazyLock::new(|| Regex::new(r"^Ω\.blk(\d+)\.attn:attention$").unwrap());

                    if let Some(stream) = stream.take() {
                        exec_.push(Exec::Graph(self.ctx.instantiate(&stream.end())))
                    }

                    destruct!([q, k, v] = inputs);
                    destruct!([o] = outputs);
                    let Some(nn::Arg::Int(dh)) = arg else {
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
                    print!("todo! {op} ({arg:?})");
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
            exec_.push(Exec::Graph(self.ctx.instantiate(&stream.end())))
        }
        exec_.into()
    }
}
