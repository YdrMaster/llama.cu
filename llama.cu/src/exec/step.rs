use crate::{
    handle::Handle,
    op::{self, Operator},
    utils::destruct,
};
use log::error;
use nn::{Arg, Tensor};
use operators::cuda::{GraphExec, VirByte};
use regex::Regex;
use std::{fmt, sync::LazyLock};

pub(super) enum Step<'ctx> {
    Graph(GraphExec<'ctx>, Box<[Tensor<*const VirByte, 2>]>),
    Attention(Box<Attention>),
}

pub(super) struct Attention {
    pub iblk: usize,
    pub q: Tensor<*const VirByte, 2>,
    pub k: Tensor<*const VirByte, 2>,
    pub v: Tensor<*const VirByte, 2>,
    pub o: Tensor<*const VirByte, 2>,
}

impl<'ctx> Handle<'ctx> {
    pub(super) fn merge_cuda_graph(
        &mut self,
        exec: impl IntoIterator<Item = nn::Exec<*const VirByte>>,
    ) -> Box<[Step<'ctx>]> {
        let mut stream = None;
        let mut exec_ = Vec::new();
        for nn::Exec {
            node,
            inputs,
            outputs,
        } in exec
        {
            let op = node.value;
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
                        exec_.push(Step::Graph(
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
                    // [n, nh * dh] -> [n, nh, dh] -> [nh, n, dh]
                    let transform = |t: Tensor<*const VirByte, 2>| {
                        t.transform(|layout| {
                            layout
                                .tile_be(1, &[layout.shape()[1] / dh, dh])
                                .transpose(&[1, 0])
                        })
                    };
                    let q = transform(q);
                    let k = transform(k);
                    let v = transform(v);
                    let o = transform(o);

                    let iblk = {
                        let (_, [iblk]) = REGEX.captures(&node.name).unwrap().extract();
                        iblk.parse().unwrap()
                    };
                    exec_.push(Step::Attention(Box::new(Attention { iblk, q, k, v, o })))
                }
                _ => {
                    error!(
                        "{}",
                        ErrorFmt {
                            name: &node.name,
                            ty: &op.name,
                            arg: &op.arg,
                            inputs: &inputs,
                            outputs: &outputs,
                        }
                    );
                    break;
                }
            }
        }
        if let Some(stream) = stream.take() {
            exec_.push(Step::Graph(
                self.ctx.instantiate(&stream.end()),
                Default::default(),
            ))
        }
        exec_.into()
    }
}

struct ErrorFmt<'a> {
    name: &'a str,
    ty: &'a str,
    arg: &'a Option<Arg>,
    inputs: &'a [Tensor<*const VirByte, 2>],
    outputs: &'a [Tensor<*const VirByte, 2>],
}

impl fmt::Display for ErrorFmt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &Self {
            name,
            ty,
            arg,
            inputs,
            outputs,
        } = self;
        write!(f, "todo! [{ty}] {name} ({arg:?})")?;
        for t in inputs {
            write!(f, " {}{:?}", t.dt(), t.shape())?
        }
        write!(f, " ->")?;
        for t in outputs {
            write!(f, " {}{:?}", t.dt(), t.shape())?
        }
        writeln!(f)
    }
}
