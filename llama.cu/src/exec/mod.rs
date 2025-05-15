mod group;
mod kv_cache;
mod model;
mod output_head;
mod step;

pub(crate) use group::{ModelGroup, Request};
pub(crate) use kv_cache::KVCache;
// use crate::{handle::Handle, memory::MemPages, upos};
// use nn::{
//     Dim, Distribution, Graph, GraphBuilder, LLaMA, NNGraph, Tensor, TensorMeta,
//     digit_layout::types, op,
// };
// use operators::{
//     Operator,
//     attention_kv_cached::cuda::Operator as Attn,
//     cuda::{CurrentCtx, DevMemSpore, Device, EventSpore, Gpu},
//     random_sample::{KVPair, cuda::Operator as Sample},
// };
// use std::{
//     collections::{BTreeMap, BTreeSet},
//     sync::mpsc::{Receiver, Sender, TryRecvError},
// };

// pub(super) enum Command {
//     Insert(usize, Request),
//     Remove(usize),
// }

// pub(crate) struct Output {
//     pub lens: Box<[(usize, usize)]>,
//     pub result: DevMemSpore,
//     pub event: EventSpore,
//     pub removed: BTreeSet<usize>,
// }

// pub(crate) fn loop_(
//     mut llama: LLaMA<Tensor<&[u8], 2>>,
//     dev: Device,
//     dist: Distribution,
//     commands: Receiver<Command>,
//     outputs: Sender<Output>,
//     handle: impl FnOnce(&CurrentCtx) -> Handle,
// ) {
//     let output_head = llama.output_head.take().unwrap();
//     let NNGraph(Graph { topo, nodes, edges }) = builder()
//         .build(
//             llama.tensor_parallel(dist),
//             [
//                 TensorMeta::new(types::U32, [Dim::var("n_tok")]),
//                 TensorMeta::new(types::U32, [Dim::var("n_tok")]),
//             ],
//         )
//         .unwrap();

//     // 权重加载
//     let mut pages = MemPages::new(dev);
//     let (_weight, edges) = pages.load_weight(edges);

//     // 推理
//     let graph = NNGraph(Graph { topo, nodes, edges });
//     let gpu = Gpu::new(pages.dev().retain_primary(), Default::default());
//     let attn = Attn::new(&gpu);
//     let sample = Sample::new(&gpu);
//     gpu.apply(|ctx| {
//         let mut handle = handle(ctx);
//         let mut models = ModelGroup::new(
//             [1, 8, 32, 128, 512],
//             &graph,
//             output_head,
//             attn,
//             sample,
//             &mut handle,
//             &mut pages,
//         );

//         let mut manager = RequestManager(Default::default());
//         let stream = ctx.stream();
//         'outer: loop {
//             // 接收指令
//             while manager.0.is_empty() {
//                 match commands.recv() {
//                     Ok(cmd) => manager.apply(cmd),
//                     Err(_) => break 'outer,
//                 }
//                 loop {
//                     match commands.try_recv() {
//                         Ok(cmd) => manager.apply(cmd),
//                         Err(TryRecvError::Empty) => break,
//                         Err(TryRecvError::Disconnected) => break 'outer,
//                     }
//                 }
//             }
//             // 组织请求
//             let (ids, reqs) = manager.unzip();
//             let kv_pair = models.launch(reqs, &mut handle, &mut pages, &stream);

//             // if let Err(SendError(_)) = output_head.send(todo!()) {
//             //     break;
//             // }
//         }
//     })
// }

// fn builder() -> GraphBuilder {
//     let mut ans = GraphBuilder::default();
//     ans.register_op("embedding", op::embedding::Embedding)
//         .register_op("rms-norm", op::normalization::RmsNorm)
//         .register_op("linear", op::linear::Linear)
//         .register_op("rope", op::rope::Rope)
//         .register_op("attention", op::attention::Attention)
//         .register_op("swiglu", op::activation::SwiGLU)
//         .register_op("concat", op::concat::Concat)
//         .register_op("split", op::split::Split)
//         .register_op("all-reduce", op::all_reduce::AllReduce);
//     ans
// }

// #[derive(Default)]
// struct RequestManager(BTreeMap<usize, Request>);

// impl RequestManager {
//     fn apply(&mut self, cmd: Command) {
//         match cmd {
//             Command::Insert(id, req) => {
//                 self.0.insert(id, req);
//             }
//             Command::Remove(id) => {
//                 self.0.remove(&id);
//             }
//         }
//     }

//     fn unzip(&self) -> (Box<[usize]>, Box<[Request]>) {
//         let mut ids = vec![0; self.0.len()].into_boxed_slice();
//         let reqs = self
//             .0
//             .iter()
//             .enumerate()
//             .map(|(i, (id, req))| {
//                 ids[i] = *id;
//                 req.clone()
//             })
//             .collect();
//         (ids, reqs)
//     }

//     fn update(&mut self) -> BTreeSet<usize> {
//         let mut remove = BTreeSet::new();
//         for (id, req) in &mut self.0 {
//             match req.out {
//                 1 => {
//                     req.pos += req.tokens.len() as upos;
//                     if req.pos>=
//                 }
//                 _ => todo!(),
//             }
//         }
//         remove
//     }
// }
