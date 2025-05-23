use super::{
    super::{Command, Output, Session, SessionId, group::Req},
    SessionStub,
};
use crate::exec::KVCache;
use log::warn;
use operators::random_sample::SampleArgs;
use std::{
    collections::BTreeMap,
    iter::repeat_n,
    mem::take,
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, Sender, TryRecvError},
    },
};
use tokeneer::utok;

#[derive(Default)]
pub(super) struct EngineManager {
    sess: BTreeMap<SessionId, SessionStub>,
    pre_output: BTreeMap<SessionId, usize>,
}

#[derive(Default)]
pub struct Round {
    pub overflow: Vec<Session>,
    pub tokens: Vec<utok>,
    pub reqs: Vec<Req<Arc<[Mutex<KVCache>]>>>,
    pub sample: Vec<SampleArgs>,
    pub output: Vec<(SessionId, usize)>,
    pub fast_map: Vec<(usize, usize)>,
    pub no_decode: Vec<Session>,
}

#[derive(Clone, Copy, Debug)]
pub enum CommandReceiveError {
    SendError,
    ReceiveError,
}

type E = CommandReceiveError;

impl EngineManager {
    /// 接收命令
    pub fn receive(
        &mut self,
        commands: &Receiver<Command>,
        outputs: &Sender<Output>,
    ) -> Result<(), E> {
        macro_rules! apply {
            ($cmd:expr) => {
                if let Some(session) = self.apply($cmd) {
                    if outputs.send(Output::Removed(session)).is_err() {
                        return Err(E::SendError);
                    }
                }
            };
        }

        loop {
            // 总是尝试进行非阻塞接收
            loop {
                match commands.try_recv() {
                    Ok(cmd) => apply!(cmd),
                    Err(TryRecvError::Disconnected) => return Err(E::ReceiveError),
                    Err(TryRecvError::Empty) => break,
                }
            }
            // 没有待处理的命令
            if self.sess.is_empty() {
                // 也没有待处理的任务，阻塞等待
                match commands.recv() {
                    Ok(cmd) => apply!(cmd),
                    Err(_) => break Err(E::ReceiveError),
                }
            } else {
                // 有待处理的任务，退出循环
                break Ok(());
            }
        }
    }

    /// 准备推理
    pub fn prepare(&mut self) -> Round {
        let mut ans = Round::default();
        let mut out_idx = 0;

        let pre_output = take(&mut self.pre_output);
        for (id, mut stub) in take(&mut self.sess) {
            let max = stub.session.cache.len;
            let pos = stub.session.cache.pos;
            let seq = stub.state.seq;
            let out = stub.state.out;
            let end = pos + seq;
            assert_eq!(out, 1, "TODO: chunked prefill");
            // 尝试填充缓存
            if end > max {
                warn!("cache overflow {end} > {max}");
                // 缓存溢出，不再推理
                ans.overflow.push(stub.session);
                continue;
            }
            stub.session.cache.pos = end;
            // 填充推理信息
            ans.sample.extend(repeat_n(stub.session.sample_args, out));
            ans.output.push((id, out));
            ans.reqs.push(Req {
                kv_cache: stub.session.cache.parts.clone(),
                pos,
                seq,
            });
            if let Some(prompt) = stub.prompt.take() {
                // prefill
                debug_assert_eq!(seq, prompt.len());
                ans.tokens.extend(prompt);
                // if out == 0 {
                //     // todo!("chunked prefill")
                //     // ans.no_decode.push(stub.session);
                //     // continue;
                // }
                stub.state.seq = 1
            } else {
                // decode
                assert_eq!(seq, 1);
                // fast embd
                ans.fast_map.push((pre_output[&id], ans.tokens.len()));
                ans.tokens.push(0)
            }
            // 回填
            assert!(self.sess.insert(id, stub).is_none());
            assert!(self.pre_output.insert(id, out_idx).is_none());
            out_idx += out
        }
        ans
    }

    pub fn into_stubs(self) -> impl IntoIterator<Item = SessionStub> {
        self.sess.into_values()
    }

    fn apply(&mut self, cmd: Command) -> Option<Session> {
        match cmd {
            Command::Insert(req) => {
                self.sess.insert(req.session.id, req.into_stub());
                None
            }
            Command::Remove(id) => {
                // fmt
                self.sess.remove(&id).map(|stub| stub.session)
            }
        }
    }
}
