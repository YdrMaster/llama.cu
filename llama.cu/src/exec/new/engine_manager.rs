use crate::exec::KVCache;

use super::{
    super::{Command, Output, Session, SessionId, group::Req},
    SessionStub,
};
use log::warn;
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, Sender, TryRecvError},
    },
};
use tokeneer::utok;

#[derive(Default)]
pub(super) struct EngineManager(BTreeMap<SessionId, SessionStub>);

#[derive(Default)]
pub struct Round {
    pub overflow: Vec<Session>,
    pub tokens: Vec<utok>,
    pub reqs: Vec<Req<Arc<[Mutex<KVCache>]>>>,
    pub output: Vec<(SessionId, usize)>,
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

        while self.0.is_empty() {
            match commands.recv() {
                Ok(cmd) => apply!(cmd),
                Err(_) => return Err(E::ReceiveError),
            }
            loop {
                match commands.try_recv() {
                    Ok(cmd) => apply!(cmd),
                    Err(TryRecvError::Disconnected) => return Err(E::ReceiveError),
                    Err(TryRecvError::Empty) => break,
                }
            }
        }
        Ok(())
    }

    /// 准备推理
    pub fn prepare(&mut self) -> Round {
        let mut ans = Round::default();
        let sessions = std::mem::take(&mut self.0);
        for (id, mut stub) in sessions {
            let max = stub.session.cache.len;
            let pos = stub.session.cache.pos;
            let seq = stub.state.seq;
            let out = stub.state.out;
            // 尝试填充缓存
            if pos + seq > max {
                warn!("overflow {}", pos + seq);
                // 缓存溢出，不再推理
                ans.overflow.push(stub.session);
                continue;
            }
            stub.session.cache.pos += seq;
            // 填充推理信息
            ans.output.push((id, out));
            ans.reqs.push(Req {
                kv_cache: stub.session.cache.parts.clone(),
                pos,
                seq,
            });
            if let Some(prompt) = stub.prompt.take() {
                // prefill
                debug_assert_eq!(stub.state.seq, prompt.len());
                ans.tokens.extend(prompt);
                // TODO fast embd
                // if stub.state.out == 0 {
                // chunked prefill
                ans.no_decode.push(stub.session);
                continue;
                // }
            } else {
                // decode
                assert_eq!(stub.state.seq, 1);
                ans.tokens.push(0);
                todo!("fast embd")
            }
            // 回填
            // TODO fast embd
            // debug_assert!(self.0.insert(id, stub).is_none())
        }
        ans
    }

    pub fn into_stubs(self) -> impl IntoIterator<Item = SessionStub> {
        self.0.into_values()
    }

    fn apply(&mut self, cmd: Command) -> Option<Session> {
        match cmd {
            Command::Insert(req) => {
                self.0.insert(req.session.id, req.into_stub());
                None
            }
            Command::Remove(id) => {
                // fmt
                self.0.remove(&id).map(|stub| stub.session)
            }
        }
    }
}
