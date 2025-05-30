use llama_cu::{DistKVCache, Session, SessionId, TextBuf};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

pub(super) struct AppSession {
    name: String,
    msgs: Vec<String>,
    info: Option<Session>,
    pub buf: TextBuf,
}

impl AppSession {
    pub fn new(name: impl ToString, cache: DistKVCache) -> Self {
        static ID: AtomicUsize = AtomicUsize::new(0);
        Self {
            name: name.to_string(),
            msgs: vec![String::new()],
            info: Some(Session {
                id: SessionId(ID.fetch_add(1, SeqCst)),
                sample_args: Default::default(),
                cache,
            }),
            buf: TextBuf::new(),
        }
    }

    pub fn id(&self) -> SessionId {
        self.info.as_ref().unwrap().id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn msgs(&self) -> &[String] {
        &self.msgs
    }

    pub fn last_sentence_mut(&mut self) -> &mut String {
        self.msgs.last_mut().unwrap()
    }

    pub fn start(&mut self) -> Option<(Session, String)> {
        let ans = self
            .info
            .take()
            .map(|s| (s, self.msgs.last().unwrap().clone()));
        self.msgs.push(Default::default());
        ans
    }

    pub fn idle(&mut self, session: Session) {
        self.msgs.push(Default::default());
        assert!(self.info.replace(session).is_none())
    }

    pub fn is_busy(&self) -> bool {
        self.info.is_none()
    }
}
