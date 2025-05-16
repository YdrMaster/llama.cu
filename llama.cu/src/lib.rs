mod exec;
mod handle;
mod infer;
mod load;
mod memory;
mod model;
mod op;
mod utils;

use log::info;
use std::{
    ffi::c_int,
    marker::PhantomData,
    path::PathBuf,
    sync::mpsc::{Receiver, Sender, channel},
    time::Duration,
};

#[allow(non_camel_case_types)]
type upos = u32;

pub struct Session {
    user: Sender<Task>,
    assistant: Receiver<Receiver<String>>,
}

struct Task {
    prompt: String,
    use_template: bool,
}

#[repr(transparent)]
pub struct BusySession<'s>(Receiver<String>, PhantomData<&'s mut Session>);

#[repr(transparent)]
pub struct Handle(Option<std::thread::JoinHandle<(Duration, usize)>>);

impl Session {
    pub fn new(model: PathBuf, gpus: Box<[c_int]>, max_steps: usize) -> (Self, Handle) {
        let (user, request) = channel();
        let (response, assistant) = channel();
        let thread: std::thread::JoinHandle<(Duration, usize)> =
            std::thread::spawn(move || infer::infer(model, &gpus, max_steps, request, response));
        (Self { user, assistant }, Handle(Some(thread)))
    }

    pub fn send(&mut self, prompt: String, use_template: bool) -> BusySession {
        self.user
            .send(Task {
                prompt,
                use_template,
            })
            .unwrap();
        BusySession(self.assistant.recv().unwrap(), PhantomData)
    }
}

impl BusySession<'_> {
    pub fn receive(&self) -> Option<String> {
        self.0.recv().ok()
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        let (duration, steps) = self.0.take().unwrap().join().unwrap();
        let time = duration / steps as _;
        info!(
            "steps = {steps}, perf: {time:?}/tok, {}tok/s",
            Duration::from_secs(1).div_duration_f32(time),
        )
    }
}

// pub struct Service {
//     handle: Handle,
//     commands: Sender<Command>,
//     responses: Receiver<Response>,
// }

// pub enum Command {}
// pub enum Response {}

// struct Cache {
//     history: History,
//     kv_cache: KVCache,
// }

// struct History(Vec<Piece>);

// struct Piece {
//     text: String,
//     tokens: SmallVec<[u32; 1]>,
// }
