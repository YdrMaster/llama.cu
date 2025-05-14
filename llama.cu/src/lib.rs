mod exec;
mod gguf;
mod handle;
mod infer;
mod loader;
mod memory;
mod model;
mod op;
mod utils;

use std::{
    ffi::c_int,
    marker::PhantomData,
    path::PathBuf,
    sync::mpsc::{Receiver, Sender, channel},
    time::Duration,
};

pub struct Session {
    user: Sender<String>,
    assistant: Receiver<Receiver<String>>,
}

#[repr(transparent)]
pub struct BusySession<'s>(Receiver<String>, PhantomData<&'s mut Session>);

#[repr(transparent)]
pub struct Handle(std::thread::JoinHandle<(Duration, usize)>);

impl Session {
    pub fn new(model: PathBuf, gpus: Box<[c_int]>, max_steps: usize) -> (Self, Handle) {
        let (user, request) = channel();
        let (response, assistant) = channel();
        let thread: std::thread::JoinHandle<(Duration, usize)> =
            std::thread::spawn(move || infer::infer(model, &gpus, max_steps, request, response));
        (Self { user, assistant }, Handle(thread))
    }

    pub fn send(&mut self, prompt: String) -> BusySession {
        self.user.send(prompt).unwrap();
        BusySession(self.assistant.recv().unwrap(), PhantomData)
    }
}

impl BusySession<'_> {
    pub fn receive(&self) -> Option<String> {
        self.0.recv().ok()
    }
}

impl Handle {
    pub fn join(self) {
        let (duration, steps) = self.0.join().unwrap();
        let time = duration.div_f32(steps as _);
        println!();
        println!();
        println!(
            "steps = {steps}, perf: {time:?}/tok, {}tok/s",
            Duration::from_secs(1).div_duration_f32(time),
        )
    }
}
