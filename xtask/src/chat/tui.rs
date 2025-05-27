use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use llama_cu::{Received, Service, Session, SessionId};
use ratatui::{
    DefaultTerminal, Frame,
    layout::{Constraint, Layout},
    style::Stylize,
    text::Line,
    widgets::{Block, Paragraph},
};

pub(super) struct App {
    service: Service,
    session: Option<Session>,
    state: State,
    stop: bool,
    cursor: Instant,
    messages: Vec<String>,
}

enum State {
    User,
    Assistant,
}

impl App {
    pub fn new(service: Service) -> Self {
        let session = Session {
            id: SessionId(0),
            sample_args: Default::default(),
            cache: service.terminal().new_cache(),
        };
        Self {
            service,
            session: Some(session),
            state: State::User,
            stop: false,
            cursor: Instant::now(),
            messages: vec![String::new()],
        }
    }

    /// Run the application's main loop.
    pub fn run(mut self, mut terminal: DefaultTerminal) -> std::io::Result<()> {
        while !self.stop {
            terminal.draw(|frame| self.render(frame))?;
            self.handle_crossterm_events()?;
            self.handle_service()
        }
        Ok(())
    }

    fn render(&mut self, frame: &mut Frame) {
        let vertical = Layout::vertical([Constraint::Fill(1), Constraint::Length(3)]).spacing(1);
        let [main, bottom] = vertical.areas(frame.area());
        frame.render_widget(self.main_dialog(), main);
        frame.render_widget(self.state_bar(), bottom);
    }

    fn main_dialog(&mut self) -> Paragraph {
        let title = Line::from("llama.cu advanced chat").bold().blue();
        let mut text = String::new();
        for (i, msg) in self.messages.iter().enumerate() {
            text.push_str(if i % 2 == 0 { "user> " } else { "assistant> " });
            text.push_str(&msg);
            text.push('\n')
        }
        if let State::User = self.state {
            let time = Instant::now();
            let duration = time.duration_since(self.cursor);
            if duration < Duration::from_millis(500) {
            } else if duration < Duration::from_secs(1) {
                text.pop();
                text.push('_')
            } else {
                self.cursor = time
            }
        }
        Paragraph::new(text).block(Block::bordered().title(title))
    }

    fn state_bar(&self) -> Paragraph {
        let title = Line::from("state").bold().blue();
        let text = format!("msgs: {}", self.messages.len());
        Paragraph::new(text).block(Block::bordered().title(title))
    }

    /// Reads the crossterm events and updates the state of [`App`].
    ///
    /// If your application needs to perform work in between handling events, you can use the
    /// [`event::poll`] function to check if there are any events available with a timeout.
    fn handle_crossterm_events(&mut self) -> std::io::Result<()> {
        let interval = match self.state {
            State::User => Duration::from_millis(250),
            State::Assistant => Duration::from_millis(5),
        };
        if event::poll(interval)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    self.on_key_event(key)
                }
            }
        }
        Ok(())
    }

    /// Handles the key events and updates the state of [`App`].
    fn on_key_event(&mut self, key: KeyEvent) {
        const CTRL: KeyModifiers = KeyModifiers::CONTROL;
        const SHIFT: KeyModifiers = KeyModifiers::SHIFT;
        match (key.modifiers, key.code) {
            (CTRL, KeyCode::Char('c') | KeyCode::Char('C')) => self.stop = true,
            (_, KeyCode::Esc) => self.cancel(),
            (_, KeyCode::Char(ch)) => self.messages.last_mut().unwrap().push(ch),
            (SHIFT, KeyCode::Enter) => self.messages.last_mut().unwrap().push('\n'),
            (_, KeyCode::Backspace) => {
                self.messages.last_mut().unwrap().pop();
            }
            (_, KeyCode::Enter) => self.send(),
            _ => {}
        }
    }

    fn send(&mut self) {
        self.service.terminal().start(
            self.session.take().unwrap(),
            self.messages.last().unwrap().clone(),
            true,
        );
        self.messages.push(String::new());
        self.state = State::Assistant
    }

    fn handle_service(&mut self) {
        let Received { sessions, outputs } = self.service.try_recv();

        for (_, (_, piece)) in outputs {
            let str = unsafe { std::str::from_utf8_unchecked(&piece) };
            self.messages.last_mut().unwrap().push_str(str)
        }
        if let Some((s, _)) = sessions.into_iter().next() {
            self.session = Some(s);
            self.messages.push(String::new());
            self.state = State::User
        }
    }

    fn cancel(&mut self) {
        if let State::Assistant = self.state {
            self.service.terminal().stop(SessionId(0));
        }
    }
}
