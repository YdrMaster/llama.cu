use super::app_session::AppSession;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use llama_cu::{Received, Service, SessionId};
use ratatui::{
    DefaultTerminal, Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{
        Block, List, ListState, Padding, Paragraph, Scrollbar, ScrollbarOrientation::VerticalRight,
        ScrollbarState,
    },
};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

pub(super) struct App {
    service: Service,
    stop: bool,
    focus: Focus,
    current: usize,
    sessions: HashMap<SessionId, AppSession>,
    sess_list: Vec<SessionId>,
    pos: usize,
    hc: usize,
    ha: usize,
}

enum Focus {
    List(usize),
    Main(Instant),
}

impl Focus {
    fn select(&self) -> bool {
        matches!(self, Self::List(_))
    }

    fn chat(&self) -> bool {
        matches!(self, Self::Main(_))
    }
}

enum State {
    User,
    Assistant,
}

impl App {
    pub fn new(service: Service) -> Self {
        let session = AppSession::new("default", service.terminal().new_cache());
        let default_id = session.id();
        Self {
            service,
            stop: false,
            focus: Focus::Main(Instant::now()),
            current: 0,
            sessions: [(default_id, session)].into(),
            sess_list: vec![default_id],
            pos: 0,
            hc: 1,
            ha: 1,
        }
    }

    pub fn run(mut self, mut terminal: DefaultTerminal) -> std::io::Result<()> {
        while !self.stop {
            terminal.draw(|frame| self.render(frame))?;
            self.handle_crossterm_events()?;
            self.handle_service()
        }
        Ok(())
    }

    fn current_id(&self) -> SessionId {
        self.sess_list[self.current]
    }

    fn state(&self) -> State {
        if self.sessions[&self.current_id()].msgs().len() % 2 == 0 {
            State::Assistant
        } else {
            State::User
        }
    }

    fn render(&mut self, frame: &mut Frame) {
        let outer = Layout::vertical([Constraint::Fill(1), Constraint::Length(3)]);
        let [main, bottom] = outer.areas(frame.area());
        let inner = Layout::horizontal([Constraint::Length(30), Constraint::Fill(1)]).spacing(1);
        let [list, main] = inner.areas(main);
        frame.render_stateful_widget(self.sess_list(), list, &mut self.list_state());
        self.render_main_dialog(main, frame);
        frame.render_widget(self.state_bar(), bottom);
    }

    fn sess_list(&self) -> List {
        let mut title = Line::from("sessions").bold();
        if self.focus.select() {
            title = title.fg(Color::Black).bg(Color::LightBlue);
        }

        let items = self.sessions.iter().map(|(_, s)| {
            if s.is_busy() {
                Span::raw(format!("[{}]", s.name()))
            } else {
                Span::raw(s.name())
            }
        });
        let mut style = Style::default().add_modifier(Modifier::BOLD);
        if self.focus.select() {
            style = style.fg(Color::Black).bg(Color::White)
        }
        List::new(items)
            .block(Block::bordered().title(title))
            .highlight_style(style)
            .highlight_symbol("> ")
    }

    fn list_state(&self) -> ListState {
        let mut state = ListState::default();
        state.select(Some(match self.focus {
            Focus::List(i) => i,
            Focus::Main(_) => self.current,
        }));
        state
    }

    fn render_main_dialog(&mut self, area: Rect, frame: &mut Frame) {
        let mut title = Line::from("llama.cu advanced chat").bold();
        if let Focus::Main(_) = &self.focus {
            title = title.fg(Color::Black).bg(Color::LightBlue);
        }

        let mut text = String::new();
        let s = &self.sessions[&self.current_id()];
        for (i, msg) in s.msgs().iter().enumerate() {
            text.push_str(if i % 2 == 0 { "user> " } else { "assistant> " });
            text.push_str(&msg);
            text.push('\n')
        }
        if let State::User = self.state() {
            if let Focus::Main(cursor) = &mut self.focus {
                let time = Instant::now();
                let duration = time.duration_since(*cursor);
                if duration < Duration::from_millis(500) {
                } else if duration < Duration::from_secs(1) {
                    text.pop();
                    text.push('_')
                } else {
                    *cursor = time
                }
            }
        }

        self.hc = text
            .lines()
            .map(|line| (line.len() + 1).div_ceil(area.width as _))
            .sum();
        self.ha = area.height as _;
        self.pos = self.pos.min(self.hc.saturating_sub(self.ha));

        let para = Paragraph::new(text)
            .block(
                Block::bordered()
                    .title(title)
                    .padding(Padding::horizontal(1)),
            )
            .scroll((self.pos as _, 0));

        frame.render_widget(para, area);

        let scrollbar = Scrollbar::new(VerticalRight);
        let mut scrollbar_state = ScrollbarState::default()
            .content_length(self.hc.saturating_sub(self.ha))
            .position(self.pos); // 滚动到末尾
        frame.render_stateful_widget(scrollbar, area, &mut scrollbar_state);
    }

    fn state_bar(&self) -> Paragraph {
        let title = Line::from("state").bold().blue();
        let text = format!(
            "msgs: {}, pos: {}/{}/{}",
            self.sessions.get(&self.current_id()).unwrap().msgs().len(),
            self.pos,
            self.ha,
            self.hc,
        );
        Paragraph::new(text).block(Block::bordered().title(title))
    }

    fn handle_crossterm_events(&mut self) -> std::io::Result<()> {
        let interval = match self.state() {
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

    fn on_key_event(&mut self, key: KeyEvent) {
        const CTRL: KeyModifiers = KeyModifiers::CONTROL;
        const SHIFT: KeyModifiers = KeyModifiers::SHIFT;
        let message = self
            .sessions
            .get_mut(&self.current_id())
            .unwrap()
            .last_sentence_mut();
        match (key.modifiers, key.code) {
            (CTRL, KeyCode::Char('c') | KeyCode::Char('C')) => self.stop = true,
            (_, KeyCode::Esc) => self.cancel(),

            (_, KeyCode::Char(ch)) if self.focus.chat() => message.push(ch),
            (SHIFT, KeyCode::Enter) if self.focus.chat() => message.push('\n'),
            (_, KeyCode::Backspace) if self.focus.chat() => {
                message.pop();
            }

            (_, KeyCode::Left) if self.focus.chat() => self.focus = Focus::List(self.current),
            (_, KeyCode::Right) if self.focus.select() => self.focus = Focus::Main(Instant::now()),
            (_, KeyCode::Up) => match &mut self.focus {
                Focus::List(i) => *i = i.saturating_sub(1),
                Focus::Main(_) => self.pos = self.pos.saturating_sub(1),
            },
            (_, KeyCode::Down) => match &mut self.focus {
                Focus::List(i) => *i = i.saturating_add(1).min(self.sess_list.len()),
                Focus::Main(_) => self.pos = self.pos.saturating_add(1),
            },
            (_, KeyCode::Enter) => match &self.focus {
                Focus::List(i) => self.current = *i,
                Focus::Main(_) => self.send(),
            },

            (_, KeyCode::Char('+') | KeyCode::Char('=')) if self.focus.select() => {
                let session = AppSession::new("new session", self.service.terminal().new_cache());
                self.sess_list.push(session.id());
                self.sessions.insert(session.id(), session);
            }
            _ => {}
        }
    }

    fn send(&mut self) {
        if let Some((session, prompt)) = self.sessions.get_mut(&self.current_id()).unwrap().start()
        {
            self.service.terminal().start(session, prompt, true);
        }
    }

    fn handle_service(&mut self) {
        let Received { sessions, outputs } = self.service.try_recv();
        for (id, (_, piece)) in outputs {
            let str = unsafe { std::str::from_utf8_unchecked(&piece) };
            self.sessions
                .get_mut(&id)
                .unwrap()
                .last_sentence_mut()
                .push_str(str)
        }
        if let Some((session, _)) = sessions.into_iter().next() {
            self.sessions.get_mut(&session.id).unwrap().idle(session)
        }
    }

    fn cancel(&mut self) {
        if let State::Assistant = self.state() {
            self.service.terminal().stop(SessionId(0));
        }
    }
}
