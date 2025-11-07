use crate::{
    app::{
        message::{self, LossType, ModelMessage},
        options::{self, Options},
    },
    data::parse,
    error::VibeError,
    model::Model,
    ui::main_screen,
};

use crossterm::event::{self, KeyCode};
use ratatui::{
    DefaultTerminal, Terminal,
    backend::CrosstermBackend,
    crossterm::execute,
    crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use std::io;
use std::sync::mpsc::Receiver;
use std::thread::{self, JoinHandle};

pub struct App {
    pub terminal: DefaultTerminal,
    pub state: State,
    pub show_generated: bool,
    pub options: Options,
    pub loss_data: Vec<(f64, f64)>,
    pub validation_loss_data: Vec<(f64, f64)>,
    pub generated_data: Vec<String>,
    pub model: Model,
    pub model_thread: Option<JoinHandle<Result<(), VibeError>>>,
}

#[derive(PartialEq)]
pub enum State {
    Main,
    Training,
    Generate,
    Exit,
}

impl App {
    pub fn new() -> Result<Self, VibeError> {
        let backend = CrosstermBackend::new(io::stdout());
        let terminal = Terminal::new(backend).unwrap_or_else(|err| {
            panic!("unable to open terminal: {}", err);
        });

        let mut options = Options::new();
        options::parse_args(&mut options)?;

        Ok(Self {
            terminal: terminal,
            state: State::Main,
            show_generated: false,
            loss_data: Vec::new(),
            validation_loss_data: Vec::new(),
            generated_data: Vec::new(),
            model: Model::init(&options)?,
            options: options,
            model_thread: None,
        })
    }

    pub fn draw_main(&mut self) -> Result<(), VibeError> {
        self.terminal.draw(|frame| {
            main_screen::draw(
                frame,
                &self.options,
                &self.loss_data,
                &self.validation_loss_data,
                &self.generated_data,
                self.show_generated,
            )
        })?;
        Ok(())
    }

    fn handle_input(&mut self) -> Result<(), VibeError> {
        let event = event::read()?;

        if event
            .as_key_press_event()
            .is_some_and(|key| key.code == KeyCode::Char('q') || key.code == KeyCode::Esc)
        {
            self.state = State::Exit;
        }

        if event
            .as_key_press_event()
            .is_some_and(|key| key.code == KeyCode::Char('t') || key.code == KeyCode::Enter)
        {
            if self.state != State::Training {
                self.state = State::Training;
            }
        }

        if event.as_key_press_event().is_some_and(|key| key.code == KeyCode::Char('g')) {
            if self.state != State::Generate {
                self.state = State::Generate;
            }
        }

        if event.as_key_press_event().is_some_and(|key| key.code == KeyCode::Char('p')) {
            self.show_generated = !self.show_generated;
        }

        Ok(())
    }

    // Run the model text generation in a new thread.
    fn start_generation(&mut self) -> Result<(), VibeError> {
        if self.model_thread.is_some() {
            return Ok(());
        }

        let (sender, receiver) = message::create_channel();
        let options = self.options.clone();
        let mut model = self.model.clone();

        self.model_thread = Some(thread::spawn(move || model.generate(options.generate, sender)));

        self.process_messages(receiver)?;

        // Wait on the generation thread then restore the model.
        if let Some(thread) = self.model_thread.take() {
            let result = thread.join().map_err(|err| VibeError::new(format!("{:?}", err)))?;
            result?;
        }

        Ok(())
    }

    // Run the model training in a new thread.
    fn start_training(&mut self) -> Result<(), VibeError> {
        if self.model_thread.is_some() {
            return Ok(());
        }

        let (sender, receiver) = message::create_channel();
        let options = self.options.clone();
        let mut model = self.model.clone();

        let data = parse::training_data(&options.data, options.block_size, &model.device)?;

        self.model_thread = Some(thread::spawn(move || model.train(options.iterations, data, sender)));

        self.process_messages(receiver)?;

        // Wait on the training thread then restore the model.
        if let Some(thread) = self.model_thread.take() {
            let result = thread.join().map_err(|err| VibeError::new(format!("{:?}", err)))?;
            result?;
        }

        Ok(())
    }

    // Process all training messages, re-drawing as needed.
    fn process_messages(&mut self, receiver: Receiver<ModelMessage>) -> Result<(), VibeError> {
        while let Ok(message) = receiver.recv() {
            match message {
                ModelMessage::Progress {
                    loss_type,
                    iteration,
                    loss,
                } => match loss_type {
                    LossType::Training => {
                        self.loss_data.push((iteration as f64, loss as f64));
                        self.draw_main()?;
                    }
                    LossType::Validation => {
                        self.validation_loss_data.push((iteration as f64, loss as f64));
                        self.draw_main()?;
                    }
                },
                ModelMessage::Generated { text } => {
                    self.generated_data.push(text);
                    if self.show_generated {
                        self.draw_main()?;
                    }
                }
                ModelMessage::Finished => {
                    break;
                }
            }
        }

        Ok(())
    }

    pub fn run(&mut self) -> Result<(), VibeError> {
        enable_raw_mode()?;
        execute!(self.terminal.backend_mut(), EnterAlternateScreen)?;

        loop {
            match self.state {
                State::Main => {
                    self.draw_main()?;
                }
                State::Training => {
                    self.start_training()?;
                    self.state = State::Main;
                }
                State::Generate => {
                    self.start_generation()?;
                    self.state = State::Main;
                }
                State::Exit => break,
            }

            self.handle_input()?;
        }

        disable_raw_mode()?;
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen)?;
        self.terminal.show_cursor()?;

        Ok(())
    }
}
