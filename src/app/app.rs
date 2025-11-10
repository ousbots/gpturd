use crate::{
    app::{
        message::{self, AppMessage, EventMessage, LossType, ModelCommandMessage, ModelResultMessage},
        options::{self, Options},
    },
    error::VibeError,
    model,
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
use std::sync::mpsc::{Receiver, Sender};
use std::thread::{self, JoinHandle};

pub struct App {
    pub terminal: DefaultTerminal,
    pub state: State,
    pub show_generated: bool,
    pub options: Options,
    pub loss_data: Vec<(f64, f64)>,
    pub validation_loss_data: Vec<(f64, f64)>,
    pub generated_data: Vec<String>,
    pub model_commands: Sender<ModelCommandMessage>,
    pub messages: Receiver<AppMessage>,
    pub model_thread: JoinHandle<Result<(), VibeError>>,
}

#[derive(PartialEq)]
pub enum State {
    Main,
    Training,
    Generate,
    Exit,
}

impl App {
    // Initialize the terminal, parse options, spawn event and model threads.
    pub fn new() -> Result<Self, VibeError> {
        let backend = CrosstermBackend::new(io::stdout());
        let terminal = Terminal::new(backend).unwrap_or_else(|err| {
            panic!("unable to open terminal: {}", err);
        });

        let mut options = Options::new();
        options::parse_args(&mut options)?;
        let model_options = options.clone();

        let (commands_tx, commands_rx) = message::create_command_channel();
        let (data_tx, data_rx) = message::create_data_channel();

        let data_tx_model = data_tx.clone();
        let model_thread = thread::spawn(move || model::run_model(commands_rx, data_tx_model, &model_options));

        thread::spawn(move || {
            loop {
                if let Ok(event) = event::read() {
                    if let Some(key) = event.as_key_press_event() {
                        if let Err(_) = data_tx.send(AppMessage::Event(EventMessage::Key { event: key })) {
                            break;
                        }
                    }
                }
            }
        });

        Ok(Self {
            terminal: terminal,
            state: State::Main,
            show_generated: false,
            loss_data: Vec::new(),
            validation_loss_data: Vec::new(),
            generated_data: Vec::new(),
            model_commands: commands_tx,
            messages: data_rx,
            options: options,
            model_thread: model_thread,
        })
    }

    // Draw the main interface screen.
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

    fn handle_messages(&mut self) -> Result<(), VibeError> {
        match self.messages.recv()? {
            AppMessage::Model(message) => {
                self.process_model_message(message)?;
            }

            AppMessage::Event(message) => {
                self.process_event_message(message)?;
            }
        }

        Ok(())
    }

    // Process user input.
    fn process_event_message(&mut self, event: EventMessage) -> Result<(), VibeError> {
        match event {
            EventMessage::Key { event } => match event.code {
                KeyCode::Char('q') | KeyCode::Esc => {
                    self.model_commands.send(ModelCommandMessage::Shutdown)?;
                    self.state = State::Exit;
                }

                KeyCode::Char('t') | KeyCode::Enter => {
                    if self.state == State::Main {
                        self.model_commands.send(ModelCommandMessage::Train {
                            iterations: self.options.iterations,
                            data_path: self.options.data.clone(),
                            start: self.loss_data.last().unwrap_or(&(0., 0.)).0 as usize,
                        })?;
                        self.state = State::Training;
                    }
                }

                KeyCode::Char('g') => {
                    if self.state == State::Main {
                        self.model_commands.send(ModelCommandMessage::Generate {
                            count: self.options.generate,
                        })?;
                        self.state = State::Generate;
                    }
                }

                KeyCode::Char('p') => {
                    self.show_generated = !self.show_generated;
                }

                _ => {}
            },
        }

        Ok(())
    }

    // Process all training messages.
    fn process_model_message(&mut self, message: ModelResultMessage) -> Result<(), VibeError> {
        match message {
            ModelResultMessage::Progress {
                loss_type,
                iteration,
                loss,
            } => match loss_type {
                LossType::Training => {
                    self.loss_data.push((iteration as f64, loss as f64));
                }
                LossType::Validation => {
                    self.validation_loss_data.push((iteration as f64, loss as f64));
                }
            },

            ModelResultMessage::Generated { text } => {
                self.generated_data.push(text);
            }

            // TODO: errors should be displayed separately from generated text.
            ModelResultMessage::Error { err } => {
                self.generated_data.push(err.to_string());
                self.state = State::Main;
            }

            ModelResultMessage::Finished => {
                self.state = State::Main;
            }
        }

        Ok(())
    }

    // App state machine.
    pub fn run(&mut self) -> Result<(), VibeError> {
        enable_raw_mode()?;
        execute!(self.terminal.backend_mut(), EnterAlternateScreen)?;

        loop {
            if self.state == State::Exit {
                break;
            }

            self.draw_main()?;
            self.handle_messages()?;
        }

        disable_raw_mode()?;
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen)?;
        self.terminal.show_cursor()?;

        Ok(())
    }
}
