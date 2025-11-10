use crate::error::VibeError;

use crossterm::event::KeyEvent;
use std::sync::mpsc::{self, Receiver, Sender};

#[derive(Debug, Clone)]
pub enum LossType {
    Training,
    Validation,
}

// Message types for communication between training thread and UI.
#[derive(Debug, Clone)]
pub enum ModelResultMessage {
    Progress { loss_type: LossType, iteration: usize, loss: f32 },
    Generated { text: String },
    Error { err: VibeError },
    Finished,
}

// Message types for sending commands to the model.
#[derive(Debug)]
pub enum ModelCommandMessage {
    Train { iterations: usize, start: usize },
    Generate { count: usize },
    Shutdown,
}

pub enum EventMessage {
    Key { event: KeyEvent },
}

pub enum AppMessage {
    Model(ModelResultMessage),
    Event(EventMessage),
}

// Create a new channel pair for model commands.
pub fn create_command_channel() -> (Sender<ModelCommandMessage>, Receiver<ModelCommandMessage>) {
    mpsc::channel()
}

// Create a new channel pair for app events.
pub fn create_data_channel() -> (Sender<AppMessage>, Receiver<AppMessage>) {
    mpsc::channel()
}
