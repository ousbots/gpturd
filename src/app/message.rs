use crate::error::VibeError;
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
    Train { iterations: usize, data_path: String },
    Generate { count: usize },
    Shutdown,
}

// Create a new channel pair for training communication.
pub fn create_results_channel() -> (Sender<ModelResultMessage>, Receiver<ModelResultMessage>) {
    mpsc::channel()
}

// Create a new channel pair for model commands.
pub fn create_command_channel() -> (Sender<ModelCommandMessage>, Receiver<ModelCommandMessage>) {
    mpsc::channel()
}
