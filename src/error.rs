use crate::app::message::{AppMessage, ModelCommandMessage};

use std::fmt;

/// Custom error type for anything and everything, the vibe error.
#[derive(Debug, Clone)]
pub struct VibeError {
    message: String,
}

impl VibeError {
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl fmt::Display for VibeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vibe Error: {}", self.message)
    }
}

impl From<candle_core::Error> for VibeError {
    fn from(err: candle_core::Error) -> Self {
        VibeError::new(format!("Candle computation error: {}", err))
    }
}

impl From<std::io::Error> for VibeError {
    fn from(err: std::io::Error) -> Self {
        VibeError::new(format!("IO operation failed: {}", err))
    }
}

impl From<std::num::ParseIntError> for VibeError {
    fn from(err: std::num::ParseIntError) -> Self {
        VibeError::new(format!("Failed to parse integer: {}", err))
    }
}

impl From<std::num::ParseFloatError> for VibeError {
    fn from(err: std::num::ParseFloatError) -> Self {
        VibeError::new(format!("Failed to parse float: {}", err))
    }
}

impl From<std::sync::mpsc::SendError<AppMessage>> for VibeError {
    fn from(err: std::sync::mpsc::SendError<AppMessage>) -> Self {
        VibeError::new(format!("Failed to send model message: {}", err))
    }
}

impl From<std::sync::mpsc::SendError<ModelCommandMessage>> for VibeError {
    fn from(err: std::sync::mpsc::SendError<ModelCommandMessage>) -> Self {
        VibeError::new(format!("Failed to send model message: {}", err))
    }
}
