use crate::{data::tokenize, error::VibeError};

use candle_core::{Device, Tensor};
use rand::seq::SliceRandom;
use std::fs;

#[derive(Clone)]
pub struct Data {
    pub input: Tensor,
    pub target: Tensor,
    pub validation_input: Tensor,
    pub validation_target: Tensor,
}

pub const DEFAULT_DATA_PATH: &str = "data/names_short.txt";

// Read the data into a list of strings using newlines as a separator.
fn parse_data(path: &String) -> Result<Vec<String>, VibeError> {
    let content = fs::read_to_string(path).map_err(|e| VibeError::new(format!("unable to open {}: {}", path, e)))?;

    let items: Vec<String> = content.lines().map(|elem| String::from(elem).trim().to_lowercase()).collect();

    Ok(items)
}

// Randomize the input data, break it into different data sets, then tokenize and convert to
// tensors for training..
//
// The two different data sets will be the training set and the validation set. The training set
// is used for model training, the validation set is a set of valid words the model hasn't been
// trained on that we can validate against.
pub fn training_data(path: &String, block_size: usize, device: &Device) -> Result<Data, VibeError> {
    let mut data = parse_data(path)?;
    data.shuffle(&mut rand::rng());

    let training_end = (data.len() as f64 * 0.9).round() as usize;

    let (input, target) = tokenize::tokenize(&data[..training_end].to_vec(), block_size, device)?;
    let (validation_input, validation_target) = tokenize::tokenize(&data[training_end..].to_vec(), block_size, device)?;

    Ok(Data {
        input: input,
        target: target,
        validation_input: validation_input,
        validation_target: validation_target,
    })
}
