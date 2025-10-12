use crate::{data::convert, error::VibeError};

use candle_core::{Device, Tensor};

// Tokenize a list of strings for neural network training.
//
// Strings are tokenized characterwise in blocks specified by options.block_size.
pub fn tokenize(
    words: &Vec<String>,
    block_size: usize,
    device: &Device,
) -> Result<(Tensor, Tensor), VibeError> {
    let delimiter: char = convert::LETTERS[0];
    let mut input: Vec<Vec<u8>> = vec![];
    let mut target: Vec<u8> = vec![];

    for word in words {
        let mut context: Vec<u8> = vec![0; block_size];

        let mut chars: Vec<char> = word.chars().collect();
        chars.push(delimiter);

        for letter in chars {
            let letter_value = convert::ltoi(letter);
            input.push(context.clone());
            target.push(letter_value);

            context.remove(0);
            context.push(letter_value);
        }
    }

    let input_tensor = Tensor::from_vec(
        input.iter().flatten().copied().collect(),
        (input.len(), input[0].len()),
        device,
    )?;

    let target_len = target.len();
    let target_tensor = Tensor::from_vec(target, target_len, device)?;

    Ok((input_tensor, target_tensor))
}
