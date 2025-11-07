use crate::{
    app::{
        device,
        message::{LossType, ModelMessage},
        options::Options,
    },
    data::{convert, parse::Data},
    error::VibeError,
};

use candle_core::{Device, Tensor, Var, backprop::GradStore};
use candle_nn::{loss, ops};
use rand::Rng;
use std::sync::mpsc::Sender;

// The vocabulary is hardcoded to the 26 letters plus the special delimiter character.
const VOCAB_SIZE: usize = 27;

#[derive(Clone)]
pub struct Model {
    pub device: Device,
    c: Var,
    weights_1: Var,
    biases_1: Var,
    weights_2: Var,
    biases_2: Var,
    hyperparameters: Hyperparameters,
}

#[derive(Clone)]
pub struct Hyperparameters {
    batch_size: usize,
    block_size: usize,
    _embedding_size: usize,
    _hidden_size: usize,
    learn_rate: f32,
}

impl Model {
    pub fn init(options: &Options) -> Result<Self, VibeError> {
        let device = device::open_device(&options.device)?;

        Ok(Self {
            c: Var::rand(0f32, 1f32, (VOCAB_SIZE, options.embedding_size), &device)?,
            // The gain (max value) is discussed in the "Delving Deep into Rectifier" paper by Kaiming He.
            // gain: (5/3) * sqrt(embedding_size * block_size).
            weights_1: Var::rand(
                0f32,
                (5.0 / 3.0) / (options.embedding_size as f32 * options.block_size as f32).sqrt(),
                (options.embedding_size * options.block_size, options.hidden_size),
                &device,
            )?,
            biases_1: Var::rand(0f32, 0.01f32, options.hidden_size, &device)?,
            weights_2: Var::rand(0f32, 0.01f32, (options.hidden_size, VOCAB_SIZE), &device)?,
            biases_2: Var::zeros(VOCAB_SIZE, candle_core::DType::F32, &device)?,
            hyperparameters: Hyperparameters {
                batch_size: options.batch_size,
                block_size: options.block_size,
                _embedding_size: options.embedding_size,
                _hidden_size: options.hidden_size,
                learn_rate: options.learn_rate,
            },
            device: device,
        })
    }

    fn backpropagate(&mut self, loss: &Tensor) -> Result<(), VibeError> {
        let loss_grad = loss.backward()?;

        backpropagate_parameter(&mut self.c, &loss_grad, self.hyperparameters.learn_rate, &self.device)?;
        backpropagate_parameter(&mut self.weights_1, &loss_grad, self.hyperparameters.learn_rate, &self.device)?;
        backpropagate_parameter(&mut self.biases_1, &loss_grad, self.hyperparameters.learn_rate, &self.device)?;
        backpropagate_parameter(&mut self.weights_2, &loss_grad, self.hyperparameters.learn_rate, &self.device)?;
        backpropagate_parameter(&mut self.biases_2, &loss_grad, self.hyperparameters.learn_rate, &self.device)?;

        Ok(())
    }

    fn forward_pass(&self, input: &Tensor, target: &Tensor) -> Result<Tensor, VibeError> {
        // Embed the input into vectors.
        let embeddings = self.c.index_select(&input.flatten_all()?, 0)?;

        // Hidden layer pre-activation with weights and biases and activation with tanh.
        let h = embeddings
            .reshape(((), self.weights_1.dims()[0]))?
            .matmul(&self.weights_1)?
            .broadcast_add(&self.biases_1)?
            .tanh()?;

        // Output layer.
        let logits = h.matmul(&self.weights_2)?.broadcast_add(&self.biases_2)?;

        Ok(loss::cross_entropy(&logits, &target.to_dtype(candle_core::DType::U32)?)?)
    }

    pub fn generate(&mut self, iterations: usize, sender: Sender<ModelMessage>) -> Result<(), VibeError> {
        for _ in 0..iterations {
            let mut output: String = "".to_string();
            let mut context: Vec<u8> = vec![0; self.hyperparameters.block_size];

            loop {
                let embeddings = self
                    .c
                    .index_select(&Tensor::new(context.clone(), &self.device)?.flatten_all()?, 0)?;

                let h = embeddings
                    .reshape(((), self.weights_1.dims()[0]))?
                    .matmul(&self.weights_1)?
                    .broadcast_add(&self.biases_1)?
                    .tanh()?;

                let logits = h.matmul(&self.weights_2)?.broadcast_add(&self.biases_2)?;

                let probs = ops::softmax(&logits, 1)?;

                let position = random_sample(&probs)?;
                if position == 0 {
                    break;
                }
                output.push(convert::itol(position as u8));

                context.remove(0);
                context.push(position as u8);
            }

            let _ = sender.send(ModelMessage::Generated {
                text: format!("    {}", output),
            });
        }

        let _ = sender.send(ModelMessage::Finished);

        Ok(())
    }

    // Training rounds.
    //
    // NOTE: the data is randomly batched every training round and all weights adjusted based on
    // the batch loss. This speeds up training by not having to calculate the entire gradient every
    // round. In the tradeoff between calculating the exact gradient every round versus running
    // more rounds, running more rounds shows better results.
    pub fn train(&mut self, iterations: usize, data: Data, sender: Sender<ModelMessage>) -> Result<(), VibeError> {
        for count in 0..iterations {
            let batch_indices = Tensor::rand(0f32, data.input.dims()[0] as f32, (self.hyperparameters.batch_size,), &self.device)?
                .to_dtype(candle_core::DType::U32)?;

            let loss = self.forward_pass(
                &data.input.index_select(&batch_indices.flatten_all()?, 0)?,
                &data.target.index_select(&batch_indices.flatten_all()?, 0)?,
            )?;

            self.backpropagate(&loss)?;

            // Send progress updates.
            let loss_val: f32 = loss.clone().to_device(&Device::Cpu)?.to_scalar()?;
            let _ = sender.send(ModelMessage::Progress {
                loss_type: LossType::Training,
                iteration: count,
                loss: loss_val.clone(),
            });

            // Send validation progress every few iterations.
            if count % 100 == 0 {
                let validation_loss = self.forward_pass(&data.validation_input, &data.validation_target)?;
                sender.send(ModelMessage::Progress {
                    loss_type: LossType::Validation,
                    iteration: count,
                    loss: validation_loss.to_vec0::<f32>()?,
                })?;
            }
        }

        // let _ = sender.send(ModelMessage::Finished);

        Ok(())
    }
}

// Run the gradient descent backpropagation on a single parameter.
fn backpropagate_parameter(param: &mut Var, loss_grad: &GradStore, learn_rate: f32, device: &Device) -> Result<(), VibeError> {
    // Clear the gradient for this parameter.
    param.backward()?.remove(param.as_tensor());

    // Get the gradient from the loss gradient store.
    let gradient = loss_grad
        .get(param.as_tensor())
        .ok_or_else(|| VibeError::new("missing loss gradient"))?;

    // Compute the update: new_param = param - (gradient * learning_rate)
    let updated_param = param.broadcast_sub(&gradient.broadcast_mul(&Tensor::new(&[learn_rate], device)?)?)?;

    // Replace the parameter with the updated value.
    *param = Var::from_tensor(&updated_param)?;

    Ok(())
}

// Take a random sample from the given probability tensor.
//
// In order to take the probability distribution into account, a cumulative sum of the
// probabilities is computed and the first index with a summed probability greater than a randomly
// chosen value is selected.
fn random_sample(probs: &Tensor) -> Result<usize, VibeError> {
    let random_val: f32 = rand::rng().random_range(0.0..1.0);

    let cumulative_sum = probs.cumsum(1)?.squeeze(0)?.to_vec1()?;
    for (index, &sum) in cumulative_sum.iter().enumerate() {
        if random_val <= sum {
            return Ok(index);
        }
    }

    Ok(cumulative_sum.len() - 1)
}
