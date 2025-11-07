use crate::{app::device, data::parse, error::VibeError};
use std::env;

const DEFAULT_DATA_PATH: &str = parse::DEFAULT_DATA_PATH;
const DEFAULT_DEVICE: &str = device::DEVICE_NAME_CPU;
const DEFAULT_ITERATIONS: usize = 1000;
const DEFAULT_BATCH_SIZE: usize = 512;
const DEFAULT_BLOCK_SIZE: usize = 3;
const DEFAULT_EMBEDDING_SIZE: usize = 5;
const DEFAULT_HIDDEN_SIZE: usize = 1000;
const DEFAULT_LEARN_RATE: f32 = 0.1;
const DEFAULT_GENERATE: usize = 20;

// User provided runtime arguments.
#[derive(Debug, Clone)]
pub struct Options {
    pub data: String,
    pub device: String,
    pub iterations: usize,
    pub batch_size: usize,
    pub block_size: usize,
    pub embedding_size: usize,
    pub hidden_size: usize,
    pub learn_rate: f32,
    pub generate: usize,
}

impl Options {
    pub fn new() -> Self {
        Self {
            data: DEFAULT_DATA_PATH.to_string(),
            device: DEFAULT_DEVICE.to_string(),
            iterations: DEFAULT_ITERATIONS,
            batch_size: DEFAULT_BATCH_SIZE,
            block_size: DEFAULT_BLOCK_SIZE,
            embedding_size: DEFAULT_EMBEDDING_SIZE,
            hidden_size: DEFAULT_HIDDEN_SIZE,
            learn_rate: DEFAULT_LEARN_RATE,
            generate: DEFAULT_GENERATE,
        }
    }
}

// Parse the command line options.
pub fn parse_args(options: &mut Options) -> Result<(), VibeError> {
    let mut args: Vec<String> = env::args().collect();
    args.reverse();
    args.pop();

    while let Some(arg) = args.pop() {
        match arg.as_str() {
            "--data" => {
                if let Some(path) = args.pop() {
                    options.data = path;
                } else {
                    print_help();
                    return Err(VibeError::new("missing the path portion of the --data flag"));
                }
            }
            "--device" => {
                if let Some(path) = args.pop() {
                    options.device = path;
                } else {
                    print_help();
                    return Err(VibeError::new("missing the device portion of the --device flag"));
                }
            }
            "--iterations" => {
                if let Some(iterations) = args.pop() {
                    options.iterations = str::parse::<usize>(iterations.as_str())?;
                } else {
                    print_help();
                    return Err(VibeError::new("missing the number portion of the --iterations flag"));
                }
            }
            "--batch-size" => {
                if let Some(size) = args.pop() {
                    options.batch_size = str::parse::<usize>(size.as_str())?;
                } else {
                    print_help();
                    return Err(VibeError::new("missing the size portion of the --batch-size flag"));
                }
            }
            "--block-size" => {
                if let Some(size) = args.pop() {
                    options.block_size = str::parse::<usize>(size.as_str())?;
                } else {
                    print_help();
                    return Err(VibeError::new("missing the size portion of the --block-size flag"));
                }
            }
            "--embedding-size" => {
                if let Some(size) = args.pop() {
                    options.embedding_size = str::parse::<usize>(size.as_str())?;
                } else {
                    print_help();
                    return Err(VibeError::new("missing the size portion of the --embedding-size flag"));
                }
            }
            "--hidden-size" => {
                if let Some(size) = args.pop() {
                    options.hidden_size = str::parse::<usize>(size.as_str())?;
                } else {
                    print_help();
                    return Err(VibeError::new("missing the size portion of the --hidden-size flag"));
                }
            }
            "--learn-rate" => {
                if let Some(rate) = args.pop() {
                    options.learn_rate = str::parse::<f32>(rate.as_str())?;
                } else {
                    print_help();
                    return Err(VibeError::new("missing the rate portion of the --learn-rate flag"));
                }
            }
            "--generate" => {
                if let Some(count) = args.pop() {
                    options.generate = str::parse::<usize>(count.as_str())?;
                } else {
                    print_help();
                    return Err(VibeError::new("missing the number portion of the --generate flag"));
                }
            }
            _ => {
                print_help();
                return Err(VibeError::new(format!("unrecognized argument: {}", arg)));
            }
        }
    }

    Ok(())
}

// Print a usage help message.
fn print_help() {
    println!("usage:");
    println!("command");
    println!("\t--data           <data path>      ({})", DEFAULT_DATA_PATH);
    println!(
        "\t--device         <{}|{}|{}> ({})",
        device::DEVICE_NAME_CPU,
        device::DEVICE_NAME_CUDA,
        device::DEVICE_NAME_METAL,
        DEFAULT_DEVICE
    );
    println!("\t--iterations     <num>            ({})", DEFAULT_ITERATIONS);
    println!("\t--batch-size     <num>            ({})", DEFAULT_BATCH_SIZE);
    println!("\t--block-size     <num>            ({})", DEFAULT_BLOCK_SIZE);
    println!("\t--embedding-size <num>            ({})", DEFAULT_EMBEDDING_SIZE);
    println!("\t--hidden-size    <num>            ({})", DEFAULT_HIDDEN_SIZE);
    println!("\t--learn-rate     <rate>           ({})", DEFAULT_LEARN_RATE);
    println!("\t--generate       <num>            ({})", DEFAULT_GENERATE);
}
