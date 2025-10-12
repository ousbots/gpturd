use crate::error::VibeError;

use candle_core::Device;

pub const DEVICE_NAME_CPU: &str = "cpu";
pub const DEVICE_NAME_CUDA: &str = "cuda";
pub const DEVICE_NAME_METAL: &str = "metal";

// Open the given device for data processing.
pub fn open_device(device: &String) -> Result<Device, VibeError> {
    match device.trim().to_lowercase().as_str() {
        DEVICE_NAME_CPU => Ok(Device::Cpu),
        DEVICE_NAME_CUDA => {
            Ok(Device::new_cuda(0).map_err(|e| VibeError::new(format!("unable to open cuda device: {}", e)))?)
        }
        DEVICE_NAME_METAL => {
            Ok(Device::new_metal(0).map_err(|e| VibeError::new(format!("unable to open metal device: {}", e)))?)
        }
        _ => Err(VibeError::new("invalid device")),
    }
}
