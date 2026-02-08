/// Audio device enumeration and selection
use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{Device, Host, SampleRate, SupportedStreamConfig};
use std::fmt;

use crate::types::{DeviceInfo, SampleRate as AudioSampleRate};

/// Errors related to audio devices
#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    #[error("No audio host available")]
    NoHost,

    #[error("No input devices found")]
    NoInputDevices,

    #[error("No output devices found")]
    NoOutputDevices,

    #[error("Device not found: {0}")]
    DeviceNotFound(String),

    #[error("Failed to get device config: {0}")]
    ConfigError(#[from] cpal::SupportedStreamConfigsError),

    #[error("Failed to get default config: {0}")]
    DefaultConfigError(#[from] cpal::DefaultStreamConfigError),

    #[error("Failed to enumerate devices: {0}")]
    DevicesError(#[from] cpal::DevicesError),

    #[error("Device error: {0}")]
    CpalDeviceError(#[from] cpal::DeviceNameError),

    #[error("Default device not available")]
    NoDefaultDevice,
}

/// Audio device manager
pub struct DeviceManager {
    host: Host,
}

impl DeviceManager {
    /// Create a new device manager
    pub fn new() -> Result<Self, DeviceError> {
        let host = cpal::default_host();
        Ok(Self { host })
    }

    /// List all input devices
    pub fn list_input_devices(&self) -> Result<Vec<DeviceInfo>, DeviceError> {
        let default_device = self.host.default_input_device();
        let default_name = default_device.as_ref().and_then(|d| d.name().ok());

        let devices: Result<Vec<_>, _> = self
            .host
            .input_devices()?
            .map(|device| self.get_device_info(device, &default_name))
            .collect();

        devices.map_err(|_| DeviceError::NoInputDevices)
    }

    /// List all output devices
    pub fn list_output_devices(&self) -> Result<Vec<DeviceInfo>, DeviceError> {
        let default_device = self.host.default_output_device();
        let default_name = default_device.as_ref().and_then(|d| d.name().ok());

        let devices: Result<Vec<_>, _> = self
            .host
            .output_devices()?
            .map(|device| self.get_device_info(device, &default_name))
            .collect();

        devices.map_err(|_| DeviceError::NoOutputDevices)
    }

    /// Get default input device
    pub fn default_input_device(&self) -> Result<Device, DeviceError> {
        self.host
            .default_input_device()
            .ok_or(DeviceError::NoDefaultDevice)
    }

    /// Get default output device
    pub fn default_output_device(&self) -> Result<Device, DeviceError> {
        self.host
            .default_output_device()
            .ok_or(DeviceError::NoDefaultDevice)
    }

    /// Find input device by name (or default if None)
    pub fn get_input_device(&self, name: Option<&str>) -> Result<Device, DeviceError> {
        if let Some(device_name) = name {
            self.find_device_by_name(device_name, true)
        } else {
            self.default_input_device()
        }
    }

    /// Find output device by name (or default if None)
    pub fn get_output_device(&self, name: Option<&str>) -> Result<Device, DeviceError> {
        if let Some(device_name) = name {
            self.find_device_by_name(device_name, false)
        } else {
            self.default_output_device()
        }
    }

    /// Get the best supported config for a device at the desired sample rate
    pub fn get_supported_config(
        &self,
        device: &Device,
        desired_rate: AudioSampleRate,
        is_input: bool,
    ) -> Result<SupportedStreamConfig, DeviceError> {
        let configs: Vec<_> = if is_input {
            device.supported_input_configs()?.collect()
        } else {
            device.supported_output_configs()?.collect()
        };

        // Try to find exact match
        for config_range in &configs {
            if config_range.min_sample_rate().0 <= desired_rate
                && config_range.max_sample_rate().0 >= desired_rate
            {
                return Ok(config_range.with_sample_rate(SampleRate(desired_rate)));
            }
        }

        // Fall back to default config
        if is_input {
            Ok(device.default_input_config()?)
        } else {
            Ok(device.default_output_config()?)
        }
    }

    /// Helper: Find device by name
    fn find_device_by_name(&self, name: &str, is_input: bool) -> Result<Device, DeviceError> {
        let devices = if is_input {
            self.host.input_devices()?
        } else {
            self.host.output_devices()?
        };

        for device in devices {
            if let Ok(device_name) = device.name() {
                if device_name.contains(name) || name.contains(&device_name) {
                    return Ok(device);
                }
            }
        }

        Err(DeviceError::DeviceNotFound(name.to_string()))
    }

    /// Helper: Extract device info
    fn get_device_info(
        &self,
        device: Device,
        default_name: &Option<String>,
    ) -> Result<DeviceInfo, DeviceError> {
        let name = device.name()?;
        let is_default = default_name.as_ref().map_or(false, |d| d == &name);

        // Try to get supported configs - try input first, fall back to output
        let configs_result = device.supported_input_configs();
        let configs: Vec<_> = if configs_result.is_ok() {
            configs_result?.collect()
        } else {
            // If input configs fail, try output configs
            device.supported_output_configs()?.collect()
        };

        let supported_sample_rates: Vec<_> = configs
            .iter()
            .flat_map(|config| {
                let min = config.min_sample_rate().0;
                let max = config.max_sample_rate().0;

                // Common sample rates within this range
                [8000, 12000, 16000, 22050, 44100, 48000, 96000]
                    .iter()
                    .copied()
                    .filter(|&rate| rate >= min && rate <= max)
                    .collect::<Vec<_>>()
            })
            .collect();

        let max_channels = configs.iter().map(|c| c.channels()).max().unwrap_or(2);

        Ok(DeviceInfo {
            name,
            is_default,
            supported_sample_rates,
            max_channels,
        })
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new().expect("Failed to create device manager")
    }
}

impl fmt::Debug for DeviceManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceManager")
            .field("host", &"cpal::Host")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_device_manager() {
        let manager = DeviceManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_list_devices() {
        let manager = DeviceManager::new().unwrap();

        // Should be able to list devices (may be empty on some systems)
        let _ = manager.list_input_devices();
        let _ = manager.list_output_devices();
    }
}
