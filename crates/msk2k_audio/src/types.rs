/// Common types for audio I/O
use std::fmt;

/// Audio sample rate in Hz
pub type SampleRate = u32;

/// Audio sample as f32 (normalized -1.0 to 1.0)
pub type AudioSample = f32;

/// Standard sample rates for amateur radio digital modes
pub const SAMPLE_RATE_8KHZ: SampleRate = 8000;
pub const SAMPLE_RATE_12KHZ: SampleRate = 12000;
pub const SAMPLE_RATE_48KHZ: SampleRate = 48000;

/// MSK2K uses 12kHz sample rate
pub const MSK2K_SAMPLE_RATE: SampleRate = SAMPLE_RATE_48KHZ;

/// Audio buffer configuration
#[derive(Debug, Clone, Copy)]
pub struct AudioConfig {
    /// Sample rate in Hz
    pub sample_rate: SampleRate,

    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u16,

    /// Buffer size in frames
    pub buffer_size: usize,
}

impl AudioConfig {
    /// Create a new audio configuration
    pub fn new(sample_rate: SampleRate, channels: u16, buffer_size: usize) -> Self {
        Self {
            sample_rate,
            channels,
            buffer_size,
        }
    }

    /// MSK2K standard configuration (12kHz mono)
    pub fn msk2k_default() -> Self {
        Self {
            sample_rate: MSK2K_SAMPLE_RATE,
            channels: 1,
            buffer_size: 1024,
        }
    }

    /// Calculate buffer duration in milliseconds
    pub fn buffer_duration_ms(&self) -> f32 {
        (self.buffer_size as f32 / self.sample_rate as f32) * 1000.0
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self::msk2k_default()
    }
}

impl fmt::Display for AudioConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}Hz, {} ch, {} samples ({:.1}ms)",
            self.sample_rate,
            self.channels,
            self.buffer_size,
            self.buffer_duration_ms()
        )
    }
}

/// Audio device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub is_default: bool,
    pub supported_sample_rates: Vec<SampleRate>,
    pub max_channels: u16,
}

impl fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{} - {} ch, rates: {:?}",
            self.name,
            if self.is_default { " (default)" } else { "" },
            self.max_channels,
            self.supported_sample_rates
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_config_defaults() {
        let config = AudioConfig::default();
        assert_eq!(config.sample_rate, MSK2K_SAMPLE_RATE);
        assert_eq!(config.channels, 1);
    }

    #[test]
    fn test_buffer_duration() {
        let config = AudioConfig::new(12000, 1, 1200);
        assert_eq!(config.buffer_duration_ms(), 100.0);
    }
}
