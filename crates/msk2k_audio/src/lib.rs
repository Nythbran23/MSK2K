//! MSK2K Audio I/O
//!
//! This crate provides real-time audio input/output capabilities for MSK2K
//! digital mode operations. It uses the `cpal` library for cross-platform
//! audio device access and `rubato` for sample rate conversion.
//!
//! # Features
//!
//! - Device enumeration and selection
//! - Real-time audio input capture
//! - Real-time audio output playback
//! - Sample rate conversion (48kHz → 12kHz for MSK2K)
//! - Automatic mono/stereo conversion
//! - Low-latency streaming with configurable buffer sizes
//!
//! # Example
//!
//! ```no_run
//! use msk2k_audio::{DeviceManager, AudioInputBuilder, AudioConfig};
//! use tokio::sync::mpsc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let manager = DeviceManager::new()?;
//!     let device = manager.default_input_device()?;
//!     
//!     let (tx, mut rx) = mpsc::unbounded_channel();
//!     
//!     let mut input = AudioInputBuilder::new()
//!         .device(device)
//!         .sample_rate(12000)
//!         .channels(1)
//!         .buffer_size(1024)
//!         .build()?;
//!     
//!     input.start(tx)?;
//!     
//!     // Process audio samples
//!     while let Some(samples) = rx.recv().await {
//!         println!("Received {} samples", samples.len());
//!     }
//!     
//!     Ok(())
//! }
//! ```

pub mod device;
pub mod input;
pub mod output;
pub mod resampler;
pub mod types;

// Re-export main types
pub use device::{DeviceError, DeviceManager};
pub use input::{AudioInput, AudioInputBuilder, InputError};
pub use output::{AudioOutput, AudioOutputBuilder, OutputError};
pub use resampler::{needs_resampling, AudioResampler, ResamplerError};
pub use types::{AudioConfig, AudioSample, DeviceInfo, SampleRate};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// MSK2K standard sample rate (12 kHz)
pub const MSK2K_SAMPLE_RATE: SampleRate = types::MSK2K_SAMPLE_RATE;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
