/// Audio input stream handling
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::device::DeviceError;
use crate::types::{AudioConfig, AudioSample};

/// Errors related to audio input
#[derive(Debug, thiserror::Error)]
pub enum InputError {
    #[error("Device error: {0}")]
    DeviceError(#[from] DeviceError),

    #[error("Failed to build stream: {0}")]
    BuildStreamError(#[from] cpal::BuildStreamError),

    #[error("Failed to play stream: {0}")]
    PlayStreamError(#[from] cpal::PlayStreamError),

    #[error("Stream error: {0}")]
    StreamError(String),
}

/// Audio input stream
///
/// Note: This struct is Send despite containing a CPAL Stream because:
/// 1. The Stream is never accessed across thread boundaries
/// 2. The Stream is only created/destroyed, never shared
/// 3. All cross-thread communication happens via channels
pub struct AudioInput {
    device: Device,
    config: AudioConfig,
    stream: Option<StreamHolder>,
}

/// Wrapper to hold the stream and mark it as Send
///
/// SAFETY: This is safe because:
/// - The Stream is never accessed after creation except to drop it
/// - All audio data flows through Send channels (mpsc)
/// - The callback captures only Send types (Arc<UnboundedSender>)
struct StreamHolder {
    _stream: Stream,
}

// SAFETY: StreamHolder is Send because we never access the Stream across threads.
// The stream's callback captures only Send types, and the stream itself is only
// created and destroyed, never accessed.
unsafe impl Send for StreamHolder {}

impl AudioInput {
    /// Create a new audio input
    pub fn new(device: Device, config: AudioConfig) -> Self {
        Self {
            device,
            config,
            stream: None,
        }
    }

    /// Start capturing audio and send samples to the provided channel
    ///
    /// The channel will receive Vec<f32> buffers at the configured sample rate.
    /// Stereo audio will be downmixed to mono if channels > 1.
    pub fn start(&mut self, tx: mpsc::UnboundedSender<Vec<AudioSample>>) -> Result<(), InputError> {
        // Query the device's default config to get supported channels
        let default_cfg = self.device.default_input_config();
        let (use_channels, use_buffer_size) = match &default_cfg {
            Ok(cfg) => {
                let ch = cfg.channels();
                log::info!("[AUDIO IN] Device default config: {}Hz, {} ch, {:?}", 
                    cfg.sample_rate().0, ch, cfg.sample_format());
                (ch, cpal::BufferSize::Default)
            }
            Err(e) => {
                log::warn!("[AUDIO IN] Could not query default config: {}, using requested config", e);
                (self.config.channels, cpal::BufferSize::Fixed(self.config.buffer_size as u32))
            }
        };

        let stream_config = StreamConfig {
            channels: use_channels,
            sample_rate: cpal::SampleRate(self.config.sample_rate),
            buffer_size: use_buffer_size,
        };

        log::info!("[AUDIO IN] Opening stream: {}Hz, {} ch, buffer={:?}", 
            self.config.sample_rate, use_channels, use_buffer_size);

        let channels = use_channels as usize;
        let tx = Arc::new(tx);

        // Build the input stream
        let stream = self.device.build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Convert to mono if needed
                let samples = if channels == 1 {
                    data.to_vec()
                } else {
                    // Downmix to mono by averaging channels
                    data.chunks(channels)
                        .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                        .collect()
                };

                // Send to processing channel (ignore send errors if receiver dropped)
                let _ = tx.send(samples);
            },
            move |err| {
                log::error!("Audio input stream error: {}", err);
            },
            None, // No timeout
        )?;

        stream.play()?;
        self.stream = Some(StreamHolder { _stream: stream });

        log::info!("Audio input started: {}", self.config);
        Ok(())
    }

    /// Stop capturing audio
    pub fn stop(&mut self) {
        if let Some(stream) = self.stream.take() {
            drop(stream);
            log::info!("Audio input stopped");
        }
    }

    /// Check if the stream is active
    pub fn is_active(&self) -> bool {
        self.stream.is_some()
    }

    /// Get the current configuration
    pub fn config(&self) -> &AudioConfig {
        &self.config
    }
}

impl Drop for AudioInput {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Builder for audio input
pub struct AudioInputBuilder {
    device: Option<Device>,
    config: AudioConfig,
}

impl AudioInputBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            device: None,
            config: AudioConfig::default(),
        }
    }

    /// Set the input device
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Set the audio configuration
    pub fn config(mut self, config: AudioConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the sample rate
    pub fn sample_rate(mut self, sample_rate: u32) -> Self {
        self.config.sample_rate = sample_rate;
        self
    }

    /// Set the number of channels
    pub fn channels(mut self, channels: u16) -> Self {
        self.config.channels = channels;
        self
    }

    /// Set the buffer size
    pub fn buffer_size(mut self, buffer_size: usize) -> Self {
        self.config.buffer_size = buffer_size;
        self
    }

    /// Build the audio input
    pub fn build(self) -> Result<AudioInput, InputError> {
        let device = self
            .device
            .ok_or_else(|| InputError::DeviceError(DeviceError::NoDefaultDevice))?;

        Ok(AudioInput::new(device, self.config))
    }
}

impl Default for AudioInputBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceManager;

    #[test]
    fn test_input_builder() {
        let manager = DeviceManager::new().unwrap();
        if let Ok(device) = manager.default_input_device() {
            let input = AudioInputBuilder::new()
                .device(device)
                .sample_rate(12000)
                .channels(1)
                .buffer_size(1024)
                .build();

            assert!(input.is_ok());
        }
    }
}
