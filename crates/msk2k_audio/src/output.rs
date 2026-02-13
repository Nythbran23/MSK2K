/// Audio output stream handling
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::device::DeviceError;
use crate::types::{AudioConfig, AudioSample};

/// Errors related to audio output
#[derive(Debug, thiserror::Error)]
pub enum OutputError {
    #[error("Device error: {0}")]
    DeviceError(#[from] DeviceError),

    #[error("Failed to build stream: {0}")]
    BuildStreamError(#[from] cpal::BuildStreamError),

    #[error("Failed to play stream: {0}")]
    PlayStreamError(#[from] cpal::PlayStreamError),

    #[error("Stream error: {0}")]
    StreamError(String),
}

/// Audio output stream
///
/// Note: This struct is Send despite containing a CPAL Stream because:
/// 1. The Stream is never accessed across thread boundaries
/// 2. The Stream is only created/destroyed, never shared
/// 3. All cross-thread communication happens via channels
pub struct AudioOutput {
    device: Device,
    config: AudioConfig,
    stream: Option<StreamHolder>,
}

/// Wrapper to hold the stream and mark it as Send
///
/// SAFETY: This is safe because:
/// - The Stream is never accessed after creation except to drop it
/// - All audio data flows through Send channels (mpsc)
/// - The callback captures only Send types (Arc<Mutex<Receiver>>)
struct StreamHolder {
    _stream: Stream,
}

// SAFETY: StreamHolder is Send because we never access the Stream across threads.
// The stream's callback captures only Send types, and the stream itself is only
// created and destroyed, never accessed.
unsafe impl Send for StreamHolder {}

impl AudioOutput {
    /// Create a new audio output
    pub fn new(device: Device, config: AudioConfig) -> Self {
        Self {
            device,
            config,
            stream: None,
        }
    }

    /// Start playing audio from the provided channel
    ///
    /// The channel should provide Vec<f32> buffers at the configured sample rate.
    /// Mono audio will be duplicated to all channels if channels > 1.
    pub fn start(
        &mut self,
        rx: mpsc::UnboundedReceiver<Vec<AudioSample>>,
    ) -> Result<(), OutputError> {
        let requested_config = StreamConfig {
            channels: self.config.channels,
            sample_rate: cpal::SampleRate(self.config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.config.buffer_size as u32),
        };

        // Try requested config first (works on macOS), fall back to device default (needed for Windows WASAPI)
        let (stream_config, channels) = {
            let test_stream = self.device.build_output_stream(
                &requested_config,
                |_: &mut [f32], _: &cpal::OutputCallbackInfo| {},
                |_| {},
                None,
            );

            if test_stream.is_ok() {
                drop(test_stream);
                log::info!("[AUDIO OUT] Using requested config: {}Hz, {} ch, {} buf",
                    self.config.sample_rate, self.config.channels, self.config.buffer_size);
                (requested_config, self.config.channels as usize)
            } else {
                log::info!("[AUDIO OUT] Requested config not supported, querying device default...");
                let default_cfg = self.device.default_output_config();
                match default_cfg {
                    Ok(cfg) => {
                        let ch = cfg.channels();
                        log::info!("[AUDIO OUT] Device default: {}Hz, {} ch — using {} ch with our sample rate",
                            cfg.sample_rate().0, ch, ch);
                        let fallback = StreamConfig {
                            channels: ch,
                            sample_rate: cpal::SampleRate(self.config.sample_rate),
                            buffer_size: cpal::BufferSize::Default,
                        };
                        (fallback, ch as usize)
                    }
                    Err(e) => {
                        log::warn!("[AUDIO OUT] Cannot query default config: {}. Trying requested config anyway.", e);
                        (requested_config, self.config.channels as usize)
                    }
                }
            }
        };

        let channels = channels;
        let rx = Arc::new(tokio::sync::Mutex::new(rx));

        // Buffer to hold samples between callbacks
        let sample_buffer: Arc<std::sync::Mutex<Vec<f32>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let sample_buffer_clone = sample_buffer.clone();

        // Build the output stream
        let stream = self.device.build_output_stream(
            &stream_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut buffer = sample_buffer_clone.lock().unwrap();

                // Try to refill buffer if it's running low
                if buffer.len() < data.len() * 2 {
                    if let Ok(new_samples) = rx.blocking_lock().try_recv() {
                        buffer.extend_from_slice(&new_samples);
                    }
                }

                if channels == 1 {
                    // Mono output - direct copy from buffer
                    let copy_len = data.len().min(buffer.len());
                    if copy_len > 0 {
                        data[..copy_len].copy_from_slice(&buffer[..copy_len]);
                        buffer.drain(..copy_len);
                    }

                    // Fill remainder with silence
                    if copy_len < data.len() {
                        data[copy_len..].fill(0.0);
                    }
                } else {
                    // Multi-channel output - duplicate mono to all channels
                    let frames = data.len() / channels;
                    let copy_frames = frames.min(buffer.len());

                    if copy_frames > 0 {
                        for i in 0..copy_frames {
                            let sample = buffer[i];
                            for ch in 0..channels {
                                data[i * channels + ch] = sample;
                            }
                        }
                        buffer.drain(..copy_frames);
                    }

                    // Fill remainder with silence
                    if copy_frames < frames {
                        data[copy_frames * channels..].fill(0.0);
                    }
                }
            },
            move |err| {
                log::error!("Audio output stream error: {}", err);
            },
            None, // No timeout
        )?;

        stream.play()?;
        self.stream = Some(StreamHolder { _stream: stream });

        log::info!("Audio output started: {}", self.config);
        Ok(())
    }

    /// Stop playing audio
    pub fn stop(&mut self) {
        if let Some(stream) = self.stream.take() {
            drop(stream);
            log::info!("Audio output stopped");
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

impl Drop for AudioOutput {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Builder for audio output
pub struct AudioOutputBuilder {
    device: Option<Device>,
    config: AudioConfig,
}

impl AudioOutputBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            device: None,
            config: AudioConfig::default(),
        }
    }

    /// Set the output device
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

    /// Build the audio output
    pub fn build(self) -> Result<AudioOutput, OutputError> {
        let device = self
            .device
            .ok_or_else(|| OutputError::DeviceError(DeviceError::NoDefaultDevice))?;

        Ok(AudioOutput::new(device, self.config))
    }
}

impl Default for AudioOutputBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceManager;

    #[test]
    fn test_output_builder() {
        let manager = DeviceManager::new().unwrap();
        if let Ok(device) = manager.default_output_device() {
            let output = AudioOutputBuilder::new()
                .device(device)
                .sample_rate(12000)
                .channels(1)
                .buffer_size(1024)
                .build();

            assert!(output.is_ok());
        }
    }
}
