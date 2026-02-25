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
    /// Stereo audio will be converted to mono by extracting only the left channel if channels > 1.
    pub fn start(&mut self, tx: mpsc::UnboundedSender<Vec<AudioSample>>) -> Result<(), InputError> {
        let requested_config = StreamConfig {
            channels: self.config.channels,
            sample_rate: cpal::SampleRate(self.config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.config.buffer_size as u32),
        };

        // Try requested config first (works on macOS), fall back to device default (needed for Windows WASAPI)
        let (stream_config, channels) = {
            // Test if the requested config works by checking supported configs
            let test_stream = self.device.build_input_stream(
                &requested_config,
                |_: &[f32], _: &cpal::InputCallbackInfo| {},
                |_| {},
                None,
            );
            
            if test_stream.is_ok() {
                drop(test_stream);
                log::info!("[AUDIO IN] Using requested config: {}Hz, {} ch, {} buf",
                    self.config.sample_rate, self.config.channels, self.config.buffer_size);
                (requested_config, self.config.channels as usize)
            } else {
                // Requested config not supported — query device default for channels
                log::info!("[AUDIO IN] Requested config not supported, querying device default...");
                let default_cfg = self.device.default_input_config();
                match default_cfg {
                    Ok(cfg) => {
                        let ch = cfg.channels();
                        log::info!("[AUDIO IN] Device default: {}Hz, {} ch — using {} ch with our sample rate",
                            cfg.sample_rate().0, ch, ch);
                        let fallback = StreamConfig {
                            channels: ch,
                            sample_rate: cpal::SampleRate(self.config.sample_rate),
                            buffer_size: cpal::BufferSize::Default,
                        };
                        (fallback, ch as usize)
                    }
                    Err(e) => {
                        log::warn!("[AUDIO IN] Cannot query default config: {}. Trying requested config anyway.", e);
                        (requested_config, self.config.channels as usize)
                    }
                }
            }
        };

        let channels = channels;
        let tx_f32 = Arc::new(tx);
        let tx_i16 = tx_f32.clone();
        let channels_i16 = channels;

        // Build the input stream
        // Try f32 first (works on macOS/Windows), fall back to i16 (needed for raw ALSA USB audio)
        let stream = self.device.build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Convert to mono if needed
                let samples = if channels == 1 {
                    data.to_vec()
                } else {
                    // EXTRACT LEFT CHANNEL ONLY (Bypass downmixing!)
                    // Ham radio USB Codecs route the audio to the Left channel.
                    // Averaging them destroys the MSK phase data.
                    data.chunks(channels)
                        .map(|chunk| chunk[0]) // Grab only the Left channel
                        .collect()
                };

                // Send to processing channel (ignore send errors if receiver dropped)
                let _ = tx_f32.send(samples);
            },
            move |err| {
                log::error!("Audio input stream error: {}", err);
            },
            None, // No timeout
        )
        .or_else(|e| {
            // f32 failed — try i16 (needed for raw ALSA with USB audio CODECs like PCM2901)
            log::info!("[AUDIO IN] f32 format not supported ({}), trying i16 (S16_LE)...", e);
            self.device.build_input_stream(
                &stream_config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let samples: Vec<f32> = if channels_i16 == 1 {
                        data.iter().map(|&s| s as f32 / 32768.0).collect()
                    } else {
                        // EXTRACT LEFT CHANNEL ONLY
                        data.chunks(channels_i16)
                            .map(|chunk| chunk[0] as f32 / 32768.0) // Grab only the Left channel
                            .collect()
                    };
                    let _ = tx_i16.send(samples);
                },
                move |err| {
                    log::error!("Audio input stream error: {}", err);
                },
                None,
            )
        })?;

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