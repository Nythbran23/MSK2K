/// Audio input stream handling
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::device::DeviceError;
use crate::types::{AudioConfig, AudioSample};

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

pub struct AudioInput {
    device: Device,
    config: AudioConfig,
    stream: Option<StreamHolder>,
}

#[cfg(not(target_os = "windows"))]
struct StreamHolder { _stream: Stream, }

#[cfg(target_os = "windows")]
struct StreamHolder {
    _cpal_stream: Option<Stream>,
    _wasapi_stop: Option<tokio::sync::oneshot::Sender<()>>,
}

unsafe impl Send for StreamHolder {}

impl AudioInput {
    pub fn new(device: Device, config: AudioConfig) -> Self {
        Self { device, config, stream: None }
    }

    pub fn start(&mut self, tx: mpsc::UnboundedSender<Vec<AudioSample>>) -> Result<(), InputError> {
        let requested_config = StreamConfig {
            channels: self.config.channels,
            sample_rate: cpal::SampleRate(self.config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.config.buffer_size as u32),
        };

        #[cfg(target_os = "windows")]
        {
            let device_name = self.device.name().unwrap_or_default();
            log::info!("[AUDIO IN] Windows OS detected. Attempting WASAPI Exclusive Mode for '{}'", device_name);
            
            match try_start_wasapi_capture(&device_name, &self.config, tx.clone()) {
                Ok(stop_tx) => {
                    log::info!("[AUDIO IN] ✅ WASAPI Exclusive Mode locked! Hardware format matched.");
                    self.stream = Some(StreamHolder { _cpal_stream: None, _wasapi_stop: Some(stop_tx) });
                    return Ok(());
                }
                Err(e) => {
                    log::warn!("[AUDIO IN] ⚠️ WASAPI Exclusive Mode unavailable ({}). Falling back to CPAL Shared Mode.", e);
                }
            }
        }

        let (stream_config, channels) = {
            let test_stream = self.device.build_input_stream(&requested_config, |_: &[f32], _| {}, |_| {}, None);
            if test_stream.is_ok() {
                drop(test_stream);
                (requested_config, self.config.channels as usize)
            } else {
                let default_cfg = self.device.default_input_config();
                match default_cfg {
                    Ok(cfg) => {
                        let ch = cfg.channels();
                        let fallback = StreamConfig { channels: ch, sample_rate: cpal::SampleRate(self.config.sample_rate), buffer_size: cpal::BufferSize::Default };
                        (fallback, ch as usize)
                    }
                    Err(_) => (requested_config, self.config.channels as usize)
                }
            }
        };

        let channels = channels;
        let tx_f32 = Arc::new(tx);
        let tx_i16 = tx_f32.clone();
        let channels_i16 = channels;

        let stream = self.device.build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let samples = if channels == 1 { data.to_vec() } else { data.chunks(channels).map(|chunk| chunk[0]).collect() };
                let _ = tx_f32.send(samples);
            },
            move |err| { log::error!("Audio input stream error: {}", err); },
            None,
        ).or_else(|_| {
            self.device.build_input_stream(
                &stream_config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let samples: Vec<f32> = if channels_i16 == 1 {
                        data.iter().map(|&s| s as f32 / 32768.0).collect()
                    } else {
                        data.chunks(channels_i16).map(|chunk| chunk[0] as f32 / 32768.0).collect()
                    };
                    let _ = tx_i16.send(samples);
                },
                move |err| { log::error!("Audio input stream error: {}", err); },
                None,
            )
        })?;

        stream.play()?;
        
        #[cfg(not(target_os = "windows"))]
        { self.stream = Some(StreamHolder { _stream: stream }); }
        
        #[cfg(target_os = "windows")]
        { self.stream = Some(StreamHolder { _cpal_stream: Some(stream), _wasapi_stop: None }); }

        log::info!("Audio input started: {}", self.config);
        Ok(())
    }

    pub fn stop(&mut self) {
        if let Some(stream) = self.stream.take() {
            #[cfg(not(target_os = "windows"))]
            drop(stream);
            
            #[cfg(target_os = "windows")]
            {
                if let Some(stop_tx) = stream._wasapi_stop { let _ = stop_tx.send(()); }
                drop(stream._cpal_stream);
            }
            log::info!("Audio input stopped");
        }
    }
}

impl Drop for AudioInput { fn drop(&mut self) { self.stop(); } }

#[cfg(target_os = "windows")]
fn try_start_wasapi_capture(
    device_name: &str,
    config: &AudioConfig,
    tx: mpsc::UnboundedSender<Vec<AudioSample>>,
) -> Result<tokio::sync::oneshot::Sender<()>, String> {
    let (init_tx, init_rx) = std::sync::mpsc::channel();
    let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel();
    
    let device_name = device_name.to_string();
    let config = config.clone();

    std::thread::spawn(move || {
        use wasapi::{initialize_mta, DeviceCollection, Direction, ShareMode, WaveFormat, SampleType};
        let _ = initialize_mta();

        let collection = match DeviceCollection::new(&Direction::Capture) {
            Ok(c) => c, Err(e) => { let _ = init_tx.send(Err(format!("Collection err: {}", e))); return; }
        };

        let mut target_dev = None;
        for i in 0..collection.get_nbr_devices().unwrap_or(0) {
            if let Ok(dev) = collection.get_device_at_index(i) {
                if let Ok(name) = dev.get_friendlyname() {
                    if name == device_name || device_name.contains(&name) || name.contains(&device_name) {
                        target_dev = Some(dev); break;
                    }
                }
            }
        }

        let device = match target_dev {
            Some(d) => d, None => { let _ = init_tx.send(Err("WASAPI Device not found".to_string())); return; }
        };

        let mut client = match device.get_iaudioclient() {
            Ok(c) => c, Err(e) => { let _ = init_tx.send(Err(e.to_string())); return; }
        };

        let mut format_16 = WaveFormat::new(16, 16, &SampleType::Int, config.sample_rate as usize, config.channels as usize, None);
        let mut actual_channels = config.channels as usize;

        if client.is_supported_exclusive_with_quirks(&format_16).is_err() {
            let alt_channels = if config.channels == 2 { 1 } else { 2 };
            format_16 = WaveFormat::new(16, 16, &SampleType::Int, config.sample_rate as usize, alt_channels, None);
            actual_channels = alt_channels;
        }

        if client.is_supported_exclusive_with_quirks(&format_16).is_err() || 
           client.initialize_client(&format_16, 0, &Direction::Capture, &ShareMode::Exclusive, false).is_err() {
            let _ = init_tx.send(Err("Hardware refused 48kHz 16-bit Exclusive Mode (Mono & Stereo)".to_string()));
            return;
        }

        let h_event = match client.set_get_eventhandle() { Ok(h) => h, Err(e) => { let _ = init_tx.send(Err(e.to_string())); return; }};
        let capture_client = match client.get_audiocaptureclient() { Ok(c) => c, Err(e) => { let _ = init_tx.send(Err(e.to_string())); return; }};
        if let Err(e) = client.start_stream() { let _ = init_tx.send(Err(e.to_string())); return; }

        if init_tx.send(Ok(())).is_err() { let _ = client.stop_stream(); return; }

        let channels = actual_channels;
        let bytes_per_frame = 2 * channels; 
        let mut byte_buffer = vec![0u8; 96000 * bytes_per_frame];

        loop {
            if stop_rx.try_recv().is_ok() { break; }
            if h_event.wait_for_event(1000).is_err() { break; }

            // 🟢 Event Fired! Read exactly what the hardware hands us and send it off.
            if let Ok((frames_read, _flags)) = capture_client.read_from_device(bytes_per_frame, &mut byte_buffer) {
                let valid_bytes = (frames_read as usize) * bytes_per_frame;
                if valid_bytes == 0 { continue; }
                
                let bytes = &byte_buffer[..valid_bytes];
                let mut f32_samples = Vec::with_capacity(frames_read as usize * channels);

                let ints: &[i16] = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i16, bytes.len() / 2) };
                if channels == 1 { 
                    for &s in ints { f32_samples.push(s as f32 / 32768.0); } 
                } else { 
                    for chunk in ints.chunks(channels) { if !chunk.is_empty() { f32_samples.push(chunk[0] as f32 / 32768.0); } } 
                }
                
                if tx.send(f32_samples).is_err() { break; }
            }
        }
        let _ = client.stop_stream();
    });

    match init_rx.recv() {
        Ok(Ok(())) => Ok(stop_tx),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("Thread died before initialization".to_string()),
    }
}

pub struct AudioInputBuilder { device: Option<Device>, config: AudioConfig }
impl AudioInputBuilder {
    pub fn new() -> Self { Self { device: None, config: AudioConfig::default() } }
    pub fn device(mut self, device: Device) -> Self { self.device = Some(device); self }
    pub fn config(mut self, config: AudioConfig) -> Self { self.config = config; self }
    pub fn build(self) -> Result<AudioInput, InputError> {
        let device = self.device.ok_or(InputError::DeviceError(DeviceError::NoDefaultDevice))?;
        Ok(AudioInput::new(device, self.config))
    }
}