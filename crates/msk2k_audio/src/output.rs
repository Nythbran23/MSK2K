/// Audio output stream handling
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::device::DeviceError;
use crate::types::{AudioConfig, AudioSample};

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

pub struct AudioOutput {
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

impl AudioOutput {
    pub fn new(device: Device, config: AudioConfig) -> Self {
        Self { device, config, stream: None }
    }

    pub fn start(&mut self, rx: mpsc::UnboundedReceiver<Vec<AudioSample>>) -> Result<(), OutputError> {
        let requested_config = StreamConfig {
            channels: self.config.channels,
            sample_rate: cpal::SampleRate(self.config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.config.buffer_size as u32),
        };
        
        let rx_mutex = Arc::new(tokio::sync::Mutex::new(rx));

        #[cfg(target_os = "windows")]
        {
            let device_name = self.device.name().unwrap_or_default();
            log::info!("[AUDIO OUT] Windows OS detected. Attempting WASAPI Exclusive Mode for '{}'", device_name);
            
            match try_start_wasapi_render(&device_name, &self.config, rx_mutex.clone()) {
                Ok(stop_tx) => {
                    log::info!("[AUDIO OUT] ✅ WASAPI Exclusive Mode locked! Transmitting bit-perfect MSK audio.");
                    self.stream = Some(StreamHolder { _cpal_stream: None, _wasapi_stop: Some(stop_tx) });
                    return Ok(());
                }
                Err(e) => {
                    log::warn!("[AUDIO OUT] ⚠️ WASAPI Exclusive Mode unavailable ({}). Falling back to CPAL Shared Mode.", e);
                }
            }
        }

        let (stream_config, channels) = {
            let test_stream = self.device.build_output_stream(&requested_config, |_: &mut [f32], _| {}, |_| {}, None);
            if test_stream.is_ok() {
                drop(test_stream);
                (requested_config, self.config.channels as usize)
            } else {
                let default_cfg = self.device.default_output_config();
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
        let sample_buffer: Arc<std::sync::Mutex<Vec<f32>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sample_buffer_f32 = sample_buffer.clone();
        let sample_buffer_i16 = sample_buffer.clone();

        let rx_f32 = rx_mutex.clone();
        let rx_i16 = rx_mutex.clone();
        let channels_i16 = channels;

        let stream = self.device.build_output_stream(
            &stream_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut buffer = sample_buffer_f32.lock().unwrap();
                if buffer.len() < data.len() * 2 {
                    if let Ok(new_samples) = rx_f32.blocking_lock().try_recv() { buffer.extend_from_slice(&new_samples); }
                }

                if channels == 1 {
                    let copy_len = data.len().min(buffer.len());
                    if copy_len > 0 { data[..copy_len].copy_from_slice(&buffer[..copy_len]); buffer.drain(..copy_len); }
                    if copy_len < data.len() { data[copy_len..].fill(0.0); }
                } else {
                    let frames = data.len() / channels;
                    let copy_frames = frames.min(buffer.len());
                    if copy_frames > 0 {
                        for i in 0..copy_frames {
                            let sample = buffer[i];
                            for ch in 0..channels { data[i * channels + ch] = sample; }
                        }
                        buffer.drain(..copy_frames);
                    }
                    if copy_frames < frames { data[copy_frames * channels..].fill(0.0); }
                }
            },
            move |err| { log::error!("Audio output stream error: {}", err); },
            None,
        ).or_else(|_| {
            self.device.build_output_stream(
                &stream_config,
                move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                    let mut buffer = sample_buffer_i16.lock().unwrap();
                    let frames = data.len() / channels_i16;
                    if buffer.len() < frames * 2 {
                        if let Ok(new_samples) = rx_i16.blocking_lock().try_recv() { buffer.extend_from_slice(&new_samples); }
                    }

                    if channels_i16 == 1 {
                        let copy_len = data.len().min(buffer.len());
                        if copy_len > 0 {
                            for i in 0..copy_len { data[i] = (buffer[i] * 32767.0).clamp(-32768.0, 32767.0) as i16; }
                            buffer.drain(..copy_len);
                        }
                        if copy_len < data.len() { data[copy_len..].fill(0); }
                    } else {
                        let copy_frames = frames.min(buffer.len());
                        if copy_frames > 0 {
                            for i in 0..copy_frames {
                                let sample_i16 = (buffer[i] * 32767.0).clamp(-32768.0, 32767.0) as i16;
                                for ch in 0..channels_i16 { data[i * channels_i16 + ch] = sample_i16; }
                            }
                            buffer.drain(..copy_frames);
                        }
                        if copy_frames < frames { data[copy_frames * channels_i16..].fill(0); }
                    }
                },
                move |err| { log::error!("Audio output stream error: {}", err); },
                None,
            )
        })?;

        stream.play()?;
        
        #[cfg(not(target_os = "windows"))]
        { self.stream = Some(StreamHolder { _stream: stream }); }
        
        #[cfg(target_os = "windows")]
        { self.stream = Some(StreamHolder { _cpal_stream: Some(stream), _wasapi_stop: None }); }

        log::info!("Audio output started: {}", self.config);
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
            log::info!("Audio output stopped");
        }
    }
}

impl Drop for AudioOutput { fn drop(&mut self) { self.stop(); } }

#[cfg(target_os = "windows")]
fn try_start_wasapi_render(
    device_name: &str,
    config: &AudioConfig,
    rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<Vec<AudioSample>>>>
) -> Result<tokio::sync::oneshot::Sender<()>, String> {
    let (init_tx, init_rx) = std::sync::mpsc::channel();
    let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel();
    
    let device_name = device_name.to_string();
    let config = config.clone();

    std::thread::spawn(move || {
        use wasapi::{initialize_mta, DeviceCollection, Direction, ShareMode, WaveFormat, SampleType};
        let _ = initialize_mta();

        let collection = match DeviceCollection::new(&Direction::Render) {
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

        // 🟢 SMART FALLBACK: Try requested channels, pivot if rejected
        let mut format_16 = WaveFormat::new(16, 16, &SampleType::Int, config.sample_rate as usize, config.channels as usize, None);
        let mut actual_channels = config.channels as usize;

        if client.is_supported_exclusive_with_quirks(&format_16).is_err() {
            let alt_channels = if config.channels == 2 { 1 } else { 2 };
            format_16 = WaveFormat::new(16, 16, &SampleType::Int, config.sample_rate as usize, alt_channels, None);
            actual_channels = alt_channels;
        }

        if client.is_supported_exclusive_with_quirks(&format_16).is_err() || 
           client.initialize_client(&format_16, 0, &Direction::Render, &ShareMode::Exclusive, false).is_err() {
            let _ = init_tx.send(Err("Hardware refused 48kHz 16-bit Exclusive Mode (Mono & Stereo)".to_string())); 
            return;
        }

        let h_event = match client.set_get_eventhandle() { Ok(h) => h, Err(e) => { let _ = init_tx.send(Err(e.to_string())); return; }};
        let render_client = match client.get_audiorenderclient() { Ok(c) => c, Err(e) => { let _ = init_tx.send(Err(e.to_string())); return; }};
        if let Err(e) = client.start_stream() { let _ = init_tx.send(Err(e.to_string())); return; }

        if init_tx.send(Ok(())).is_err() { let _ = client.stop_stream(); return; }

        let channels = actual_channels;
        let mut sample_buffer = Vec::new();
        let byte_per_frame = 2 * channels; // 16-bit = 2 bytes

        loop {
            if stop_rx.try_recv().is_ok() { break; }
            if h_event.wait_for_event(1000).is_err() { break; }

            let frames_available = client.get_available_space_in_frames().unwrap_or(0) as usize;
            if frames_available > 0 {
                
                if let Ok(mut lock) = rx.try_lock() {
                    while sample_buffer.len() < frames_available {
                        match lock.try_recv() {
                            Ok(new_samples) => sample_buffer.extend_from_slice(&new_samples),
                            Err(_) => break, 
                        }
                    }
                }

                let copy_frames = frames_available.min(sample_buffer.len());
                let mut data = vec![0i16; frames_available * channels]; 
                
                for i in 0..copy_frames {
                    let s = (sample_buffer[i] * 32767.0).clamp(-32768.0, 32767.0) as i16;
                    for ch in 0..channels { data[i * channels + ch] = s; }
                }
                
                sample_buffer.drain(..copy_frames);
                let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
                let _ = render_client.write_to_device(frames_available, byte_per_frame, bytes, None);
            }
        }
        let _ = client.stop_stream();
    });

    match init_rx.recv() {
        Ok(Ok(())) => Ok(stop_tx),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("Audio thread panicked during init".to_string())
    }
}

pub struct AudioOutputBuilder { device: Option<Device>, config: AudioConfig }
impl AudioOutputBuilder {
    pub fn new() -> Self { Self { device: None, config: AudioConfig::default() } }
    pub fn device(mut self, device: Device) -> Self { self.device = Some(device); self }
    pub fn config(mut self, config: AudioConfig) -> Self { self.config = config; self }
    pub fn build(self) -> Result<AudioOutput, OutputError> {
        let device = self.device.ok_or(OutputError::DeviceError(DeviceError::NoDefaultDevice))?;
        Ok(AudioOutput::new(device, self.config))
    }
}