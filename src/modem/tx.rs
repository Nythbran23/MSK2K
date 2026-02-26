// src/modem/tx.rs

use anyhow::{anyhow, Result};
use log::{info, warn};
use tokio::sync::mpsc;

use msk2k_audio::{AudioConfig, AudioOutput, AudioOutputBuilder, DeviceManager};
use msk2k_dsp::callsign::CallsignCodec;
use msk2k_dsp::message::Message;
use msk2k_dsp::{fec, fmt1, fmt2};

use std::f32::consts::PI;

use crate::modem::TxRequest;

const SYMBOL_RATE: u32 = 2_000;
const CENTER_FREQ: f32 = 1350.0;
const FREQ_DEV: f32 = 500.0;

const GENERAL_ADDRESS_49: [i32; 49] = [
    1,1,0,1,0,1,0,0,0,0,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,
    0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1
];

#[derive(Debug, Clone)]
struct TxAudioCfg {
    output_device: Option<String>,
    sample_rate: u32,
    buffer_size: usize,
    output_level: f32,
    my_call: String,
    their_call: String,
}

impl Default for TxAudioCfg {
    fn default() -> Self {
        Self {
            output_device: None,
            sample_rate: 48_000,
            buffer_size: 1024,
            output_level: 0.4,
            my_call: String::new(),
            their_call: String::new(),
        }
    }
}

pub async fn run_transmitter_task(
    mut tx_req_rx: mpsc::UnboundedReceiver<TxRequest>,
) -> Result<()> {
    let mut cfg = TxAudioCfg::default();
    let codec = CallsignCodec::new();

    info!("[TX] worker started");

    while let Some(req) = tx_req_rx.recv().await {
        match req {
            TxRequest::ApplyAudio {
                output_device,
                output_level,
                sample_rate,
                buffer_size,
                my_call,
                their_call,
            } => {
                cfg.output_device = output_device;
                cfg.output_level = output_level;
                cfg.sample_rate = sample_rate;
                cfg.buffer_size = buffer_size;
                cfg.my_call = my_call.clone();
                cfg.their_call = their_call.clone();

                info!(
                    "[TX] ApplyAudio dev={:?} sr={} buf={} my={} their={}",
                    cfg.output_device, cfg.sample_rate, cfg.buffer_size,
                    cfg.my_call, cfg.their_call
                );
            }

            TxRequest::Stop => {
                info!("[TX] STOP");
            }

            // 🟢 GRID MODE: Raw bits from runtime (already 71 bits)
            TxRequest::RawBits { bits, slot_len_ms, my_call, their_call: _ } => {
                let (out_stream, audio_tx) = match build_output_and_sender(&cfg) {
                    Ok(res) => res,
                    Err(e) => {
                         warn!("[TX] Failed to open audio: {}", e);
                         continue;
                    }
                };

                info!("[TX] CQ+Grid request from={} (71-bit packet)", my_call);

                // 🟢 CRITICAL: Bits are already 71 bits from encode_cq_with_grid()
                // They contain: 55 data + 2 type + 15 CRC
                let waveform = match generate_transmission_from_bits(&bits, &codec, cfg.sample_rate, slot_len_ms) {
                    Ok(w) => w,
                    Err(e) => {
                        warn!("[TX] Failed to generate grid waveform: {}", e);
                        continue;
                    }
                };

                let output_scalar = cfg.output_level;
                let scaled: Vec<f32> = waveform.iter().map(|&s| s * output_scalar).collect();

                if let Err(e) = audio_tx.send(scaled.clone()) {
                    warn!("[TX] Failed to send grid waveform: {}", e);
                } else {
                    let duration_secs = scaled.len() as f32 / cfg.sample_rate as f32;
                    info!("[TX] 📤 Sent {} Grid samples ({:.2}s)", scaled.len(), duration_secs);
                    drop(audio_tx);
                    let wait_ms = (duration_secs * 1000.0) as u64 + 500;
                    tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;
                }
                drop(out_stream);
            }

            TxRequest::Text { rendered, slot_len_ms, my_call, their_call } => {
                let (out_stream, audio_tx) = match build_output_and_sender(&cfg) {
                    Ok(res) => res,
                    Err(e) => {
                         warn!("[TX] Failed to open audio: {}", e);
                         continue;
                    }
                };

                info!("[TX] Text request: '{}' my={} their={}", rendered, my_call, their_call);

                let is_short_fmt2 = is_short_format2(&rendered);
                let msg_result = if is_short_fmt2 {
                    message_from_short_format2(&rendered, &my_call, &their_call)
                } else {
                    message_from_rendered(&rendered)
                };
                
                let msg = match msg_result {
                    Ok(m) => m,
                    Err(e) => {
                        warn!("[TX] Failed to parse '{}': {} - SKIPPING TX", rendered, e);
                        continue;
                    }
                };

                let waveform = match generate_transmission(&msg, &codec, cfg.sample_rate, slot_len_ms) {
                    Ok(w) => w,
                    Err(e) => {
                        warn!("[TX] Failed to generate waveform: {} - SKIPPING TX", e);
                        continue;
                    }
                };

                let output_scalar = cfg.output_level;
                let scaled: Vec<f32> = waveform.iter().map(|&s| s * output_scalar).collect();

                if let Err(e) = audio_tx.send(scaled.clone()) {
                    warn!("[TX] Failed to send waveform to audio driver: {}", e);
                } else {
                    let duration_secs = scaled.len() as f32 / cfg.sample_rate as f32;
                    info!("[TX] 📤 Sent {} samples ({:.2}s)", scaled.len(), duration_secs);
                    drop(audio_tx);
                    let wait_ms = (duration_secs * 1000.0) as u64 + 500;
                    tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;
                    info!("[TX] ✅ TX complete");
                }
                drop(out_stream);
            }
        }
    }
    Ok(())
}

/// 🟢 NEW: Waveform generator for 71-bit CQ+Grid packets
fn generate_transmission_from_bits(
    bits: &[i32],
    _codec: &CallsignCodec, 
    sample_rate: u32,
    slot_len_ms: u32,
) -> Result<Vec<f32>> {
    // 🟢 CRITICAL: Input is already 71 bits (55 data + 2 type + 15 CRC)
    // This is the standard Format 1 information block size
    
    if bits.len() != 71 {
        return Err(anyhow!("Grid packet must be exactly 71 bits, got {}", bits.len()));
    }
    
    // Convert to array
    let info71 = vec_to_array::<71>(bits.to_vec(), "Info71")?;
    
    // Use general address (CQ)
    let addr49 = GENERAL_ADDRESS_49; 

    // 🟢 FEC ENCODING: Standard Format 1 convolutional encoding
    let (p1, p2) = fec::encode_format1(&info71);
    
    let poly1 = vec_to_array::<83>(p1, "Poly1")?;
    let poly2 = vec_to_array::<83>(p2, "Poly2")?;
    let sync = fmt1::SYNC_PATTERN_HD43;
    
    // 🟢 INTERLEAVING: Standard Format 1 pattern
    let packet258 = vec_to_array::<258>(
        fmt1::interleave_format1(&sync, &addr49, &poly1, &poly2), 
        "Packet"
    )?;
    
    // 🟢 MODULATION: Standard MSK
    let single_packet_symbols: Vec<i32> = packet258.iter().map(|&b| if b == 0 { 1 } else { -1 }).collect();
    
    let packet_duration_s = 258.0 / SYMBOL_RATE as f32;
    let target_duration_s = slot_len_ms as f32 / 1000.0;
    let repeats = (target_duration_s / packet_duration_s).ceil() as usize;

    let mut continuous_symbols = Vec::with_capacity(single_packet_symbols.len() * repeats);
    for _ in 0..repeats {
        continuous_symbols.extend_from_slice(&single_packet_symbols);
    }

    let mut full_waveform = generate_msk(&continuous_symbols, sample_rate, (sample_rate / SYMBOL_RATE) as usize);

    let target_samples = (target_duration_s * sample_rate as f32) as usize;
    if full_waveform.len() > target_samples {
        full_waveform.truncate(target_samples);
    }

    Ok(full_waveform)
}

fn build_output_and_sender(cfg: &TxAudioCfg) -> Result<(AudioOutput, mpsc::UnboundedSender<Vec<f32>>)> {
    let manager = DeviceManager::new()?;
    let device = if let Some(ref name) = cfg.output_device {
        log::info!("[TX] Opening output device: '{}'", name);
        find_nth_output_device(name).unwrap_or_else(|| {
            let (base, _) = parse_device_suffix(name);
            log::warn!("[TX] find_nth_output_device failed, trying DeviceManager with base='{}' then full='{}'", base, name);
            manager.get_output_device(Some(&base))
                .or_else(|_| manager.get_output_device(Some(name)))
                .or_else(|_| {
                    log::warn!("[TX] All device lookups failed, using system default output");
                    manager.default_output_device()
                })
                .unwrap()
        })
    } else {
        log::info!("[TX] No output device configured, using system default");
        manager.default_output_device()?
    };
    log::info!("[TX] Output device opened successfully");
    let audio_cfg = AudioConfig::new(cfg.sample_rate, 1, cfg.buffer_size);
    let mut out = AudioOutputBuilder::new().device(device).config(audio_cfg).build().map_err(|e| anyhow!("AudioOutput error: {e:?}"))?;
    let (audio_tx, audio_rx) = mpsc::unbounded_channel::<Vec<f32>>();
    out.start(audio_rx)?;
    Ok((out, audio_tx))
}

fn message_from_rendered(rendered: &str) -> Result<Message> {
    let s = rendered.trim();
    if s.starts_with("CQ") {
        let toks: Vec<&str> = s.split_whitespace().collect();
        if toks.len() >= 3 && toks[0] == "CQ" && toks[1].eq_ignore_ascii_case("de") {
            return Message::cq(toks[2]).map_err(|e| anyhow!("{e}"));
        }
        return Message::cq(toks[toks.len()-1]).map_err(|e| anyhow!("{e}"));
    }
    let parts: Vec<&str> = s.split(" de ").collect();
    if parts.len() != 2 { return Err(anyhow!("Format error")); }
    let to = parts[0].trim();
    let rhs = parts[1].trim();
    let mut toks = rhs.split_whitespace();
    let from = toks.next().ok_or_else(|| anyhow!("Missing call"))?;
    match toks.next() {
        None => Message::cold_call(from, to).map_err(|e| anyhow!("{e}")),
        Some(tok) => Message::call_with_report(from, to, tok).map_err(|e| anyhow!("{e}"))
    }
}

fn vec_to_array<const N: usize>(v: Vec<i32>, _what: &str) -> Result<[i32; N]> {
    if v.len() < N { 
        let mut padded = v; 
        while padded.len() < N { padded.push(0); }
        let mut arr = [0i32; N];
        arr.copy_from_slice(&padded[..N]);
        return Ok(arr);
    }
    let mut arr = [0i32; N];
    arr.copy_from_slice(&v[..N]);
    Ok(arr)
}

fn generate_transmission(message: &Message, codec: &CallsignCodec, sample_rate: u32, slot_len_ms: u32) -> Result<Vec<f32>> {
    let samples_per_symbol = (sample_rate / SYMBOL_RATE) as usize;
    let packet258: [i32; 258] = match message.format {
        1 => {
            let (info_vec, _) = message.to_format1_bits(codec).map_err(|e| anyhow!("{e}"))?;
            let info71 = vec_to_array::<71>(info_vec, "Info")?;
            let addr49 = if let Some(ref to) = message.to_call {
                let v = codec.generate_private_address(to).map_err(anyhow::Error::msg)?;
                vec_to_array::<49>(v[..49.min(v.len())].to_vec(), "Addr")?
            } else { GENERAL_ADDRESS_49 };
            let (p1, p2) = fec::encode_format1(&info71);
            vec_to_array::<258>(fmt1::interleave_format1(&fmt1::SYNC_PATTERN_HD43, &addr49, &vec_to_array::<83>(p1, "P1")?, &vec_to_array::<83>(p2, "P2")?), "Pkt")?
        },
        2 => {
            let info18 = vec_to_array::<18>(message.to_format2_bits(codec).map_err(|e| anyhow!("{e}"))?, "I18")?;
            let to = message.to_call.as_ref().ok_or_else(|| anyhow!("No TO"))?;
            let addr49 = vec_to_array::<49>(codec.generate_private_address(to).map_err(anyhow::Error::msg)?[..49].to_vec(), "Addr")?;
            vec_to_array::<258>(fmt2::interleave_format2(&fmt1::SYNC_PATTERN_HD43, &addr49, &fec::encode_format2(&info18)), "Pkt")?
        },
        _ => return Err(anyhow!("Bad format")),
    };
    let syms: Vec<i32> = packet258.iter().map(|&b| if b == 0 { 1 } else { -1 }).collect();
    let mut continuous = Vec::new();
    let repeats = (slot_len_ms as f32 / 1000.0 / (258.0 / SYMBOL_RATE as f32)).ceil() as usize;
    for _ in 0..repeats { continuous.extend_from_slice(&syms); }
    let mut wf = generate_msk(&continuous, sample_rate, samples_per_symbol);
    wf.truncate((slot_len_ms as f32 / 1000.0 * sample_rate as f32) as usize);
    Ok(wf)
}

fn generate_msk(symbols: &[i32], sample_rate: u32, samples_per_symbol: usize) -> Vec<f32> {
    let mut samples = Vec::new();
    let mut phase = 0.0f32;
    for &symbol in symbols {
        let phase_step = 2.0 * PI * (CENTER_FREQ + (symbol as f32 * FREQ_DEV)) / sample_rate as f32;
        for _ in 0..samples_per_symbol {
            phase += phase_step;
            samples.push(0.5 * phase.sin());
            if phase > 2.0 * PI { phase -= 2.0 * PI; }
        }
    }
    samples
}

fn is_short_format2(s: &str) -> bool {
    let s = s.trim();
    s == "RR" || s == "RRR" || s == "73" || (s.starts_with('R') && s.len() == 3 && s[1..].parse::<u32>().is_ok())
}

fn message_from_short_format2(s: &str, my_call: &str, their_call: &str) -> Result<Message> {
    if my_call.is_empty() || their_call.is_empty() { return Err(anyhow!("No calls")); }
    match s.trim() {
        "RR" | "RRR" => Message::roger_roger(my_call, their_call).map_err(|e| anyhow!("{e}")),
        "73" => Message::seventy_three(my_call, their_call).map_err(|e| anyhow!("{e}")),
        _ => Message::format2(my_call, their_call, s).map_err(|e| anyhow!("{e}")),
    }
}

/// Resolve a display name like "USB Audio CODEC (TX)" to the correct cpal output device.
/// On Linux/ALSA, USB CODECs may only appear in input_devices() enumeration.
/// We search output → all → input devices, since ALSA device handles work for both directions.
fn find_nth_output_device(display_name: &str) -> Option<cpal::Device> {
    use cpal::traits::{DeviceTrait, HostTrait};
    let (base_name, suffix) = parse_device_suffix(display_name);
    let host = cpal::default_host();
    
    log::info!("[TX] Resolving output device: '{}' (base='{}', suffix='{}')", display_name, base_name, suffix);
    
    // 1. Try output_devices() — standard path, works on Windows/macOS
    if let Ok(devs) = host.output_devices() {
        for dev in devs {
            if let Ok(name) = dev.name() {
                if name == display_name || name == base_name {
                    log::info!("[TX] Found in output_devices(): '{}'", name);
                    return Some(dev);
                }
            }
        }
    }

    // 2. Try host.devices() — duplex devices (macOS CoreAudio path)
    if let Ok(devs) = host.devices() {
        let dev_list: Vec<cpal::Device> = devs.collect();
        match suffix.as_str() {
            "TX" | "RX/TX" => {
                let mut fallback: Option<cpal::Device> = None;
                for dev in dev_list {
                    if let Ok(name) = dev.name() {
                        if name == base_name {
                            let has_out = dev.supported_output_configs().map(|mut c| c.next().is_some()).unwrap_or(false)
                                || dev.default_output_config().is_ok();
                            if has_out { 
                                log::info!("[TX] Found TX device via capability detection: '{}'", name);
                                return Some(dev); 
                            }
                            let has_in = dev.supported_input_configs().map(|mut c| c.next().is_some()).unwrap_or(false)
                                || dev.default_input_config().is_ok();
                            if !has_in && fallback.is_none() { fallback = Some(dev); }
                        }
                    }
                }
                if let Some(fb) = fallback { 
                    log::info!("[TX] Using inferred TX device (not-input fallback)");
                    return Some(fb); 
                }
            }
            "" => {
                // No suffix — match by name in all devices
                for dev in dev_list {
                    if let Ok(name) = dev.name() {
                        if name == base_name || name == display_name {
                            log::info!("[TX] Found in host.devices(): '{}'", name);
                            return Some(dev);
                        }
                    }
                }
            }
            _ => {
                let occurrence: usize = suffix.parse().unwrap_or(1);
                let mut count = 0usize;
                for dev in dev_list {
                    if let Ok(name) = dev.name() {
                        if name == base_name {
                            count += 1;
                            if count == occurrence { return Some(dev); }
                        }
                    }
                }
            }
        }
    }

    // 3. 🟢 LINUX FIX: Check input_devices() too!
    // ALSA USB CODECs often only appear in input enumeration but the underlying
    // device handle works for both input and output (same hardware PCM).
    if let Ok(devs) = host.input_devices() {
        for dev in devs {
            if let Ok(name) = dev.name() {
                if name == display_name || name == base_name {
                    log::warn!("[TX] Found '{}' in INPUT list — using for OUTPUT (Linux ALSA workaround)", name);
                    return Some(dev);
                }
            }
        }
    }
    
    log::error!("[TX] CRITICAL: Device '{}' not found in ANY list. Falling back to System Default.", display_name);
    None
}

fn parse_device_suffix(display_name: &str) -> (String, String) {
    // Only match suffixes WE added: (RX), (TX), (RX/TX), or (N) where N is a number
    // Windows device names like "Speakers (2- USB Audio CODEC )" must NOT be split
    if let Some(pos) = display_name.rfind(" (") {
        if display_name.ends_with(')') {
            let suffix = display_name[pos+2..display_name.len()-1].trim();
            if suffix == "RX" || suffix == "TX" || suffix == "RX/TX" || suffix.chars().all(|c| c.is_ascii_digit()) {
                return (display_name[..pos].to_string(), suffix.to_string());
            }
        }
    }
    // No known suffix — the entire string is the device name
    (display_name.to_string(), String::new())
}
