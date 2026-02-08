// src/modem/tx.rs
//
// TX worker: UI/runtime sends TxRequest::Text { rendered, slot_len_ms }.
// We convert rendered string -> msk2k_dsp::message::Message,
// then generate real MSK2K waveform using the proven pipeline
// (fec/fmt1/fmt2 + MSK modulator) and push to audio output.

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
const CENTER_FREQ: f32 = 1500.0;
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
    my_call: String,
    their_call: String,
}

impl Default for TxAudioCfg {
    fn default() -> Self {
        Self {
            output_device: None,
            sample_rate: 48_000,
            buffer_size: 1024,
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
                output_level: _, // Ignored
                sample_rate,
                buffer_size,
                my_call,
                their_call,
            } => {
                cfg.output_device = output_device;
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

                // Generate waveform
                let waveform = match generate_transmission(&msg, &codec, cfg.sample_rate, slot_len_ms) {
                    Ok(w) => w,
                    Err(e) => {
                        warn!("[TX] Failed to generate waveform: {} - SKIPPING TX", e);
                        continue;
                    }
                };

                // FIXED OUTPUT LEVEL (Standard Line Level ~ -3dB)
                let output_scalar = 0.707;
                let scaled: Vec<f32> = waveform.iter().map(|&s| s * output_scalar).collect();

                if let Err(e) = audio_tx.send(scaled.clone()) {
                    warn!("[TX] Failed to send waveform to audio driver: {}", e);
                } else {
                    let duration_secs = scaled.len() as f32 / cfg.sample_rate as f32;
                    info!("[TX] 📤 Sent {} samples ({:.2}s)", scaled.len(), duration_secs);

                    drop(audio_tx);
                    
                    // Wait for playback to finish
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

fn build_output_and_sender(cfg: &TxAudioCfg) -> Result<(AudioOutput, mpsc::UnboundedSender<Vec<f32>>)> {
    let manager = DeviceManager::new()?;

    let device = if let Some(ref name) = cfg.output_device {
        info!("[TX] Opening selected output device: {}", name);
        manager.get_output_device(Some(name)).or_else(|_| {
            warn!("[TX] Selected device '{}' not found, falling back to default.", name);
            manager.default_output_device()
        })?
    } else {
        manager.default_output_device()?
    };

    let audio_cfg = AudioConfig::new(cfg.sample_rate, 1, cfg.buffer_size);

    let mut out = AudioOutputBuilder::new()
        .device(device)
        .config(audio_cfg)
        .build()
        .map_err(|e| anyhow!("Failed to build AudioOutput: {e:?}"))?;

    let (audio_tx, audio_rx) = mpsc::unbounded_channel::<Vec<f32>>();
    out.start(audio_rx)?;

    Ok((out, audio_tx))
}

fn message_from_rendered(rendered: &str) -> Result<Message> {
    let s = rendered.trim();

    if s.starts_with("CQ") {
        let toks: Vec<&str> = s.split_whitespace().collect();
        if toks.len() >= 3 && toks[0] == "CQ" && toks[1].eq_ignore_ascii_case("de") {
            return Message::cq(toks[2]).map_err(|e| anyhow!("Message::cq failed: {e}"));
        }
        if toks.len() >= 2 && toks[0] == "CQ" {
            return Message::cq(toks[1]).map_err(|e| anyhow!("Message::cq failed: {e}"));
        }
        return Err(anyhow!("CQ missing callsign: '{}'", s));
    }

    let toks: Vec<&str> = s.split_whitespace().collect();
    
    if toks.len() == 3 {
        let token = toks[0];
        let from = toks[1];
        let to = toks[2];
        
        if token == "RR" {
            return Message::roger_roger(from, to).map_err(|e| anyhow!("Message::roger_roger failed: {e}"));
        }
        if token == "73" {
            return Message::seventy_three(from, to).map_err(|e| anyhow!("Message::seventy_three failed: {e}"));
        }
        if token.starts_with('R') && token.len() == 3 {
            return Message::call_with_report(from, to, token).map_err(|e| anyhow!("Message::call_with_report failed: {e}"));
        }
    }

    let parts: Vec<&str> = s.split(" de ").collect();
    if parts.len() != 2 {
        return Err(anyhow!("Unrecognized TX string format: '{}'", s));
    }

    let to = parts[0].trim();
    let rhs = parts[1].trim();
    let mut toks = rhs.split_whitespace();

    let from = toks.next().ok_or_else(|| anyhow!("Missing FROM callsign"))?;
    let tail = toks.next();

    match tail {
        None => Message::cold_call(from, to).map_err(|e| anyhow!("Message::cold_call failed: {e}")),
        Some("RR") => Message::roger_roger(from, to).map_err(|e| anyhow!("Message::roger_roger failed: {e}")),
        Some("73") => Message::seventy_three(from, to).map_err(|e| anyhow!("Message::seventy_three failed: {e}")),
        Some(tok) => Message::call_with_report(from, to, tok).map_err(|e| anyhow!("Message::call_with_report failed: {e}"))
    }
}

fn vec_to_array<const N: usize>(v: Vec<i32>, what: &str) -> Result<[i32; N]> {
    if v.len() != N {
        return Err(anyhow!("{} length mismatch: expected {} bits, got {}", what, N, v.len()));
    }
    let mut arr = [0i32; N];
    arr.copy_from_slice(&v[..N]);
    Ok(arr)
}

fn generate_transmission(
    message: &Message,
    codec: &CallsignCodec,
    sample_rate: u32,
    slot_len_ms: u32,
) -> Result<Vec<f32>> {
    if sample_rate % SYMBOL_RATE != 0 {
        return Err(anyhow!(
            "sample_rate {} must be divisible by symbol rate {}",
            sample_rate,
            SYMBOL_RATE
        ));
    }
    let samples_per_symbol = (sample_rate / SYMBOL_RATE) as usize;

    let packet258: [i32; 258] = match message.format {
        1 => {
            let (info_vec, _) = message.to_format1_bits(codec).map_err(|e| anyhow!("Fmt1 encode: {}", e))?;
            let info71 = vec_to_array::<71>(info_vec, "Info")?;
            
            let addr49 = if let Some(ref to) = message.to_call {
                let v = codec.generate_private_address(to).map_err(anyhow::Error::msg)?;
                let v49 = v[..49.min(v.len())].to_vec();
                vec_to_array::<49>(v49, "Addr")?
            } else {
                GENERAL_ADDRESS_49
            };

            let (p1, p2) = fec::encode_format1(&info71);
            let poly1 = vec_to_array::<83>(p1, "Poly1")?;
            let poly2 = vec_to_array::<83>(p2, "Poly2")?;
            let sync = fmt1::SYNC_PATTERN_HD43;
            
            vec_to_array::<258>(fmt1::interleave_format1(&sync, &addr49, &poly1, &poly2), "Packet")?
        },
        2 => {
            let info18_vec = message.to_format2_bits(codec).map_err(|e| anyhow!("Fmt2 encode: {}", e))?;
            let info18 = vec_to_array::<18>(info18_vec, "Info18")?;
            
            let to = message.to_call.as_ref().ok_or_else(|| anyhow!("Format-2 requires to_call"))?;
            let addr_full = codec.generate_private_address(to).map_err(anyhow::Error::msg)?;
            let addr49 = vec_to_array::<49>(addr_full[..49].to_vec(), "Addr")?;
            
            let poly_dict = fec::encode_format2(&info18);
            let sync = fmt1::SYNC_PATTERN_HD43;
            
            vec_to_array::<258>(fmt2::interleave_format2(&sync, &addr49, &poly_dict), "Packet")?
        },
        _ => return Err(anyhow!("Unsupported format")),
    };

    let single_packet_symbols: Vec<i32> = packet258.iter().map(|&b| if b == 0 { 1 } else { -1 }).collect();

    let packet_duration_s = 258.0 / SYMBOL_RATE as f32;
    let target_duration_s = slot_len_ms as f32 / 1000.0;
    let repeats = (target_duration_s / packet_duration_s).ceil() as usize;

    let mut continuous_symbols = Vec::with_capacity(single_packet_symbols.len() * repeats);
    for _ in 0..repeats {
        continuous_symbols.extend_from_slice(&single_packet_symbols);
    }

    let mut full_waveform = generate_msk(&continuous_symbols, sample_rate, samples_per_symbol);

    let target_samples = (target_duration_s * sample_rate as f32) as usize;
    if full_waveform.len() > target_samples {
        full_waveform.truncate(target_samples);
    }

    Ok(full_waveform)
}

fn generate_msk(symbols: &[i32], sample_rate: u32, samples_per_symbol: usize) -> Vec<f32> {
    let mut samples = Vec::with_capacity(symbols.len() * samples_per_symbol);
    let mut phase = 0.0f32;

    for &symbol in symbols {
        let freq = CENTER_FREQ + (symbol as f32 * FREQ_DEV);
        let phase_step = 2.0 * PI * freq / sample_rate as f32;

        for _ in 0..samples_per_symbol {
            phase += phase_step;
            samples.push(0.5 * phase.sin());
            if phase > 2.0 * PI {
                phase -= 2.0 * PI;
            }
        }
    }

    samples
}

fn is_short_format2(s: &str) -> bool {
    let s = s.trim();
    if s == "RR" || s == "RRR" || s == "73" {
        return true;
    }
    if s.starts_with('R') && s.len() == 3 {
        if s[1..].parse::<u32>().is_ok() {
            return true;
        }
    }
    false
}

fn message_from_short_format2(s: &str, my_call: &str, their_call: &str) -> Result<Message> {
    let s = s.trim();
    if my_call.is_empty() || their_call.is_empty() {
        return Err(anyhow!("Short Format-2 message needs calls"));
    }
    if s == "RR" || s == "RRR" {
        return Message::roger_roger(my_call, their_call).map_err(|e| anyhow!("RR fail: {e}"));
    }
    if s == "73" {
        return Message::seventy_three(my_call, their_call).map_err(|e| anyhow!("73 fail: {e}"));
    }
    if s.starts_with('R') && s.len() == 3 {
        return Message::format2(my_call, their_call, s).map_err(|e| anyhow!("Fmt2 fail: {e}"));
    }
    Err(anyhow!("Unknown short fmt2: {s}"))
}