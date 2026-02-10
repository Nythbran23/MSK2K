// src/modem/rx.rs

use log::{debug, info};
use tokio::sync::mpsc;
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};
use cpal::traits::DeviceTrait;

use msk2k_audio::{AudioConfig, AudioInputBuilder, DeviceManager};
use msk2k_dsp::message::Message;
use msk2k_dsp::rx::{PhaseDemodState, MatrixSyncExtractor, PacketCandidate, RxSync};
use msk2k_dsp::decode::{decode_packet_soft, is_general_addr};
use msk2k_dsp::callsign::CallsignCodec;
use crate::engine::accumulator::Accumulator;

#[derive(Debug, Clone)]
pub struct RxAudioCfg {
    pub input_device: Option<String>,
    pub sample_rate: u32,
    pub buffer_size: usize,
    pub decode_window_secs: f32,
    pub my_call: String,
    pub their_call: Option<String>,
    pub slot_len_ms: u32,
    pub my_tx_slot: u8,
    pub rx_slot_override: Option<u8>,
    pub listen_all_slots: bool,
}

#[derive(Debug, Clone)]
pub enum RxConfigUpdate {
    TheirCall(Option<String>),
    SlotTiming {
        my_tx_slot: u8,
        rx_slot_override: Option<u8>,
        listen_all_slots: bool,
        slot_len_ms: u32,
    },
    EndOfPeriod,
}

#[derive(Debug, Clone)]
pub struct RxDecoded {
    pub msg: Message,
    pub snr: Option<f32>,
    pub utc_ms: i64,
    pub rx_slot: u8,
    pub is_private: bool,
    pub is_accumulated: bool, 
}

pub async fn run_receiver(
    mut cfg: RxAudioCfg,
    decoded_tx: mpsc::UnboundedSender<RxDecoded>,
    mut stop_rx: mpsc::UnboundedReceiver<()>,
    mut config_rx: mpsc::UnboundedReceiver<RxConfigUpdate>,
) {
    let env_noise = env::var("MSK2K_NOISE").ok().and_then(|v| v.parse::<f32>().ok());
    let env_burst = env::var("MSK2K_BURST").ok().and_then(|v| v.parse::<u8>().ok());

    let mut sim_noise_floor = 0.0;
    let mut channel_sim = None;

    if env_noise.is_some() || env_burst.is_some() {
        sim_noise_floor = env_noise.unwrap_or(0.0);
        let burst_mode = env_burst.unwrap_or(0);
        channel_sim = Some(ChannelSimulator::new(12345, burst_mode));
    }

    let manager = DeviceManager::new().unwrap_or_else(|e| panic!("DeviceManager failed: {}", e));
    let device = if let Some(ref name) = cfg.input_device {
        manager.get_input_device(Some(name)).unwrap_or_else(|_| manager.default_input_device().unwrap())
    } else {
        manager.default_input_device().unwrap()
    };
    log::info!("[RX] Opening Device: {}", device.name().unwrap_or_default());

    let audio_cfg = AudioConfig::new(cfg.sample_rate, 1, cfg.buffer_size);
    let mut audio_input = match AudioInputBuilder::new().device(device).config(audio_cfg).build() {
        Ok(ai) => ai,
        Err(e) => { log::error!("[RX] Failed to build AudioInput: {:?}", e); return; }
    };
    let (audio_chunk_tx, mut audio_chunk_rx) = mpsc::unbounded_channel::<Vec<f32>>();
    if let Err(e) = audio_input.start(audio_chunk_tx) {
        log::error!("[RX] Failed to start audio capture: {:?}", e);
        return;
    }

    let mut demod = PhaseDemodState::new();
    let mut extractor = MatrixSyncExtractor::new();
    let mut accumulator = Accumulator::new(&cfg.my_call, cfg.their_call.clone());
    let mut max_retain_samples = (cfg.sample_rate as u64 * (cfg.slot_len_ms as u64 + 1000) / 1000) as usize;
    let mut retained_audio: Vec<f32> = Vec::with_capacity(max_retain_samples);
    let mut pending_accumulation = false;
    
    // 🔍 DIAGNOSTIC: Audio level monitoring
    let mut diag_sample_count: u64 = 0;
    let mut diag_sum_sq: f64 = 0.0;
    let mut diag_peak: f32 = 0.0;
    let mut diag_chunk_count: u64 = 0;
    let mut diag_candidate_count: u64 = 0;
    let mut diag_decode_count: u64 = 0;

    loop {
        tokio::select! {
            _ = stop_rx.recv() => { break; }
            Some(update) = config_rx.recv() => {
                match update {
                    RxConfigUpdate::TheirCall(tc) => {
                        cfg.their_call = tc.clone();
                        accumulator.update_config(&cfg.my_call, tc); 
                    },
                    RxConfigUpdate::SlotTiming { my_tx_slot, rx_slot_override, listen_all_slots, slot_len_ms } => {
                        cfg.my_tx_slot = my_tx_slot;
                        cfg.rx_slot_override = rx_slot_override;
                        cfg.listen_all_slots = listen_all_slots;
                        cfg.slot_len_ms = slot_len_ms;
                        max_retain_samples = (cfg.sample_rate as u64 * (cfg.slot_len_ms as u64 + 1000) / 1000) as usize;
                        if retained_audio.capacity() < max_retain_samples {
                            retained_audio.reserve(max_retain_samples - retained_audio.len());
                        }
                    },
                    RxConfigUpdate::EndOfPeriod => {
                        pending_accumulation = true;
                        while let Ok(mut chunk) = audio_chunk_rx.try_recv() {
                            for s in &mut chunk { *s *= 0.5; }
                            retained_audio.extend_from_slice(&chunk);
                            let phases = demod.push_audio(&chunk);
                            if !phases.is_empty() {
                                let candidates = extractor.push_phase(&phases);
                                for candidate in candidates { 
                                    accumulator.add(candidate);
                                }
                            }
                        }
                        demod = PhaseDemodState::new();
                        extractor = MatrixSyncExtractor::new();
                    },
                }
            }

            maybe_audio = audio_chunk_rx.recv() => {
                match maybe_audio {
                    Some(mut chunk) => {
                        let now = SystemTime::now();
                        let utc_ms = now.duration_since(UNIX_EPOCH).unwrap().as_millis() as i64;
                        let period_ms = cfg.slot_len_ms as u64;
                        let current_rx_slot = ((utc_ms as u64 / period_ms) % 2) as u8;

                        // 🔍 DIAGNOSTIC: Measure audio levels BEFORE any processing
                        for s in &chunk {
                            diag_sum_sq += (*s as f64) * (*s as f64);
                            let abs = s.abs();
                            if abs > diag_peak { diag_peak = abs; }
                        }
                        diag_sample_count += chunk.len() as u64;
                        diag_chunk_count += 1;
                        
                        // Log every ~5 seconds (approx 240 chunks at 48kHz/1024)
                        if diag_chunk_count % 240 == 0 {
                            let rms = if diag_sample_count > 0 {
                                (diag_sum_sq / diag_sample_count as f64).sqrt()
                            } else { 0.0 };
                            log::info!("🔊 RX AUDIO: rms={:.6}, peak={:.4}, samples={}, chunks={}, candidates={}, decodes={}",
                                rms, diag_peak, diag_sample_count, diag_chunk_count,
                                diag_candidate_count, diag_decode_count);
                            // Reset for next window
                            diag_sample_count = 0;
                            diag_sum_sq = 0.0;
                            diag_peak = 0.0;
                            diag_candidate_count = 0;
                            diag_decode_count = 0;
                        }

                        if let Some(sim) = &mut channel_sim {
                            for s in &mut chunk {
                                let gain = sim.next_gain();
                                *s = (*s * gain) + (sim.rng.cheap_noise() * sim_noise_floor);
                            }
                        }
                        for s in &mut chunk { *s *= 0.5; }

                        let phases = demod.push_audio(&chunk);
                        if !phases.is_empty() {
                            let candidates = extractor.push_phase(&phases);
                            for candidate in &candidates {
                                diag_candidate_count += 1; // 🔍 DIAGNOSTIC
                                // 🟢 RESTORED: Only add to accumulator if Live Decode FAILS
                                if let Some(decoded) = decode_candidate(candidate, &cfg, utc_ms, current_rx_slot) {
                                    diag_decode_count += 1; // 🔍 DIAGNOSTIC
                                    log::info!("[LIVE]  ✅ '{}'", decoded.msg.text);
                                    let _ = decoded_tx.send(decoded);
                                } else {
                                    accumulator.add(candidate.clone());
                                }
                            }
                        }

                        if pending_accumulation {
                            // 🟢 LOG: Debug level (hidden by default) to stop "192" spam
                            if accumulator.candidate_count() > 0 {
                                debug!("[ACCUM] Processing {} candidates...", accumulator.candidate_count());
                            }

                            if let Some(msg) = accumulator.process() {
                                let is_private = msg.format == 2 || (msg.format == 1 && msg.to_call.as_deref().unwrap_or("CQ") != "CQ");
                                let decoded = RxDecoded { msg, snr: None, utc_ms, rx_slot: current_rx_slot, is_private, is_accumulated: true };
                                let _ = decoded_tx.send(decoded);
                            }
                            
                            // 🟢 CRITICAL: Reset accumulator
                            accumulator.clear();
                            retained_audio.clear();
                            pending_accumulation = false;
                        }
                        
                        if retained_audio.len() > max_retain_samples { retained_audio.clear(); }
                    }
                    None => break,
                }
            }
        }
    }
    audio_input.stop();
}

fn decode_candidate(candidate: &PacketCandidate, cfg: &RxAudioCfg, utc_ms: i64, rx_slot: u8) -> Option<RxDecoded> {
    let sync = &candidate.sync;
    if sync.sync_bits < 35 && sync.correlation < 0.40 { return None; }

    let codec = CallsignCodec::new();
    let shifts = [(0, 1), (14, 2), (29, 2)];

    for &(shift, fmt) in &shifts {
        let ts = RxSync {
            found: true, correlation: sync.correlation, position: sync.position,
            sync_bits: sync.sync_bits, polarity: sync.polarity,
            sync_shift: shift, sync_rotation: sync.sync_rotation, format_hint: fmt,
        };

        if let Some(pkt) = decode_packet_soft(&candidate.packet_soft, &ts) {
            let mut msg_res = None;

            // 🟢 FORMAT 1: Try both Grid and Standard decode paths
            if pkt.format == 1 {
                let is_general = is_general_addr(&pkt.addr_bits);
                // Type field at bit positions 54,55 (0-indexed)
                // [1,1] = Type 11 (CQ+Grid), [0,1] = Type 01 (Standard CQ/directed)
                let b54 = pkt.info_bits.get(54).unwrap_or(&0);
                let b55 = pkt.info_bits.get(55).unwrap_or(&0);
                
                // 🟢 TRY 1: If type bits suggest Grid CQ, try grid decode first
                if is_general && *b54 == 1 && *b55 == 1 {
                    if let Ok(grid_text) = codec.decode_cq_with_grid(&pkt.info_bits) {
                        let parts: Vec<&str> = grid_text.split_whitespace().collect();
                        let clean_call = parts.get(0).unwrap_or(&"?").to_string();
                        let grid = parts.get(1).unwrap_or(&"");
                        let ui_text = format!("CQ {} {}", clean_call, grid);
                        
                        msg_res = Some(Ok(Message {
                            from_call: clean_call,
                            to_call: None,
                            text: ui_text.clone(),
                            content: msk2k_dsp::message::MessageContent::Format1 { text: ui_text },
                            format: 1,
                        }));
                    }
                }
                
                // 🟢 TRY 2: If grid decode didn't produce a result, try standard decode
                // This handles: standard CQ, directed calls, AND grid CQ with decode errors
                if msg_res.is_none() {
                    msg_res = Some(Message::from_format1_bits(&codec, &pkt.info_bits, &pkt.addr_bits, is_general, Some(&cfg.my_call)));
                }
            } 
            // 🟢 FORMAT 2: Short messages
            else if pkt.format == 2 {
                msg_res = Some(Message::from_format2_bits(&codec, &pkt.info_bits, &pkt.addr_bits, &cfg.my_call, cfg.their_call.as_deref().unwrap_or("")));
            }

            if let Some(Ok(msg)) = msg_res {
                if msg.from_call == cfg.my_call { continue; }
                let is_private_msg = pkt.format == 2 || (pkt.format == 1 && msg.to_call.as_deref().unwrap_or("CQ") != "CQ");
                return Some(RxDecoded { 
                    msg, snr: Some(sync.correlation), utc_ms, rx_slot, 
                    is_private: is_private_msg, is_accumulated: false 
                });
            }
        }
    }
    None
}

// ─── UTILS ───
struct SimpleRng { state: u64 }
impl SimpleRng {
    fn new(seed: u64) -> Self { Self { state: if seed == 0 { 1 } else { seed } } }
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12; self.state ^= self.state << 25; self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn cheap_noise(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / 16777216.0; 
        (u - 0.5) * 2.0 
    }
}
struct ChannelSimulator {
    rng: SimpleRng, envelope: f32, samples_until_ping: usize, decay_rate: f32, burst_mode: u8, logged_bypass: bool,
}
impl ChannelSimulator {
    fn new(seed: u64, burst_mode: u8) -> Self {
        let rng = SimpleRng::new(seed);
        let decay = match burst_mode { 0 => 1.0, 1 => 0.9995, _ => 0.99992 };
        let start_env = if burst_mode == 0 { 1.0 } else { 0.0 };
        Self { rng, envelope: start_env, samples_until_ping: 48000 * 2, decay_rate: decay, burst_mode, logged_bypass: false }
    }
    fn next_gain(&mut self) -> f32 {
        if self.burst_mode == 0 { if !self.logged_bypass { log::warn!("[SIM] Burst Logic: BYPASSED"); self.logged_bypass = true; } return 1.0; }
        self.envelope *= self.decay_rate;
        if self.samples_until_ping == 0 { self.envelope = 1.0; self.samples_until_ping = ((self.rng.next_u64() % 120000) + 24000) as usize; } 
        else { self.samples_until_ping -= 1; }
        if self.envelope < 0.05 { 0.0 } else { self.envelope }
    }
}
