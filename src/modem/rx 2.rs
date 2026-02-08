// src/modem/rx.rs
//
// RX pipeline using DJ5HG matrix bit-sync architecture:
//
//   AudioInput  ->  PhaseDemodState  ->  MatrixSyncExtractor  ->  decode
//   (cpal)         (IQ demod,           (try all 24 sub-bit     (Viterbi FEC,
//                   phase unwrap,        offsets, HD43 sync,      parity check,
//                   sample-rate out)     best timing wins)        source decode)
//
// Per-period workflow:
//   1. FAST PASS  - real-time streaming decode as audio arrives
//   2. ACCUMULATE - at end of RX period, re-process full 15s for weak signals
//   3. DISCARD    - audio from that period is thrown away forever
//
// The RX task runs continuously until explicitly stopped. Config changes
// arrive via an update channel - no restart needed.

use tokio::sync::mpsc;

use msk2k_audio::{AudioConfig, AudioInputBuilder, DeviceManager};
use msk2k_dsp::message::Message;
use msk2k_dsp::rx::{PhaseDemodState, MatrixSyncExtractor, PacketCandidate};

#[derive(Debug, Clone)]
pub struct RxAudioCfg {
    pub input_device: Option<String>,
    pub sample_rate: u32,
    pub buffer_size: usize,
    pub input_gain: f32,
    pub decode_window_secs: f32,

    pub my_call: String,
    pub their_call: Option<String>,
    pub slot_len_ms: u32,

    pub my_tx_slot: u8,
    pub rx_slot_override: Option<u8>,
    pub listen_all_slots: bool,
}

/// Live config updates sent to a running RX task without restarting it.
#[derive(Debug, Clone)]
pub enum RxConfigUpdate {
    /// Update the target callsign for Format-2 decode
    TheirCall(Option<String>),
    /// Update slot timing parameters
    SlotTiming {
        my_tx_slot: u8,
        rx_slot_override: Option<u8>,
        listen_all_slots: bool,
        slot_len_ms: u32,
    },
    /// TX period starting — run accumulation on retained audio, then discard.
    EndOfPeriod,
}

#[derive(Debug, Clone)]
pub struct RxDecoded {
    pub msg: Message,
    pub snr: Option<f32>,
    pub utc_ms: i64,
    pub rx_slot: u8,
}

/// Main RX entry point.
///
/// Runs continuously until `stop_rx` fires. Config changes arrive via
/// `config_rx` without needing to restart (preserves DSP state).
pub async fn run_receiver(
    mut cfg: RxAudioCfg,
    decoded_tx: mpsc::UnboundedSender<RxDecoded>,
    mut stop_rx: mpsc::UnboundedReceiver<()>,
    mut config_rx: mpsc::UnboundedReceiver<RxConfigUpdate>,
) {
    // -- 1. Open audio input device --
    let manager = match DeviceManager::new() {
        Ok(m) => m,
        Err(e) => {
            log::error!("[RX] Failed to create DeviceManager: {}", e);
            return;
        }
    };

    let device = if let Some(ref name) = cfg.input_device {
        log::info!("[RX] Opening input device: {}", name);
        match manager.get_input_device(Some(name)) {
            Ok(d) => d,
            Err(e) => {
                log::warn!("[RX] Device '{}' not found ({}), trying default", name, e);
                match manager.default_input_device() {
                    Ok(d) => d,
                    Err(e2) => {
                        log::error!("[RX] No input device: {}", e2);
                        return;
                    }
                }
            }
        }
    } else {
        match manager.default_input_device() {
            Ok(d) => d,
            Err(e) => {
                log::error!("[RX] No default input device: {}", e);
                return;
            }
        }
    };

    let audio_cfg = AudioConfig::new(cfg.sample_rate, 1, cfg.buffer_size);
    let mut audio_input = match AudioInputBuilder::new()
        .device(device)
        .config(audio_cfg)
        .build()
    {
        Ok(ai) => ai,
        Err(e) => {
            log::error!("[RX] Failed to build AudioInput: {:?}", e);
            return;
        }
    };

    let (audio_chunk_tx, mut audio_chunk_rx) = mpsc::unbounded_channel::<Vec<f32>>();
    if let Err(e) = audio_input.start(audio_chunk_tx) {
        log::error!("[RX] Failed to start audio capture: {:?}", e);
        return;
    }

    log::info!(
        "[RX] Audio capture started: sr={} buf={} dev={:?}",
        cfg.sample_rate, cfg.buffer_size, cfg.input_device
    );

    // -- 2. DSP pipeline --
    let mut demod = PhaseDemodState::new();
    let mut extractor = MatrixSyncExtractor::new();

    let mut total_samples: u64 = 0;
    let mut total_candidates: u64 = 0;
    let mut total_decodes: u64 = 0;

    // Audio retention buffer for accumulation pass.
    // Collects raw audio for the current RX period (up to ~16s).
    // Discarded completely after accumulation pass runs.
    let max_retain_samples = (cfg.sample_rate as usize) * 16; // 16s headroom
    let mut retained_audio: Vec<f32> = Vec::with_capacity(max_retain_samples);

    // Deferred accumulation: set when EndOfPeriod arrives, fires after
    // the last audio chunk from the period has been fast-decoded.
    let mut pending_accumulation = false;

    // -- 3. Main loop --
    loop {
        tokio::select! {
            _ = stop_rx.recv() => {
                log::info!("[RX] stop requested");
                break;
            }

            // -- Config updates (no restart needed) --
            Some(update) = config_rx.recv() => {
                match update {
                    RxConfigUpdate::TheirCall(tc) => {
                        log::info!("[RX] Config update: their_call={:?}", tc);
                        cfg.their_call = tc;
                    }
                    RxConfigUpdate::SlotTiming { my_tx_slot, rx_slot_override, listen_all_slots, slot_len_ms } => {
                        log::info!("[RX] Config update: tx_slot={} rx_override={:?} listen_all={}", my_tx_slot, rx_slot_override, listen_all_slots);
                        cfg.my_tx_slot = my_tx_slot;
                        cfg.rx_slot_override = rx_slot_override;
                        cfg.listen_all_slots = listen_all_slots;
                        cfg.slot_len_ms = slot_len_ms;
                    }
                    RxConfigUpdate::EndOfPeriod => {
                        pending_accumulation = true;
                    }
                }
            }

            // -- Audio processing --
            maybe_audio = audio_chunk_rx.recv() => {
                match maybe_audio {
                    Some(mut chunk) => {
                        if cfg.input_gain != 1.0 {
                            for s in &mut chunk {
                                *s *= cfg.input_gain;
                            }
                        }
                        total_samples += chunk.len() as u64;

                        if pending_accumulation {
                            // End of RX period. Run accumulation on what we have,
                            // then clear buffer for next period. Continue to
                            // normal DSP below — don't suppress anything.
                            run_accumulation_pass(&retained_audio, &cfg, &decoded_tx);
                            retained_audio.clear();
                            pending_accumulation = false;
                        }

                        // Normal processing: retain audio and run fast decode
                        if retained_audio.len() + chunk.len() <= max_retain_samples {
                            retained_audio.extend_from_slice(&chunk);
                        }

                        let phases = demod.push_audio(&chunk);
                        if phases.is_empty() {
                            continue;
                        }

                        let candidates = extractor.push_phase(&phases);
                        if candidates.is_empty() {
                            continue;
                        }

                        // Fast decode: try ALL candidates, both F1 and F2.
                        let mut sorted: Vec<&PacketCandidate> = candidates.iter().collect();
                        sorted.sort_by(|a, b| {
                            b.sync.correlation.partial_cmp(&a.sync.correlation)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });

                        for candidate in &sorted {
                            total_candidates += 1;
                            if let Some(decoded) = decode_candidate(candidate, &cfg) {
                                total_decodes += 1;
                                log::info!(
                                    "\u{2705} Decoded: '{}' from {} (corr={:.3})",
                                    decoded.msg.text,
                                    decoded.msg.from_call,
                                    candidate.sync.correlation,
                                );
                                let _ = decoded_tx.send(decoded);
                            }
                        }
                    }
                    None => {
                        log::info!("[RX] audio channel closed");
                        break;
                    }
                }
            }
        }
    }

    audio_input.stop();
    log::info!(
        "[RX] stopped: {} samples ({:.1}s), {} candidates, {} decodes",
        total_samples,
        total_samples as f64 / cfg.sample_rate as f64,
        total_candidates,
        total_decodes,
    );
}

// -- Accumulation pass --
//
// Re-processes the full RX period audio through a fresh demod+extractor
// pipeline to catch weak signals the fast streaming pass may have missed.
// In future this will implement proper soft-bit accumulation across
// repeated packets within the window to improve SNR.
//
// Any new decodes are sent via decoded_tx just like fast-pass decodes.
// After this returns, the caller discards the audio — it's done forever.

fn run_accumulation_pass(
    audio: &[f32],
    cfg: &RxAudioCfg,
    decoded_tx: &mpsc::UnboundedSender<RxDecoded>,
) {
    let secs = audio.len() as f64 / cfg.sample_rate as f64;
    if audio.len() < cfg.sample_rate as usize {
        log::info!("[RX] Accumulation: only {:.1}s of audio, skipping", secs);
        return;
    }

    // ── 1. Fresh demod over full period, fed in chunks ──
    // Must feed in small chunks to trigger multiple evaluation strides
    // in the extractor (it only evaluates once per push_phase call).
    let mut demod = PhaseDemodState::new();
    let mut extractor = MatrixSyncExtractor::new();
    extractor.corr_threshold = 0.15; // Lower threshold for fragment detection

    let mut candidates: Vec<PacketCandidate> = Vec::new();
    let chunk_size = 1024usize; // Same as real-time audio input
    for chunk in audio.chunks(chunk_size) {
        let phases = demod.push_audio(chunk);
        if !phases.is_empty() {
            let cands = extractor.push_phase(&phases);
            candidates.extend(cands);
        }
    }

    if candidates.is_empty() {
        log::info!("[RX] Accumulation: {:.1}s, {} phases, 0 candidates", secs, phases.len());
        return;
    }

    log::info!(
        "[RX] Accumulation: {:.1}s, {} candidates found (threshold=0.15)",
        secs,
        candidates.len(),
    );

    // ── 3. Phase clustering ──
    // Each candidate's packet_soft came from a specific sample position.
    // Candidates from the same repeating packet share the same phase (position mod packet_period).
    // The packet period in samples = 258 bits × 24 samples/bit = 6192 samples.
    // But we work in terms of the extractor's end_index (in bit units).
    // Phase = end_index mod 258 (packet length in bits).
    use msk2k_dsp::accumulator::{Accumulator, PhaseClustering};

    let mut clustering = PhaseClustering::new();
    for (i, cand) in candidates.iter().enumerate() {
        let phase = (cand.end_index % 258) as i32;
        let weight = cand.sync.correlation * cand.sync.correlation;
        clustering.add_candidate(phase, i, weight);
    }

    let (dominant_phase, dominant_indices, dominance_ratio) = match clustering.get_dominant_bin() {
        Some(result) => result,
        None => {
            log::info!("[RX] Accumulation: no dominant phase bin");
            return;
        }
    };

    log::info!(
        "[RX] Accumulation: phase={}, {} fragments in dominant bin (ratio={:.2}), {} total",
        dominant_phase,
        dominant_indices.len(),
        dominance_ratio,
        candidates.len(),
    );

    // ── 4. Accumulate soft bits from dominant cluster ──
    let mut acc = Accumulator::new();

    let mut corr_sum = 0.0f32;
    for &idx in &dominant_indices {
        let cand = &candidates[idx];
        let packet_soft = &cand.packet_soft;
        let weight = cand.sync.correlation * cand.sync.correlation;

        // Valid mask: bits where |soft| is above noise floor
        // Use median-based threshold like Python
        let mut abs_vals: Vec<f32> = packet_soft.iter().map(|v| v.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = abs_vals[abs_vals.len() / 2];
        let threshold = (median * 0.7).max(0.05);

        let valid_mask: Vec<bool> = packet_soft.iter().map(|v| v.abs() >= threshold).collect();
        let valid_count: usize = valid_mask.iter().filter(|&&v| v).count();

        // Skip fragments with too few valid bits (pure noise)
        if valid_count < 40 {
            continue;
        }

        // Confidence: |soft_bit| as per-bit confidence (for MSK, this is reasonable
        // since the phase demod produces magnitude-proportional soft values)
        let conf: Vec<f32> = packet_soft.iter().map(|v| v.abs()).collect();

        acc.accumulate_soft_packet(packet_soft, weight, Some(&valid_mask), Some(&conf));
        corr_sum += cand.sync.correlation;
    }

    if acc.num_pings() == 0 {
        log::info!("[RX] Accumulation: no valid fragments after filtering");
        return;
    }

    let avg_corr = corr_sum / acc.num_pings() as f32;
    log::info!(
        "[RX] Accumulation: {} fragments accumulated, avg_corr={:.3}",
        acc.num_pings(),
        avg_corr,
    );

    // ── 5. Decode accumulated packet ──
    let averaged_soft = acc.get_averaged_soft();

    // Build a synthetic RxSync for the accumulated packet
    // Try all format/shift combos
    use msk2k_dsp::decode::decode_packet_soft;
    use msk2k_dsp::decode::is_general_addr;
    use msk2k_dsp::callsign::CallsignCodec;
    use msk2k_dsp::rx::RxSync;

    let shifts_to_try: &[(i32, i32)] = &[
        (0, 1),    // Format-1
        (14, 2),   // Format-2 shift=14
        (29, 2),   // Format-2 shift=29
    ];

    let codec = CallsignCodec::new();

    for &(shift, fmt_hint) in shifts_to_try {
        let try_sync = RxSync {
            found: true,
            correlation: avg_corr,
            position: 0,
            sync_bits: 43,  // Synthetic — accumulated
            polarity: 1,     // Already polarity-corrected
            sync_shift: shift,
            sync_rotation: 0,
            format_hint: fmt_hint,
        };

        if let Some(pkt) = decode_packet_soft(&averaged_soft, &try_sync) {
            let is_general = is_general_addr(&pkt.addr_bits);

            let msg_result = if pkt.format == 1 {
                Message::from_format1_bits(
                    &codec, &pkt.info_bits, &pkt.addr_bits, is_general,
                    Some(&cfg.my_call),
                )
            } else if pkt.format == 2 {
                if cfg.my_call.is_empty() {
                    continue;
                }
                Message::from_format2_bits(
                    &codec, &pkt.info_bits, &pkt.addr_bits,
                    &cfg.my_call,
                    cfg.their_call.as_deref().unwrap_or(""),
                )
            } else {
                continue;
            };

            if let Ok(msg) = msg_result {
                // Loopback protection: reject our own transmissions
                if msg.from_call == cfg.my_call {
                    continue;
                }

                log::info!(
                    "🔧 Accumulation decode: '{}' from {} ({} pings, avg_corr={:.3})",
                    msg.text,
                    msg.from_call,
                    acc.num_pings(),
                    avg_corr,
                );

                let decoded = RxDecoded {
                    msg,
                    snr: None,
                    timestamp_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0),
                };
                let _ = decoded_tx.send(decoded);
                return; // One successful decode per accumulation pass
            }
        }
    }

    log::info!(
        "[RX] Accumulation: {} fragments, decode failed (no parity match)",
        acc.num_pings(),
    );
}

// -- Decode a PacketCandidate --
//
// Per DJ5HG Section 8.2-8.3, candidates are pre-vetted BEFORE Viterbi:
//   1. Sync quality: sync_bits count or correlation must be high enough
//   2. Address correlation: the 49 address bits must correlate with a known
//      address pattern (general CQ address or user's private address)
//   3. Only then run the expensive Viterbi decode + parity check

/// Minimum normalised address correlation (0..1) to proceed to Viterbi.
const MIN_ADDR_CORRELATION: f32 = 0.65;

fn decode_candidate(
    candidate: &PacketCandidate,
    cfg: &RxAudioCfg,
) -> Option<RxDecoded> {
    let packet_soft = &candidate.packet_soft;
    let sync = &candidate.sync;

    if packet_soft.len() != 258 {
        return None;
    }

    // ── Gate 1: Sync quality ──
    // sync_bits and correlation are already computed by the extractor.
    // Require reasonable quality before spending CPU on address check + Viterbi.
    if sync.sync_bits < 35 && sync.correlation < 0.40 {
        log::debug!("[RX] Gate1 reject: sync_bits={} corr={:.3}", sync.sync_bits, sync.correlation);
        return None;
    }

    use msk2k_dsp::decode::decode_packet_soft;
    use msk2k_dsp::decode::is_general_addr;
    use msk2k_dsp::decode::GENERAL_ADDRESS_49;
    use msk2k_dsp::callsign::CallsignCodec;
    use msk2k_dsp::rx::RxSync;

    // ── Gate 2: Address correlation (pre-Viterbi) ──
    // packet_soft is ALREADY polarity-corrected by to_candidate().
    // Positive soft bit = hard 1, negative = hard 0.
    // Extract address bits and correlate against known patterns.
    //
    // Per DJ5HG spec Section 4: the 49-bit address encodes the SENDER's
    // callsign for directed messages. So:
    //   - CQ messages: general address (all stations)
    //   - "DJ5HG de GW4WND 26": address = GW4WND (sender)
    //   - R27/RR/73 from DJ5HG to GW4WND: address = DJ5HG (sender)
    //
    // We check: general address, our own address, and their_call's address.
    // During listen mode (their_call unknown), we accept any candidate that
    // passed the sync gate — we can't predict who might call us.
    let addr_hard = extract_addr_hard(packet_soft, sync.format_hint);

    let mut addr_pass = false;

    // Check general address (CQ/QRZ/QST)
    let general_corr = bit_match_ratio(&addr_hard, &GENERAL_ADDRESS_49);
    if general_corr >= MIN_ADDR_CORRELATION {
        addr_pass = true;
    }

    // Check our own private address (for messages FROM us — shouldn't happen,
    // but needed for completeness; loopback protection catches self-decodes later)
    if !addr_pass && !cfg.my_call.is_empty() {
        let codec = CallsignCodec::new();
        if let Ok(my_addr) = codec.generate_private_address(&cfg.my_call) {
            let my_corr = bit_match_ratio(&addr_hard, &my_addr);
            if my_corr >= MIN_ADDR_CORRELATION {
                addr_pass = true;
            }
        }
    }

    // Check their_call's private address (for messages FROM them to us)
    if !addr_pass {
        if let Some(ref their_call) = cfg.their_call {
            if !their_call.is_empty() {
                let codec = CallsignCodec::new();
                if let Ok(their_addr) = codec.generate_private_address(their_call) {
                    let their_corr = bit_match_ratio(&addr_hard, &their_addr);
                    if their_corr >= MIN_ADDR_CORRELATION {
                        addr_pass = true;
                    }
                }
            }
        }
    }

    // In listen mode (no their_call set), accept any candidate that passed
    // the sync quality gate. We don't know who might be calling us, so we
    // can't filter by address. Viterbi + parity will reject false positives.
    if !addr_pass && cfg.their_call.is_none() {
        log::info!("[RX] Gate2 bypass: listen mode, general_corr={:.3}, sync_bits={}", general_corr, sync.sync_bits);
        addr_pass = true;
    }

    if !addr_pass {
        log::info!(
            "[RX] Gate2 reject: general={:.3} my={} their={:?} sync_bits={} corr={:.3} fmt={}",
            general_corr,
            cfg.my_call,
            cfg.their_call,
            sync.sync_bits,
            sync.correlation,
            sync.format_hint,
        );
        return None;
    }

    // ── Gates passed — proceed to Viterbi decode ──
    let shifts_to_try: &[(i32, i32)] = &[
        (0, 1),    // Format-1
        (14, 2),   // Format-2 shift=14
        (29, 2),   // Format-2 shift=29
    ];

    let codec = CallsignCodec::new();

    for &(shift, fmt_hint) in shifts_to_try {
        let try_sync = RxSync {
            found: true,
            correlation: sync.correlation,
            position: sync.position,
            sync_bits: sync.sync_bits,
            polarity: sync.polarity,
            sync_shift: shift,
            sync_rotation: sync.sync_rotation,
            format_hint: fmt_hint,
        };

        if let Some(pkt) = decode_packet_soft(packet_soft, &try_sync) {
            let is_general = is_general_addr(&pkt.addr_bits);

            let msg_result = if pkt.format == 1 {
                Message::from_format1_bits(&codec, &pkt.info_bits, &pkt.addr_bits, is_general, Some(&cfg.my_call))
            } else if pkt.format == 2 {
                if cfg.my_call.is_empty() {
                    continue;
                }
                Message::from_format2_bits(
                    &codec,
                    &pkt.info_bits,
                    &pkt.addr_bits,
                    &cfg.my_call,
                    cfg.their_call.as_deref().unwrap_or(""),
                )
            } else {
                continue;
            };

            match msg_result {
                Ok(msg) => {
                    // Loopback protection
                    if !cfg.my_call.is_empty() && msg.from_call.eq_ignore_ascii_case(&cfg.my_call) {
                        return None;
                    }

                    let utc_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as i64;

                    let rx_slot = ((utc_ms / cfg.slot_len_ms as i64) % 2) as u8;

                    return Some(RxDecoded {
                        msg,
                        snr: None,
                        utc_ms,
                        rx_slot,
                    });
                }
                Err(_) => {
                    // Try next format/shift
                }
            }
        }
    }

    None
}

/// Extract 49 address hard bits from a polarity-corrected 258-bit packet.
/// Uses the appropriate interleave table based on format hint.
/// packet_soft is already polarity-corrected: positive = 1, negative = 0.
fn extract_addr_hard(packet_soft: &[f32], format_hint: i32) -> Vec<i32> {
    let mut addr = Vec::with_capacity(49);

    if format_hint == 2 {
        // Format-2: use FORMAT2_TABLE
        use msk2k_dsp::fmt2;
        for (pos, &(ref typ, _idx)) in fmt2::FORMAT2_TABLE.iter().enumerate() {
            if typ.eq_ignore_ascii_case("a") {
                if pos < packet_soft.len() {
                    addr.push(if packet_soft[pos] > 0.0 { 1 } else { 0 });
                }
            }
        }
    } else {
        // Format-1: use FORMAT1_TABLE
        use msk2k_dsp::fmt1;
        for (pos, &(typ, _idx)) in fmt1::FORMAT1_TABLE.iter().enumerate() {
            if typ == b'A' {
                if pos < packet_soft.len() {
                    addr.push(if packet_soft[pos] > 0.0 { 1 } else { 0 });
                }
            }
        }
    }

    addr
}

/// Fraction of bits that match between two bit arrays (0..1).
fn bit_match_ratio(a: &[i32], b: &[i32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let matches = a[..n].iter().zip(&b[..n]).filter(|(&x, &y)| x == y).count();
    matches as f32 / n as f32
}
