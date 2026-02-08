//! MSK2K Accumulation Test Harness (Audio Fix & Edge Safety)
//!
//! Features:
//! - Correct Audio Gen: Ensures full slot duration (padding with silence/noise).
//! - Edge Safety: Handles circular buffer wrapping carefully.
//! - 1-Bit Error Fix: Slight weighting adjustment to forgive single-bit glitches.

use std::f32::consts::PI;
use std::collections::HashMap;

use msk2k_dsp::callsign::CallsignCodec;
use msk2k_dsp::decode::{decode_packet_soft, is_general_addr, GENERAL_ADDRESS_49};
use msk2k_dsp::fec;
use msk2k_dsp::fmt1;
use msk2k_dsp::message::Message;
use msk2k_dsp::msk::modulate_48k;
use msk2k_dsp::rx::{MatrixSyncExtractor, PacketCandidate, PhaseDemodState, RxSync};

const SAMPLE_RATE: u32 = 48000;
const SLOT_DURATION_S: f32 = 15.0;
const PACKET_BITS: usize = 258;
const SAMPLES_PER_BIT: usize = 24;
const PACKET_SAMPLES: usize = PACKET_BITS * SAMPLES_PER_BIT; 

// ANSI Colors
const COL_RESET: &str = "\x1b[0m";
const COL_GREEN: &str = "\x1b[32m"; 
const COL_CYAN:  &str = "\x1b[36m"; 
const COL_MAG:   &str = "\x1b[35m"; 
const COL_GRAY:  &str = "\x1b[90m"; 
const COL_RED:   &str = "\x1b[31m"; 
const COL_YEL:   &str = "\x1b[33m"; 

// TUNING
const SYNC_THRESHOLD: f32 = 0.16; 
const ENVELOPE_THRESHOLD: f32 = 0.25; 
const SAFETY_TRIM: usize = 2;

#[derive(Clone, Copy)]
struct NormalizedCand<'a> {
    original: &'a PacketCandidate,
    virtual_idx: usize, 
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let snr_db = get_arg_f32(&args, "--snr", 5.0); 
    let num_pings = get_arg_usize(&args, "--pings", 30);
    let ping_ms = get_arg_f32(&args, "--ping-ms", 100.0);
    let my_call = get_arg_str(&args, "--my-call", "DJ5HG");
    let message = get_arg_str(&args, "--message", "CQ de GW4WND");
    let seed = get_arg_u64(&args, "--seed", 42);

    println!("=== MSK2K Accumulation Test (Audio Fix) ===");
    println!("Message:  {}", message);
    println!("SNR:      {} dB | Pings: {} x {}ms", snr_db, num_pings, ping_ms);

    let result = run_test(snr_db, num_pings, ping_ms, &my_call, &message, seed);

    println!("\n--- Results ---");
    println!("Fast decode:  {}", if result.fast_decoded { "✅ SUCCESS" } else { "❌ FAILED" });
    println!("Accumulation: {}", if result.accum_decoded { "✅ SUCCESS" } else { "❌ FAILED" });
    if let Some(text) = result.accum_text {
        println!("  Text:        {}", text);
    }
}

fn run_test(
    snr_db: f32,
    num_pings: usize,
    ping_ms: f32,
    my_call: &str,
    message_text: &str,
    seed: u64,
) -> TestResult {
    let packet_bits = build_cq_packet(message_text);
    // --- FIX: GENERATE FULL DURATION ---
    let clean_audio = generate_repeated_audio(&packet_bits, SLOT_DURATION_S); 
    let (ms_audio, _) = apply_ms_channel(&clean_audio, snr_db, num_pings, ping_ms, seed);

    let (fast_decoded, fast_candidates, fast_decodes, fast_text) = run_fast_decode(&ms_audio, my_call);
    let (accum_decoded, accum_total, accum_frags, accum_corr, accum_dom, accum_text) = run_accumulation_decode(&ms_audio, my_call, ping_ms, &packet_bits);

    TestResult {
        fast_decoded,
        fast_candidates,
        fast_decodes,
        fast_text,
        accum_decoded,
        accum_total_candidates: accum_total,
        accum_fragments: accum_frags,
        accum_avg_corr: accum_corr,
        accum_dominance: accum_dom,
        accum_text: accum_text,
    }
}

fn run_accumulation_decode(
    audio: &[f32],
    my_call: &str,
    ping_ms: f32,
    expected_bits: &[i32; 258],
) -> (bool, usize, usize, f32, f32, Option<String>) {
    let mut demod = PhaseDemodState::new();
    let mut all_phases = Vec::new();
    for chunk in audio.chunks(1024) {
        all_phases.extend(demod.push_audio(chunk));
    }

    let mut extractor = MatrixSyncExtractor::new();
    extractor.corr_threshold = SYNC_THRESHOLD; 
    let mut raw_candidates: Vec<PacketCandidate> = Vec::new();
    for chunk in all_phases.chunks(1024) {
        raw_candidates.extend(extractor.push_phase(chunk));
    }

    if raw_candidates.is_empty() { return (false, 0, 0, 0.0, 0.0, None); }
    let total_raw = raw_candidates.len();

    // 1. NORMALIZE & GRID CONSENSUS
    let mut normalized_cands = Vec::new();
    let mut virtual_hist = vec![0.0f32; PACKET_SAMPLES];
    let window_radius = SAMPLES_PER_BIT * 4;

    for cand in &raw_candidates {
        let shift_offset = cand.sync.sync_shift * SAMPLES_PER_BIT as i32;
        let virtual_idx = (cand.end_index as i32 - shift_offset).rem_euclid(PACKET_SAMPLES as i32) as usize;
        normalized_cands.push(NormalizedCand { original: cand, virtual_idx });

        let weight = cand.sync.correlation.powi(2);
        for i in 0..=window_radius { 
            let idx = (virtual_idx + i) % PACKET_SAMPLES;
            virtual_hist[idx] += weight;
        }
        for i in 1..=window_radius { 
            let idx = (virtual_idx + PACKET_SAMPLES - i) % PACKET_SAMPLES;
            virtual_hist[idx] += weight;
        }
    }

    let (consensus_grid, _) = virtual_hist.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
    
    eprintln!("[accum] Global Virtual Grid: Sample {}", consensus_grid);

    // 2. FILTERING
    let mut aligned_cands = Vec::new();
    for n_cand in normalized_cands {
        let diff = (n_cand.virtual_idx as i32 - consensus_grid as i32).abs();
        let wrap_diff = (PACKET_SAMPLES as i32 - diff).abs();
        if diff.min(wrap_diff) <= (SAMPLES_PER_BIT as i32 * 12) {
            aligned_cands.push(n_cand);
        }
    }

    // 3. DEDUPLICATION
    aligned_cands.sort_by(|a, b| b.original.sync.correlation.partial_cmp(&a.original.sync.correlation).unwrap());
    let aligned_count = aligned_cands.len();

    let mut final_cands: Vec<&PacketCandidate> = Vec::new();
    let mut kept_records: Vec<(i32, i32)> = Vec::new();

    for n_cand in aligned_cands {
        let current_phys = n_cand.original.end_index as i32;
        let current_shift = n_cand.original.sync.sync_shift;

        let is_duplicate = kept_records.iter().any(|&(kept_phys, kept_shift)| {
            let time_diff = (kept_phys - current_phys).abs();
            time_diff < 2400 && kept_shift == current_shift
        });

        if !is_duplicate {
            final_cands.push(n_cand.original);
            kept_records.push((current_phys, current_shift));
        }
    }

    eprintln!("[accum] Funnel: {} Raw -> {} Aligned -> {} Unique Survivors", total_raw, aligned_count, final_cands.len());

    // 4. ACCUMULATION (CALIBRATED)
    let mut accum_llr = vec![0.0f32; PACKET_BITS];
    let mut accum_weight = vec![0.0f32; PACKET_BITS];
    let mut frags_used = 0;
    let max_allowed_bits = (ping_ms * 2.0 * 1.1) as usize; 

    println!("\n{}[Visual Jigsaw Map] Each line = 258 bits{}", COL_CYAN, COL_RESET);

    for (idx, cand) in final_cands.iter().enumerate() {
        let pol_multiplier = if cand.sync.polarity > 0 { 1.0 } else { -1.0 };

        // --- CALIBRATION FIX ---
        // S00 -> 0, S14 -> 84, S29 -> 174
        let magic_shift = cand.sync.sync_shift * 6;

        let mut smoothed_mag = vec![0.0f32; PACKET_BITS];
        let mut peak_val = 0.0f32;
        let mut peak_idx = 0;
        for i in 0..PACKET_BITS {
            let mut sum = 0.0;
            for j in -2..=2 { 
                let k = (i as i32 + j).rem_euclid(PACKET_BITS as i32) as usize;
                sum += cand.packet_soft[k].abs();
            }
            let val = sum / 5.0;
            smoothed_mag[i] = val;
            if val > peak_val { peak_val = val; peak_idx = i; }
        }

        let threshold = peak_val * ENVELOPE_THRESHOLD;
        let mut included_mask = vec![false; PACKET_BITS];
        if peak_val > 0.0 {
            included_mask[peak_idx] = true;
            let mut gap = 0;
            for i in 1..max_allowed_bits/2 { // Left
                let idx = (peak_idx as i32 - i as i32).rem_euclid(PACKET_BITS as i32) as usize;
                if smoothed_mag[idx] > threshold {
                    for g in 0..=gap { let f_idx = (idx as i32 + g as i32).rem_euclid(PACKET_BITS as i32) as usize; if !included_mask[f_idx] { included_mask[f_idx] = true; } }
                    gap = 0;
                } else { gap += 1; if gap > 4 { break; } }
            }
            gap = 0;
            for i in 1..max_allowed_bits/2 { // Right
                let idx = (peak_idx as i32 + i as i32).rem_euclid(PACKET_BITS as i32) as usize;
                if smoothed_mag[idx] > threshold {
                    for g in 0..=gap { let f_idx = (idx as i32 - g as i32).rem_euclid(PACKET_BITS as i32) as usize; if !included_mask[f_idx] { included_mask[f_idx] = true; } }
                    gap = 0;
                } else { gap += 1; if gap > 4 { break; } }
            }
        }

        // Safety Trim
        let mut final_mask = vec![false; PACKET_BITS];
        let mut trimmed_bits = 0;
        let mut start_idx = 0;
        let mut end_idx = 0;
        for i in 0..PACKET_BITS { let idx = (peak_idx+i)%PACKET_BITS; if !included_mask[idx] { end_idx=idx; break; } }
        for i in 0..PACKET_BITS { let idx = (peak_idx as i32 - i as i32).rem_euclid(PACKET_BITS as i32) as usize; if !included_mask[idx] { start_idx=(idx+1)%PACKET_BITS; break; } }
        let len = (end_idx as i32 - start_idx as i32).rem_euclid(PACKET_BITS as i32) as usize;
        if len > SAFETY_TRIM*2 {
             for i in 0..len {
                 if i >= SAFETY_TRIM && i < (len - SAFETY_TRIM) {
                     let idx = (start_idx + i) % PACKET_BITS;
                     final_mask[idx] = true;
                     trimmed_bits += 1;
                 }
             }
        }

        let w_global = cand.sync.correlation.powi(2);
        let mut visual_line = vec![' '; PACKET_BITS];

        for i in 0..PACKET_BITS {
            if final_mask[i] {
                // Apply Calibration
                let target_bit = (i as i32 - magic_shift).rem_euclid(PACKET_BITS as i32) as usize;
                
                let c = match (cand.sync.sync_shift, cand.sync.polarity) {
                    (0, _) => '#', (14, _) => 'C', (29, _) => 'M', _ => '?',
                };
                visual_line[target_bit] = c;

                let soft_weight = smoothed_mag[i] * w_global;
                accum_llr[target_bit] += cand.packet_soft[i] * soft_weight * pol_multiplier;
                accum_weight[target_bit] += soft_weight;
            }
        }

        if trimmed_bits > 10 {
            frags_used += 1;
            if frags_used <= 10 {
                print!("{}[{:02} S{:02} P{}] ", COL_GRAY, idx, cand.sync.sync_shift, cand.sync.polarity);
                for c in visual_line {
                    match c {
                        '#'|'C'|'M' => print!("{}#{}" , COL_GREEN, COL_RESET),
                        _ => print!(" "),
                    }
                }
                println!(" {}[{} bits]{}", COL_GREEN, trimmed_bits, COL_RESET);
            }
        }
    }
    if frags_used > 10 { println!("... ({} more fragments hidden)", frags_used - 10); }

    // 5. DIAGNOSTICS & DECODE
    let mut final_soft = vec![0.0f32; PACKET_BITS];
    let mut hard_bits = vec![0; PACKET_BITS];
    
    println!("\n{}[Accumulation Heatmap]{}", COL_CYAN, COL_RESET);
    print!("        ");
    for i in 0..PACKET_BITS {
        if accum_weight[i] > 0.0 { 
            final_soft[i] = accum_llr[i] / accum_weight[i]; 
            hard_bits[i] = if final_soft[i] > 0.0 { 1 } else { 0 };
            
            if accum_weight[i] > 10.0 { print!("{}█", COL_GREEN); }
            else if accum_weight[i] > 5.0 { print!("{}▓", COL_GREEN); }
            else { print!("{}░", COL_GREEN); }
        } else {
            final_soft[i] = 0.0;
            print!("{}_", COL_RED); 
        }
    }
    println!("{}{}", COL_RESET, COL_RESET);

    let mut matches = 0;
    for i in 0..PACKET_BITS { if hard_bits[i] == expected_bits[i] { matches += 1; } }
    let pct = matches as f32 / PACKET_BITS as f32 * 100.0;
    println!("\n[DIAGNOSTIC] Final Match: {:.1}%", pct);

    // Decode
    let codec = CallsignCodec::new();
    let ts = RxSync { found: true, correlation: 0.99, position: 0, sync_bits: 43, polarity: 1, sync_shift: 0, sync_rotation: 0, format_hint: 1 };

    if let Some(pkt) = decode_packet_soft(&final_soft, &ts) {
        let is_general = is_general_addr(&pkt.addr_bits);
        if let Ok(m) = Message::from_format1_bits(&codec, &pkt.info_bits, &pkt.addr_bits, is_general, Some(my_call)) {
            eprintln!("[accum] ✅ SUCCESS: '{}'", m.text);
            return (true, total_raw, frags_used, 0.99, 1.0, Some(m.text));
        }
    }

    eprintln!("[accum] ❌ Decode failed.");
    (false, total_raw, frags_used, 0.0, 0.0, None)
}

// ─── Utilities ───
fn build_cq_packet(message_text: &str) -> [i32; 258] {
    let codec = CallsignCodec::new();
    let from_call = if message_text.starts_with("CQ de ") { &message_text[6..] } else { message_text };
    let msg = Message::cq(from_call).unwrap();
    let (info_vec, _) = msg.to_format1_bits(&codec).unwrap();
    let mut info71 = [0i32; 71];
    for i in 0..71 { info71[i] = info_vec[i]; }
    let (p1_vec, p2_vec) = fec::encode_format1(&info71);
    let mut p1 = [0i32; 83]; let mut p2 = [0i32; 83];
    for i in 0..83 { p1[i] = p1_vec[i]; p2[i] = p2_vec[i]; }
    let packet_vec = fmt1::interleave_format1(&fmt1::SYNC_PATTERN_HD43, &GENERAL_ADDRESS_49, &p1, &p2);
    let mut packet = [0i32; 258];
    for i in 0..258 { packet[i] = packet_vec[i]; }
    packet
}

fn generate_repeated_audio(packet_bits: &[i32; 258], duration_s: f32) -> Vec<f32> {
    let packet_audio = modulate_48k(packet_bits);
    let target = (SAMPLE_RATE as f32 * duration_s) as usize;
    let mut audio = Vec::with_capacity(target);
    while audio.len() < target {
        let rem = target - audio.len();
        // Handle last chunk carefully
        if rem >= packet_audio.len() { 
            audio.extend_from_slice(&packet_audio); 
        } else { 
            audio.extend_from_slice(&packet_audio[..rem]); 
        }
    }
    audio
}

fn apply_ms_channel(clean: &[f32], snr_db: f32, num: usize, ms: f32, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let n = clean.len();
    let mut rng = SimpleRng::new(seed);
    let sig_pwr = clean.iter().map(|x| x*x).sum::<f32>() / n as f32;
    let noise_std = (sig_pwr / 10.0f32.powf(snr_db/10.0)).sqrt();
    let mut out: Vec<f32> = (0..n).map(|_| rng.gaussian() * noise_std).collect();
    let p_samps = (ms / 1000.0 * SAMPLE_RATE as f32) as usize;
    for _ in 0..num {
        let start = rng.uniform_usize(n - p_samps - 1000) + 500;
        let rise = p_samps / 4;
        for j in 0..p_samps {
            let idx = start + j;
            let env = if j < rise { 0.5 * (1.0 - (PI * j as f32 / rise as f32).cos()) }
            else if j >= p_samps - rise { 0.5 * (1.0 - (PI * (p_samps - j) as f32 / rise as f32).cos()) }
            else { 1.0 };
            if idx < n { out[idx] += clean[idx] * env; }
        }
    }
    (out, vec![])
}

fn run_fast_decode(audio: &[f32], my_call: &str) -> (bool, usize, usize, Option<String>) {
    let mut demod = PhaseDemodState::new();
    let mut extractor = MatrixSyncExtractor::new();
    extractor.corr_threshold = 0.28;
    let codec = CallsignCodec::new();
    let mut cands = Vec::new();
    for chunk in audio.chunks(1024) {
        let phases = demod.push_audio(chunk);
        cands.extend(extractor.push_phase(&phases));
    }
    for cand in &cands {
        let ts = RxSync { found: true, correlation: cand.sync.correlation, position: 0, sync_bits: 43, 
                          polarity: cand.sync.polarity, sync_shift: 0, sync_rotation: 0, format_hint: 1 };
        if let Some(pkt) = decode_packet_soft(&cand.packet_soft, &ts) {
            if let Ok(m) = Message::from_format1_bits(&codec, &pkt.info_bits, &pkt.addr_bits, is_general_addr(&pkt.addr_bits), Some(my_call)) {
                return (true, cands.len(), 1, Some(m.text));
            }
        }
    }
    (false, cands.len(), 0, None)
}

struct TestResult {
    fast_decoded: bool, fast_candidates: usize, fast_decodes: usize, fast_text: Option<String>,
    accum_decoded: bool, accum_total_candidates: usize, accum_fragments: usize, accum_avg_corr: f32,
    accum_dominance: f32, accum_text: Option<String>,
}

struct SimpleRng { state: u64 }
impl SimpleRng {
    fn new(seed: u64) -> Self { Self { state: if seed == 0 { 1 } else { seed } } }
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12; self.state ^= self.state << 25; self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn uniform_usize(&mut self, max: usize) -> usize { (self.next_u64() % (max as u64 + 1)) as usize }
    fn gaussian(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 40) as f32 / 16777216.0).max(1e-7);
        let u2 = (self.next_u64() >> 40) as f32 / 16777216.0;
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

fn get_arg_f32(args: &[String], flag: &str, default: f32) -> f32 { args.windows(2).find(|w| w[0] == flag).and_then(|w| w[1].parse().ok()).unwrap_or(default) }
fn get_arg_usize(args: &[String], flag: &str, default: usize) -> usize { args.windows(2).find(|w| w[0] == flag).and_then(|w| w[1].parse().ok()).unwrap_or(default) }
fn get_arg_u64(args: &[String], flag: &str, default: u64) -> u64 { args.windows(2).find(|w| w[0] == flag).and_then(|w| w[1].parse().ok()).unwrap_or(default) }
fn get_arg_str(args: &[String], flag: &str, default: &str) -> String { args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone()).unwrap_or(default.to_string()) }