// crates/msk2k_dsp/src/rx.rs
//
// MSK2K / PSK2K Receiver DSP
//
// Refactored to use 1-Symbol Delay-and-Multiply Discriminator (MSK144 style)
// instead of the DJ5HG coherent BPSK architecture. This avoids phase wrapping
// and the need for a coherent carrier lock (which MSK lacks), making it highly
// robust for real-world noisy meteor scatter reception.
//
// Pipeline:
//   1. Real audio
//   2. Downconvert: Mix to complex baseband (1500 Hz center)
//   3. Matched Filter: 24-sample boxcar (perfect for 2000 baud MSK)
//   4. Delay-and-Multiply: Extract frequency shift directly without atan2()
//   5. Block AGC: Normalize soft bits for Viterbi ingestion
//   6. SYN: Matrix search over all 24 timing offsets
//   7. DEC: Viterbi + FEC (unchanged)

use std::f32::consts::PI;
use std::collections::VecDeque;
use log::{info, debug};
use num_complex::Complex32;

use crate::{fmt1, fmt2};

// ============================================================================
// Constants
// ============================================================================

const SAMPLE_RATE: f32 = 48_000.0;
/// MSK144 / MSK2K center frequency
const CARRIER_HZ: f32 = 1350.0;

const SAMPLES_PER_BIT: usize = 24; // 48000 / 2000
pub const PACKET_BITS:  usize = 258;
pub const SYNC_BITS:    usize = 43;

/// Two-packet analysis window
const TWO_PKT_SAMPLES: usize = 2 * PACKET_BITS * SAMPLES_PER_BIT; // 12 384

/// Evaluation stride: slide by one packet
const EVAL_STRIDE_SAMPLES: usize = PACKET_BITS * SAMPLES_PER_BIT; // 6 192

const SYNC_PATTERN_HD43: [i32; 43] = [
    0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0,
    0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0
];

// ============================================================================
// Public types
// ============================================================================

#[derive(Debug, Clone, PartialEq, Default)]
pub struct RxSync {
    pub found: bool,
    pub correlation: f32,
    pub position: i32,
    pub sync_bits: i32,
    pub polarity: i32,
    pub sync_shift: i32,
    pub sync_rotation: i32,
    pub format_hint: i32,
}

#[derive(Debug, Clone)]
pub struct PacketCandidate {
    pub packet_soft: Vec<f32>,
    pub sync: RxSync,
    pub end_index: u64,
}

// ============================================================================
// Backward-compatible public entrypoints
// ============================================================================

pub fn demodulate_48k(audio: &[f32]) -> Vec<f32> {
    demodulate_msk_soft(audio)
}

pub fn find_sync(soft_bits: &[f32]) -> RxSync {
    find_sync_soft(soft_bits)
}

pub fn extract_packet_soft(soft_bits: &[f32], sync: &RxSync) -> Option<Vec<f32>> {
    extract_packet_soft_format1(soft_bits, sync)
}

/// Batch demodulator: runs the Delay-and-Multiply pipeline on a window of audio.
/// Returns soft bits sampled at tau=0. 
pub fn demodulate_msk_soft(audio: &[f32]) -> Vec<f32> {
    if audio.is_empty() { return Vec::new(); }
    let soft_stream = demodulate_msk_delay_multiply(audio);
    
    // Sample at tau=0
    let mut soft = Vec::with_capacity(soft_stream.len() / SAMPLES_PER_BIT);
    let mut b = 0;
    while b * SAMPLES_PER_BIT < soft_stream.len() {
        soft.push(soft_stream[b * SAMPLES_PER_BIT]);
        b += 1;
    }
    soft
}

// ============================================================================
// MSK Robust Delay-and-Multiply Demodulator
// ============================================================================

/// Demodulates MSK audio entirely in the time domain using a 1-symbol 
/// delay-and-multiply approach. It produces a continuous stream of soft bits 
/// evaluated at the sample rate (48 kHz), normalized and clamped for the Viterbi decoder.
pub fn demodulate_msk_delay_multiply(audio_window: &[f32]) -> Vec<f32> {
    let n = audio_window.len();
    if n < SAMPLES_PER_BIT { return vec![0.0; n]; }

    let mut baseband = vec![num_complex::Complex32::new(0.0, 0.0); n];
    
    // 1. Downconvert to complex baseband (MSK144/MSK2K center frequency = 1500 Hz)
    let step = 2.0 * std::f32::consts::PI * CARRIER_HZ / 48000.0;
    for i in 0..n {
        let phase = step * i as f32;
        let i_val = audio_window[i] * phase.cos();
        let q_val = audio_window[i] * -phase.sin();
        baseband[i] = num_complex::Complex32::new(i_val, q_val);
    }
    
    // 2. Matched Filter (Boxcar of 24 samples = 1 bit period)
    let mut filtered = vec![Complex32::new(0.0, 0.0); n];
    let mut sum = Complex32::new(0.0, 0.0);
    for i in 0..n {
        sum += baseband[i];
        if i >= SAMPLES_PER_BIT {
            sum -= baseband[i - SAMPLES_PER_BIT];
        }
        filtered[i] = sum / (SAMPLES_PER_BIT as f32);
    }
    
    // 3. Delay-and-Multiply Discriminator (1 symbol delay)
    // The imaginary part of (curr * conj(delayed)) directly extracts the soft bit.
    let mut soft_stream = vec![0.0f32; n];
    for i in SAMPLES_PER_BIT..n {
        let curr = filtered[i];
        let delayed = filtered[i - SAMPLES_PER_BIT];
        soft_stream[i] = (curr * delayed.conj()).im;
    }
    
    // 4. Block AGC / Normalization
    let mut sq_sum = 0.0;
    for &val in &soft_stream {
        sq_sum += val * val;
    }
    let rms = (sq_sum / n as f32).sqrt();
    info!("[DSP] audio_window: n={}, soft_rms_pre_agc={:.6}", n, rms);
    if rms > 1e-6 {
        let gain = 1.0 / rms;
        for val in &mut soft_stream {
            *val = (*val * gain).clamp(-4.0, 4.0);
        }
    }
    
    soft_stream
}

// ============================================================================
// Matrix Bit-Sync Extractor
// ============================================================================

#[derive(Debug, Clone)]
pub struct MatrixSyncExtractor {
    audio_buf: VecDeque<f32>,
    buf_start_index: u64,
    total_samples: u64,
    next_eval_at: u64,
    pub corr_threshold: f32,
    f1_last_peak_sample: Option<u64>,
    f2_last_peak_sample: Option<u64>,
}

impl Default for MatrixSyncExtractor {
    fn default() -> Self {
        Self {
            audio_buf: VecDeque::with_capacity(3 * PACKET_BITS * SAMPLES_PER_BIT),
            buf_start_index: 0,
            total_samples: 0,
            next_eval_at: TWO_PKT_SAMPLES as u64,
            corr_threshold: 0.28,
            f1_last_peak_sample: None,
            f2_last_peak_sample: None,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct TimingEval {
    offset: usize,
    corr_abs: f32,
    polarity: i32,
    sync_bits: i32,
    sync_shift: i32,
    sync_rotation: i32,
    format_hint: i32,
    soft_bits: Vec<f32>,
    end_sample: u64,
}

impl MatrixSyncExtractor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_audio(&mut self, audio: &[f32]) -> Vec<PacketCandidate> {
        let mut out = Vec::new();
        let max_buf = 3 * PACKET_BITS * SAMPLES_PER_BIT;

        for &s in audio {
            self.audio_buf.push_back(s);
            self.total_samples += 1;

            while self.audio_buf.len() > max_buf {
                self.audio_buf.pop_front();
                self.buf_start_index += 1;
            }
        }

        if self.total_samples >= self.next_eval_at
            && self.audio_buf.len() >= TWO_PKT_SAMPLES
        {
            self.evaluate(&mut out);
            self.next_eval_at = self.total_samples + EVAL_STRIDE_SAMPLES as u64;
        }

        out
    }

    pub fn push_soft_bits(&mut self, _bits: &[f32]) -> Vec<PacketCandidate> {
        Vec::new()
    }

    pub fn push_phase(&mut self, _phases: &[f32]) -> Vec<PacketCandidate> {
        Vec::new()
    }

    fn evaluate(&mut self, out: &mut Vec<PacketCandidate>) {
        let buf_len = self.audio_buf.len();
        if buf_len < TWO_PKT_SAMPLES { return; }

        let window_start     = buf_len - TWO_PKT_SAMPLES;
        let window_start_abs = self.buf_start_index + window_start as u64;

        let audio_window: Vec<f32> = self.audio_buf
            .range(window_start..window_start + TWO_PKT_SAMPLES)
            .copied()
            .collect();

        // Generate continuous soft bits for the entire block using DM
        let soft_stream = demodulate_msk_delay_multiply(&audio_window);

        let mut pat0 = [0.0f32; SYNC_BITS];
        for i in 0..SYNC_BITS {
            pat0[i] = if SYNC_PATTERN_HD43[i] == 1 { 1.0 } else { -1.0 };
        }
        let pat_label_14 = rotate_left_43(&pat0, 29);
        let pat_label_29 = rotate_left_43(&pat0, 14);

        let mut f1_sync_pos: Vec<(usize, usize)> = Vec::with_capacity(SYNC_BITS);
        for (pos, &(typ, idx)) in fmt1::FORMAT1_TABLE.iter().enumerate() {
            if typ == b'S' {
                let logical = (idx - 1) as usize;
                if logical < SYNC_BITS { f1_sync_pos.push((pos, logical)); }
            }
        }
        let mut f2_sync_pos: Vec<(usize, usize)> = Vec::with_capacity(SYNC_BITS);
        for (pos, &(typ, idx)) in fmt2::FORMAT2_TABLE.iter().enumerate() {
            if typ.eq_ignore_ascii_case("s") {
                let logical = (idx - 1) as usize;
                if logical < SYNC_BITS { f2_sync_pos.push((pos, logical)); }
            }
        }

        let mut f1_best: Option<TimingEval> = None;
        let mut f2_best: Option<TimingEval> = None;

        // ── Matrix timing search (Static rigid hop for sync only) ────────────
        for tau in 0..SAMPLES_PER_BIT {
            // Directly subsample the soft_stream at offset tau
            let mut soft = Vec::with_capacity((soft_stream.len() - tau) / SAMPLES_PER_BIT);
            let mut b = 0;
            while tau + b * SAMPLES_PER_BIT < soft_stream.len() {
                soft.push(soft_stream[tau + b * SAMPLES_PER_BIT]);
                b += 1;
            }

            if soft.len() < PACKET_BITS { continue; }

            let max_start = soft.len() - PACKET_BITS;

            for start in 0..=max_start {
                let pkt = &soft[start..start + PACKET_BITS];
                let end_sample = window_start_abs
                    + tau as u64
                    + ((start + PACKET_BITS) as u64) * SAMPLES_PER_BIT as u64;

                // Format 1 sync correlation
                let mut corr_f1 = 0.0f32;
                let mut norm_sq_f1 = 0.0f32;
                for &(pos, logical) in &f1_sync_pos {
                    corr_f1   += pkt[pos] * pat0[logical];
                    norm_sq_f1 += pkt[pos] * pkt[pos];
                }
                let ca_f1 = if norm_sq_f1 > 0.0 {
                    corr_f1.abs() / (norm_sq_f1.sqrt() * (SYNC_BITS as f32).sqrt())
                } else { 0.0 };

                if ca_f1 > f1_best.as_ref().map(|b| b.corr_abs).unwrap_or(self.corr_threshold) {
                    let pol = if corr_f1 >= 0.0 { 1i32 } else { -1i32 };
                    
                    // 🟢 WSJT-X GARDNER TIMING RECOVERY
                    // We found a peak using rigid sampling. Now mathematically
                    // re-extract the exact soft bits using dynamic PLL tracking.
                    let start_sample_idx = tau + start * SAMPLES_PER_BIT;
                    let dynamic_soft = extract_dynamic_timing(&soft_stream, start_sample_idx);
                    
                    f1_best = Some(TimingEval {
                        offset: tau,
                        corr_abs: ca_f1,
                        polarity: pol,
                        sync_bits: count_f1_sync_matches(pkt, &pat0, &f1_sync_pos, pol),
                        sync_shift: 0,
                        sync_rotation: 0,
                        format_hint: 1,
                        soft_bits: dynamic_soft,
                        end_sample,
                    });
                }

                // Format 2 sync correlation
                let mut sync_f2 = [0.0f32; SYNC_BITS];
                for &(pos, logical) in &f2_sync_pos {
                    sync_f2[logical] = pkt[pos];
                }

                for (label, rotation, pat) in [
                    (0i32,  0i32,  &pat0),
                    (14,    29,    &pat_label_14),
                    (29,    14,    &pat_label_29),
                ] {
                    let mut corr = 0.0f32;
                    let mut norm_sq = 0.0f32;
                    for i in 0..SYNC_BITS {
                        corr    += sync_f2[i] * pat[i];
                        norm_sq += sync_f2[i] * sync_f2[i];
                    }
                    let ca = if norm_sq > 0.0 {
                        corr.abs() / (norm_sq.sqrt() * (SYNC_BITS as f32).sqrt())
                    } else { 0.0 };

                    if ca > f2_best.as_ref().map(|b| b.corr_abs).unwrap_or(self.corr_threshold) {
                        let pol = if corr >= 0.0 { 1i32 } else { -1i32 };
                        
                        // 🟢 WSJT-X GARDNER TIMING RECOVERY
                        let start_sample_idx = tau + start * SAMPLES_PER_BIT;
                        let dynamic_soft = extract_dynamic_timing(&soft_stream, start_sample_idx);
                        
                        f2_best = Some(TimingEval {
                            offset: tau,
                            corr_abs: ca,
                            polarity: pol,
                            sync_bits: count_sync_matches(&sync_f2, pat, pol),
                            sync_shift: label,
                            sync_rotation: rotation,
                            format_hint: 2,
                            soft_bits: dynamic_soft,
                            end_sample,
                        });
                    }
                }
            }
        }

        if let Some(eval) = f1_best {
            let sep_ok = self.f1_last_peak_sample
                .map(|last| eval.end_sample > last + EVAL_STRIDE_SAMPLES as u64 / 2)
                .unwrap_or(true);
                info!("[SYN] F1 Peak Candidate: tau={}, corr={:.3}, pol={}, sep_ok={}", 
                  eval.offset, eval.corr_abs, eval.polarity, sep_ok);
            if sep_ok {
                self.f1_last_peak_sample = Some(eval.end_sample);
                out.push(self.to_candidate(eval));
            }
        }

        if let Some(eval) = f2_best {
            let sep_ok = self.f2_last_peak_sample
                .map(|last| eval.end_sample > last + EVAL_STRIDE_SAMPLES as u64 / 2)
                .unwrap_or(true);
                info!("[SYN] F2 Peak Candidate: tau={}, corr={:.3}, shift={}, rot={}, pol={}, sep_ok={}", 
                  eval.offset, eval.corr_abs, eval.sync_shift, eval.sync_rotation, eval.polarity, sep_ok);
            if sep_ok {
                self.f2_last_peak_sample = Some(eval.end_sample);
                out.push(self.to_candidate(eval));
            }
        }
    }

    fn to_candidate(&self, eval: TimingEval) -> PacketCandidate {
        let pol = eval.polarity as f32;
        let packet_soft: Vec<f32> = eval.soft_bits.iter().map(|&b| b * pol).collect();

        let sync = RxSync {
            found: true,
            correlation: eval.corr_abs,
            position: 0,
            sync_bits: eval.sync_bits,
            polarity: eval.polarity,
            sync_shift: eval.sync_shift,
            sync_rotation: eval.sync_rotation,
            format_hint: eval.format_hint,
        };

        PacketCandidate {
            packet_soft,
            sync,
            end_index: eval.end_sample / SAMPLES_PER_BIT as u64,
        }
    }
}

pub type ContinuousPacketExtractor = MatrixSyncExtractor;

// ============================================================================
// PhaseDemodState — kept for API compatibility
// ============================================================================

#[derive(Debug, Clone)]
pub struct PhaseDemodState {
    lo_phase_acc: f32,
    i_buf: [f32; SAMPLES_PER_BIT],
    q_buf: [f32; SAMPLES_PER_BIT],
    buf_pos: usize,
    i_sum: f32,
    q_sum: f32,
    last_phase: f32,
    wrap_offset: f32,
}

impl Default for PhaseDemodState {
    fn default() -> Self {
        Self {
            lo_phase_acc: 0.0,
            i_buf: [0.0; SAMPLES_PER_BIT],
            q_buf: [0.0; SAMPLES_PER_BIT],
            buf_pos: 0,
            i_sum: 0.0,
            q_sum: 0.0,
            last_phase: 0.0,
            wrap_offset: 0.0,
        }
    }
}

impl PhaseDemodState {
    pub fn new() -> Self { Self::default() }

    pub fn push_audio(&mut self, audio: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(audio.len());
        for &sample in audio {
            let lo_phase = self.lo_phase_acc;
            self.lo_phase_acc += 2.0 * PI * CARRIER_HZ / SAMPLE_RATE;
            if self.lo_phase_acc >= 2.0 * PI { self.lo_phase_acc -= 2.0 * PI; }

            let i_new = sample * lo_phase.cos();
            let q_new = sample * (-lo_phase.sin());

            self.i_sum -= self.i_buf[self.buf_pos];
            self.q_sum -= self.q_buf[self.buf_pos];
            self.i_buf[self.buf_pos] = i_new;
            self.q_buf[self.buf_pos] = q_new;
            self.i_sum += i_new;
            self.q_sum += q_new;
            self.buf_pos = (self.buf_pos + 1) % SAMPLES_PER_BIT;

            let phase = (self.q_sum / SAMPLES_PER_BIT as f32)
                .atan2(self.i_sum / SAMPLES_PER_BIT as f32);
            let diff = phase - self.last_phase;
            if diff >  PI { self.wrap_offset -= 2.0 * PI; }
            else if diff < -PI { self.wrap_offset += 2.0 * PI; }
            self.last_phase = phase;
            out.push(phase + self.wrap_offset);
        }
        out
    }
}

pub type SoftDemodState = PhaseDemodState;

// ============================================================================
// Windowed sync search (used by batch decoders)
// ============================================================================

pub fn find_sync_soft_format1(soft_bits: &[f32]) -> RxSync {
    let mut best = RxSync::default();
    let mut best_corr_abs = 0.0f32;
    if soft_bits.len() < PACKET_BITS { return best; }

    let mut pat0 = [0.0f32; SYNC_BITS];
    for i in 0..SYNC_BITS {
        pat0[i] = if SYNC_PATTERN_HD43[i] == 1 { 1.0 } else { -1.0 };
    }

    for pos in 0..=soft_bits.len().saturating_sub(PACKET_BITS) {
        let window = &soft_bits[pos..pos + PACKET_BITS];
        let sync_soft = deinterleave_sync_soft_f1(window);
        let corr = dot43(&sync_soft, &pat0);
        let corr_abs = corr.abs();
        if corr_abs > best_corr_abs {
            best_corr_abs = corr_abs;
            best = RxSync {
                found: true,
                position: pos as i32,
                correlation: corr_abs / SYNC_BITS as f32,
                polarity: if corr >= 0.0 { 1 } else { -1 },
                sync_bits: count_sync_matches(&sync_soft, &pat0, if corr >= 0.0 { 1 } else { -1 }),
                sync_shift: 0,
                sync_rotation: 0,
                format_hint: 1,
            };
        }
    }
    if best.correlation < 0.25 { best.found = false; }
    best
}

pub fn find_sync_soft(soft_bits: &[f32]) -> RxSync {
    let mut best = RxSync::default();
    let mut best_corr_abs = 0.0f32;
    if soft_bits.len() < PACKET_BITS { return best; }

    let mut pat0 = [0.0f32; SYNC_BITS];
    for i in 0..SYNC_BITS {
        pat0[i] = if SYNC_PATTERN_HD43[i] == 1 { 1.0 } else { -1.0 };
    }
    let pat_label_14 = rotate_left_43(&pat0, 29);
    let pat_label_29 = rotate_left_43(&pat0, 14);

    for pos in 0..=soft_bits.len().saturating_sub(PACKET_BITS) {
        let window = &soft_bits[pos..pos + PACKET_BITS];

        // Format 1
        {
            let s = deinterleave_sync_soft_f1(window);
            let corr = dot43(&s, &pat0);
            let ca = corr.abs();
            if ca > best_corr_abs {
                best_corr_abs = ca;
                best = RxSync {
                    found: true,
                    position: pos as i32,
                    correlation: ca / SYNC_BITS as f32,
                    polarity: if corr >= 0.0 { 1 } else { -1 },
                    sync_bits: count_sync_matches(&s, &pat0, if corr >= 0.0 { 1 } else { -1 }),
                    sync_shift: 0,
                    sync_rotation: 0,
                    format_hint: 1,
                };
            }
        }

        // Format 2 (three pattern rotations)
        {
            let s = deinterleave_sync_soft_f2(window);
            for (label, rotation, pat) in [
                (0i32,  0i32,  &pat0),
                (14,    29,    &pat_label_14),
                (29,    14,    &pat_label_29),
            ] {
                let corr = dot43(&s, pat);
                let ca = corr.abs();
                if ca > best_corr_abs {
                    best_corr_abs = ca;
                    best = RxSync {
                        found: true,
                        position: pos as i32,
                        correlation: ca / SYNC_BITS as f32,
                        polarity: if corr >= 0.0 { 1 } else { -1 },
                        sync_bits: count_sync_matches(&s, pat, if corr >= 0.0 { 1 } else { -1 }),
                        sync_shift: label,
                        sync_rotation: rotation,
                        format_hint: 2,
                    };
                }
            }
        }
    }

    if best.correlation < 0.25 { best.found = false; }
    best
}

// ============================================================================
// Packet extraction
// ============================================================================

pub fn extract_packet_soft_format1(soft_bits: &[f32], sync: &RxSync) -> Option<Vec<f32>> {
    if !sync.found { return None; }
    let start = sync.position as usize;
    let end   = start + PACKET_BITS;
    if end > soft_bits.len() { return None; }
    let pol = sync.polarity as f32;
    Some(soft_bits[start..end].iter().map(|&b| b * pol).collect())
}

// ============================================================================
// Utility functions
// ============================================================================

fn dot43(a: &[f32; SYNC_BITS], b: &[f32; SYNC_BITS]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..SYNC_BITS { s += a[i] * b[i]; }
    s
}

fn rotate_left_43(pat: &[f32; SYNC_BITS], k: usize) -> [f32; SYNC_BITS] {
    let mut out = [0.0f32; SYNC_BITS];
    for i in 0..SYNC_BITS { out[i] = pat[(i + k) % SYNC_BITS]; }
    out
}

fn count_sync_matches(
    sync_soft: &[f32; SYNC_BITS],
    pat: &[f32; SYNC_BITS],
    polarity: i32,
) -> i32 {
    let pol = polarity as f32;
    (0..SYNC_BITS).filter(|&i| {
        let v = sync_soft[i] * pol;
        (v > 0.0 && pat[i] > 0.0) || (v < 0.0 && pat[i] < 0.0)
    }).count() as i32
}

fn count_f1_sync_matches(
    pkt: &[f32],
    pat: &[f32; SYNC_BITS],
    positions: &[(usize, usize)],
    polarity: i32,
) -> i32 {
    let pol = polarity as f32;
    positions.iter().filter(|&&(pos, logical)| {
        let v = pkt[pos] * pol;
        (v > 0.0 && pat[logical] > 0.0) || (v < 0.0 && pat[logical] < 0.0)
    }).count() as i32
}

fn deinterleave_sync_soft_f1(window: &[f32]) -> [f32; SYNC_BITS] {
    let mut out = [0.0f32; SYNC_BITS];
    for (pos, &(typ, idx)) in fmt1::FORMAT1_TABLE.iter().enumerate() {
        if typ == b'S' {
            let logical = (idx - 1) as usize;
            if logical < SYNC_BITS && pos < window.len() {
                out[logical] = window[pos];
            }
        }
    }
    out
}

fn deinterleave_sync_soft_f2(window: &[f32]) -> [f32; SYNC_BITS] {
    let mut out = [0.0f32; SYNC_BITS];
    for (pos, &(typ, idx)) in fmt2::FORMAT2_TABLE.iter().enumerate() {
        if typ.eq_ignore_ascii_case("s") {
            let logical = (idx - 1) as usize;
            if logical < SYNC_BITS && pos < window.len() {
                out[logical] = window[pos];
            }
        }
    }
    out
}

// ============================================================================
// Gardner Timing Error Detector (TED)
// ============================================================================

/// Uses a Gardner phase-locked loop (PLL) to dynamically surf the exact center
/// of the MSK eye diagram. This corrects clock drift in real-world soundcards
/// or WAV file playback, similar to the WSJT-X MSK144 approach.
fn extract_dynamic_timing(soft_stream: &[f32], start_sample: usize) -> Vec<f32> {
    let mut soft_bits = Vec::with_capacity(PACKET_BITS);
    
    let mut current_idx = start_sample as f32;
    let mut prev_soft = 0.0f32;
    let mut samples_per_bit = SAMPLES_PER_BIT as f32;

    // PLL Proportional and Integral loop gains
    let alpha = 0.1_f32;
    let beta = 0.005_f32;

    for i in 0..PACKET_BITS {
        let idx_usize = current_idx.round() as usize;
        
        // Bounds check in case the packet runs off the edge of the block
        if idx_usize >= soft_stream.len() {
            soft_bits.push(0.0);
            continue;
        }

        let soft = soft_stream[idx_usize];
        soft_bits.push(soft);

        // Once we have a previous bit, we can check the transition between them
        if i > 0 {
            // Find the sample exactly halfway between the previous bit and the current bit
            let boundary_idx = (current_idx - samples_per_bit / 2.0).round() as usize;
            
            if boundary_idx < soft_stream.len() {
                let boundary_val = soft_stream[boundary_idx];
                
                // Only run the TED if there is a zero-crossing transition (+ to - or - to +)
                let sign_diff = prev_soft.signum() - soft.signum();
                if sign_diff.abs() > 0.5 {
                    // Gardner Error Math: E = V_boundary * (sgn(V_prev) - sgn(V_curr))
                    let error = boundary_val * sign_diff * 0.5;
                    
                    // Nudge the fractional sampling pointer
                    current_idx += alpha * error;
                    
                    // Slightly adjust the long-term clock rate (to handle actual drift)
                    samples_per_bit += beta * error;
                    
                    // Clamp to prevent wild runaways on noisy data (±2 samples)
                    samples_per_bit = samples_per_bit.clamp(22.0, 26.0);
                }
            }
        }

        prev_soft = soft;
        current_idx += samples_per_bit;
    }

    // Log the timing drift for diagnostics
    if (samples_per_bit - SAMPLES_PER_BIT as f32).abs() > 0.05 {
        debug!("[TED] Clock drift compensated. Final SPB: {:.3} (drift: {:.2} Hz)", 
               samples_per_bit, 
               (48000.0 / SAMPLES_PER_BIT as f32) - (48000.0 / samples_per_bit));
    }

    soft_bits
}