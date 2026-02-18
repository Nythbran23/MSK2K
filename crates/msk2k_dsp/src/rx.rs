// crates/msk2k_dsp/src/rx.rs
//
// MSK2K Receiver DSP primitives
//
// Implements the DJ5HG PSK2K V5 receiver architecture (Section 8.2):
//   1. Quadrature demod → unwrapped phase at sample rate
//   2. Matrix-based bit synchronisation: try ALL sub-bit timing offsets
//   3. Sync correlation picks the best timing + packet alignment in one shot
//
// This avoids a tracking PLL and works on very short meteor pings.

use std::f32::consts::PI;
use std::collections::VecDeque;

use crate::{fmt1, fmt2};

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

const SAMPLE_RATE: f32 = 48_000.0;
const CARRIER_HZ: f32 = 1500.0;
const SAMPLES_PER_BIT: usize = 24; // 48k / 2000
const LO_PERIOD: u64 = 32; // 48000/1500 = 32 samples exactly

pub const PACKET_BITS: usize = 258;
pub const SYNC_BITS: usize = 43;

const SYNC_PATTERN_HD43: [i32; 43] = [
    0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0,
    0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0
];

// ============================================================================
// Backwards-compatible entrypoints (used by windowed decoders)
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

// ============================================================================
// Stateless demod (for windowed/batch use)
// ============================================================================

/// Phase-based MSK soft demodulator (48 kHz, 2000 bps -> 24 samples/bit).
pub fn demodulate_msk_soft(audio: &[f32]) -> Vec<f32> {
    if audio.is_empty() { return Vec::new(); }

    let mut phases = Vec::with_capacity(audio.len());
    let mut prev_i = 1.0f32;
    let mut prev_q = 0.0f32;

    for (idx, &sample) in audio.iter().enumerate() {
        let lo_phase = 2.0 * PI * CARRIER_HZ * (idx as f32 / SAMPLE_RATE);
        let lo_i = lo_phase.cos();
        let lo_q = -lo_phase.sin();

        let i = sample * lo_i;
        let q = sample * lo_q;

        prev_i = 0.95 * prev_i + 0.05 * i;
        prev_q = 0.95 * prev_q + 0.05 * q;

        phases.push(prev_q.atan2(prev_i));
    }

    let unwrapped = unwrap_phases(&phases);

    let avg_window = SAMPLES_PER_BIT / 4; // 6
    let n_bits = unwrapped.len() / SAMPLES_PER_BIT;
    let mut soft_bits = Vec::with_capacity(n_bits);

    for i in 0..n_bits.saturating_sub(1).max(0) {
        let start = i * SAMPLES_PER_BIT;
        let end = (i + 1) * SAMPLES_PER_BIT;

        if end > unwrapped.len() {
            break;
        }

        let sum_a: f32 = unwrapped[start..start + avg_window].iter().sum();
        let phase_a = sum_a / avg_window as f32;

        let sum_b: f32 = unwrapped[end - avg_window..end].iter().sum();
        let phase_b = sum_b / avg_window as f32;

        let delta_phi = phase_b - phase_a;
        let soft_bit = delta_phi / (PI / 2.0);

        soft_bits.push(soft_bit.clamp(-4.0, 4.0));
    }

    soft_bits
}

// ============================================================================
// Streaming phase demodulator (sample-rate output)
// ============================================================================

/// Streaming quadrature demodulator that outputs unwrapped phase at sample rate.
///
/// Uses a matched filter (boxcar average over one bit period = 24 samples)
/// on the I/Q channels after downconversion, per DJ5HG Section 8 / Figure 6.
/// This maximises SNR for sinc-pulse BPSK at 2000 bps.
#[derive(Debug, Clone)]
pub struct PhaseDemodState {
    sample_index: u64,
    // Boxcar matched filter buffers (one bit period = 24 samples)
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
            sample_index: 0,
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
    pub fn new() -> Self {
        Self::default()
    }

    /// Push audio samples. Returns unwrapped phase values at sample rate.
    pub fn push_audio(&mut self, audio: &[f32]) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }

        let mut out = Vec::with_capacity(audio.len());

        for &sample in audio {
            let lo_phase = 2.0 * PI * CARRIER_HZ
                * ((self.sample_index % LO_PERIOD) as f32 / SAMPLE_RATE);
            let lo_i = lo_phase.cos();
            let lo_q = -lo_phase.sin();

            let i_new = sample * lo_i;
            let q_new = sample * lo_q;

            // Boxcar matched filter: subtract outgoing sample, add incoming
            self.i_sum -= self.i_buf[self.buf_pos];
            self.q_sum -= self.q_buf[self.buf_pos];
            self.i_buf[self.buf_pos] = i_new;
            self.q_buf[self.buf_pos] = q_new;
            self.i_sum += i_new;
            self.q_sum += q_new;
            self.buf_pos = (self.buf_pos + 1) % SAMPLES_PER_BIT;

            let avg_i = self.i_sum / SAMPLES_PER_BIT as f32;
            let avg_q = self.q_sum / SAMPLES_PER_BIT as f32;

            let phase = avg_q.atan2(avg_i);
            let diff = phase - self.last_phase;
            if diff > PI {
                self.wrap_offset -= 2.0 * PI;
            } else if diff < -PI {
                self.wrap_offset += 2.0 * PI;
            }
            let unwrapped = phase + self.wrap_offset;
            self.last_phase = phase;

            out.push(unwrapped);
            self.sample_index += 1;
        }

        out
    }
}

// Keep the old name as an alias for API compatibility
pub type SoftDemodState = PhaseDemodState;

// ============================================================================
// DJ5HG Matrix Bit-Sync Extractor (Section 8.2)
// ============================================================================
//
// DJ5HG's approach (at 16kHz, 8 samples/bit):
//   - Buffer 2 packets = 2 × 258 × 8 = 4128 samples
//   - Reshape to matrix 86 × 48 (86 = 4128/8/6, 48 = 8×6)
//   - Correlate each column with sync pattern → 48 correlation values
//   - Best column determines bit-timing offset
//
// At 48kHz, 24 samples/bit:
//   - 2 packets = 2 × 258 × 24 = 12384 samples
//   - For each of 24 sub-bit offsets (τ), demodulate → ~516 soft bits
//   - Slide 258-bit window across those soft bits
//   - Run sync correlation (F1 and F2) on each window
//   - Best correlation across all offsets and positions → the packet candidate
//
// Evaluated every packet-length (6192 samples) with 2-packet overlap.

/// Two-packet analysis window in samples.
const TWO_PKT_SAMPLES: usize = 2 * PACKET_BITS * SAMPLES_PER_BIT; // 12384

/// How often to evaluate (in samples). Slide by one packet.
const EVAL_STRIDE_SAMPLES: usize = PACKET_BITS * SAMPLES_PER_BIT; // 6192

/// A single packet candidate emitted by the extractor.
#[derive(Debug, Clone)]
pub struct PacketCandidate {
    pub packet_soft: Vec<f32>,
    pub sync: RxSync,
    pub end_index: u64,
}

/// DJ5HG-style matrix bit-synchronisation extractor.
///
/// Buffers unwrapped phase samples and periodically evaluates all possible
/// bit-timing offsets to find the best sync correlation.
#[derive(Debug, Clone)]
pub struct MatrixSyncExtractor {
    /// Circular buffer of unwrapped phase samples
    phase_buf: VecDeque<f32>,
    /// Absolute sample index of the first element in phase_buf
    buf_start_index: u64,
    /// Absolute sample count (total samples pushed)
    total_samples: u64,
    /// Next sample index at which to run an evaluation
    next_eval_at: u64,

    // Tunables
    pub corr_threshold: f32,

    // Peak tracking (separate for F1 and F2)
    f1_last_peak_sample: Option<u64>,
    f2_last_peak_sample: Option<u64>,
}

impl Default for MatrixSyncExtractor {
    fn default() -> Self {
        Self {
            phase_buf: VecDeque::with_capacity(TWO_PKT_SAMPLES + EVAL_STRIDE_SAMPLES),
            buf_start_index: 0,
            total_samples: 0,
            next_eval_at: TWO_PKT_SAMPLES as u64,
            corr_threshold: 0.28,
            f1_last_peak_sample: None,
            f2_last_peak_sample: None,
        }
    }
}

// Internal evaluation result
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct TimingEval {
    offset: usize,         // sub-bit offset τ (0..23) that produced this
    corr_abs: f32,
    polarity: i32,
    sync_bits: i32,
    sync_shift: i32,
    sync_rotation: i32,
    format_hint: i32,
    soft_bits: Vec<f32>,   // the 258 soft bits at this timing/alignment
    end_sample: u64,       // absolute sample index of last sample of this packet
}

impl MatrixSyncExtractor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compatibility shim: accepts soft bits but this extractor needs phase samples.
    /// For the new pipeline, call `push_phase()` instead.
    pub fn push_soft_bits(&mut self, _bits: &[f32]) -> Vec<PacketCandidate> {
        // This should not be called in the new pipeline.
        // Return empty — the modem rx.rs will be updated to call push_phase().
        Vec::new()
    }

    /// Push unwrapped phase samples. Returns any packet candidates found.
    pub fn push_phase(&mut self, phases: &[f32]) -> Vec<PacketCandidate> {
        let mut out = Vec::new();

        for &p in phases {
            self.phase_buf.push_back(p);
            self.total_samples += 1;

            // Trim buffer: keep at most 3 packets worth
            let max_buf = 3 * PACKET_BITS * SAMPLES_PER_BIT;
            while self.phase_buf.len() > max_buf {
                self.phase_buf.pop_front();
                self.buf_start_index += 1;
            }
        }

        // Check if we have enough data and it's time to evaluate
        if self.total_samples >= self.next_eval_at
            && self.phase_buf.len() >= TWO_PKT_SAMPLES
        {
            self.evaluate(&mut out);
            self.next_eval_at = self.total_samples + EVAL_STRIDE_SAMPLES as u64;
        }

        out
    }

    /// DJ5HG Section 8.2.2 — Matrix bit-synchronization.
    ///
    /// The spec describes reshaping a 2-packet sample window into a matrix
    /// where each column represents a different sub-bit timing offset. 
    /// Correlating each column with the sync pattern simultaneously finds
    /// the best timing alignment.
    ///
    /// Implementation:
    ///   1. Demod soft bits at ALL 24 sub-bit offsets in one batch
    ///   2. For each offset, slide 258-bit window across all positions
    ///   3. Proper deinterleaved sync correlation (F1 + F2)
    ///   4. Track best F1 and F2 across all offsets and positions
    ///
    /// Optimised: soft bits for all 24 offsets precomputed once.
    fn evaluate(&mut self, out: &mut Vec<PacketCandidate>) {
        let buf_len = self.phase_buf.len();
        if buf_len < TWO_PKT_SAMPLES {
            return;
        }

        let window_start = buf_len - TWO_PKT_SAMPLES;
        let window_start_abs = self.buf_start_index + window_start as u64;

        let mut pat0 = [0.0f32; SYNC_BITS];
        for i in 0..SYNC_BITS {
            pat0[i] = if SYNC_PATTERN_HD43[i] == 1 { 1.0 } else { -1.0 };
        }
        let pat_label_14 = rotate_left_43(&pat0, 29);
        let pat_label_29 = rotate_left_43(&pat0, 14);

        // ── Precompute sync-bit positions for fast lookup ──
        // Instead of calling deinterleave_sync_soft_f1/f2 at every position
        // (which iterates 258 entries), precompute the position lists once.
        let mut f1_sync_positions: Vec<(usize, usize)> = Vec::with_capacity(SYNC_BITS);
        for (pos, &(typ, idx)) in fmt1::FORMAT1_TABLE.iter().enumerate() {
            if typ == b'S' {
                let logical = (idx - 1) as usize;
                if logical < SYNC_BITS {
                    f1_sync_positions.push((pos, logical));
                }
            }
        }
        let mut f2_sync_positions: Vec<(usize, usize)> = Vec::with_capacity(SYNC_BITS);
        for (pos, &(typ, idx)) in fmt2::FORMAT2_TABLE.iter().enumerate() {
            if typ.eq_ignore_ascii_case("s") {
                let logical = (idx - 1) as usize;
                if logical < SYNC_BITS {
                    f2_sync_positions.push((pos, logical));
                }
            }
        }

        let mut f1_best: Option<TimingEval> = None;
        let mut f2_best: Option<TimingEval> = None;

        // ── Exhaustive search over all 24 τ offsets ──
        for tau in 0..SAMPLES_PER_BIT {
            let soft = self.demod_at_offset(window_start, tau, 0);
            if soft.len() < PACKET_BITS {
                continue;
            }
            let max_start = soft.len() - PACKET_BITS;

            for start in 0..=max_start {
                // ── Format-1 sync (inline deinterleave + correlate) ──
                let mut corr_f1 = 0.0f32;
                let mut norm_sq_f1 = 0.0f32;
                for &(pos, logical) in &f1_sync_positions {
                    let soft_val = soft[start + pos];
                    corr_f1 += soft_val * pat0[logical];
                    norm_sq_f1 += soft_val * soft_val;
                }
                // Proper correlation: dot / (norm_soft * norm_pattern)
                // norm_pattern = sqrt(43) since pattern is all ±1
                let norm_f1 = norm_sq_f1.sqrt();
                let norm_pattern = (SYNC_BITS as f32).sqrt();
                let ca_f1 = if norm_f1 > 0.0 {
                    corr_f1.abs() / (norm_f1 * norm_pattern)
                } else {
                    0.0
                };

                if ca_f1 > f1_best.as_ref().map(|b| b.corr_abs).unwrap_or(self.corr_threshold) {
                    let pol = if corr_f1 >= 0.0 { 1 } else { -1 };
                    let end_sample = window_start_abs
                        + tau as u64
                        + ((start + PACKET_BITS) as u64) * SAMPLES_PER_BIT as u64;
                    f1_best = Some(TimingEval {
                        offset: tau,
                        corr_abs: ca_f1,
                        polarity: pol,
                        sync_bits: {
                            let mut count = 0i32;
                            for &(pos, logical) in &f1_sync_positions {
                                let bit_sign = if soft[start + pos] * pol as f32 >= 0.0 { 1.0 } else { -1.0 };
                                if bit_sign == pat0[logical] { count += 1; }
                            }
                            count
                        },
                        sync_shift: 0,
                        sync_rotation: 0,
                        format_hint: 1,
                        soft_bits: soft[start..start + PACKET_BITS].to_vec(),
                        end_sample,
                    });
                }

                // ── Format-2 sync (inline, 3 shifts) ──
                // Compute F2 sync deinterleave once
                let mut sync_f2 = [0.0f32; SYNC_BITS];
                for &(pos, logical) in &f2_sync_positions {
                    sync_f2[logical] = soft[start + pos];
                }

                for (label, rotation, pat) in [
                    (0i32, 0i32, &pat0),
                    (14, 29, &pat_label_14),
                    (29, 14, &pat_label_29),
                ] {
                    let mut corr = 0.0f32;
                    let mut norm_sq = 0.0f32;
                    for i in 0..SYNC_BITS {
                        let soft_val = sync_f2[i];
                        corr += soft_val * pat[i];
                        norm_sq += soft_val * soft_val;
                    }
                    // Proper correlation: dot / (norm_soft * norm_pattern)
                    let norm_soft = norm_sq.sqrt();
                    let norm_pattern = (SYNC_BITS as f32).sqrt();
                    let ca = if norm_soft > 0.0 {
                        corr.abs() / (norm_soft * norm_pattern)
                    } else {
                        0.0
                    };

                    if ca > f2_best.as_ref().map(|b| b.corr_abs).unwrap_or(self.corr_threshold) {
                        let pol = if corr >= 0.0 { 1 } else { -1 };
                        let end_sample = window_start_abs
                            + tau as u64
                            + ((start + PACKET_BITS) as u64) * SAMPLES_PER_BIT as u64;
                        f2_best = Some(TimingEval {
                            offset: tau,
                            corr_abs: ca,
                            polarity: pol,
                            sync_bits: count_sync_matches(&sync_f2, pat, pol),
                            sync_shift: label,
                            sync_rotation: rotation,
                            format_hint: 2,
                            soft_bits: soft[start..start + PACKET_BITS].to_vec(),
                            end_sample,
                        });
                    }
                }
            }
        }

        // Emit best F1 candidate (with separation check)
        if let Some(eval) = f1_best {
            let sep_ok = self.f1_last_peak_sample
                .map(|last| eval.end_sample > last + EVAL_STRIDE_SAMPLES as u64 / 2)
                .unwrap_or(true);
            if sep_ok {
                self.f1_last_peak_sample = Some(eval.end_sample);
                out.push(self.to_candidate(eval));
            }
        }

        // Emit best F2 candidate (with separation check)
        if let Some(eval) = f2_best {
            let sep_ok = self.f2_last_peak_sample
                .map(|last| eval.end_sample > last + EVAL_STRIDE_SAMPLES as u64 / 2)
                .unwrap_or(true);
            if sep_ok {
                self.f2_last_peak_sample = Some(eval.end_sample);
                out.push(self.to_candidate(eval));
            }
        }
    }

    /// Demodulate phase samples to soft bits at a given sub-bit offset.
    ///
    /// Measures the phase slope within each bit by comparing the phase at
    /// 1/4 of the way through the bit to 3/4 of the way through. With the
    /// boxcar matched filter, these regions are well-settled and give near-full
    /// amplitude soft bits. This preserves the original bit-to-sample alignment
    /// that the interleave maps expect.
    fn demod_at_offset(&self, window_start: usize, tau: usize, _avg_w: usize) -> Vec<f32> {
        let buf_len = self.phase_buf.len();
        let first_sample = window_start + tau;

        if first_sample + SAMPLES_PER_BIT > buf_len {
            return Vec::new();
        }
        let n_complete_bits = (buf_len - first_sample) / SAMPLES_PER_BIT;
        if n_complete_bits < 1 {
            return Vec::new();
        }

        // Sample positions within each bit (24 samples):
        //   Quarter point: sample 6  (1/4 of bit — filter settled from previous transition)
        //   Three-quarter: sample 18 (3/4 of bit — before next transition starts)
        // Average 3 samples around each point for slight smoothing.
        let q1_centre = SAMPLES_PER_BIT / 4;     // 6
        let q3_centre = 3 * SAMPLES_PER_BIT / 4; // 18
        let avg_r = 1; // ±1 sample = 3 samples total

        let mut soft = Vec::with_capacity(n_complete_bits);

        for b in 0..n_complete_bits {
            let bit_start = first_sample + b * SAMPLES_PER_BIT;

            let p1 = bit_start + q1_centre;
            let p2 = bit_start + q3_centre;

            if p2 + avg_r >= buf_len {
                break;
            }

            let origin = self.phase_buf[p1];

            // Phase at quarter point
            let mut sum_a = 0.0f32;
            let mut n_a = 0;
            for k in p1.saturating_sub(avg_r)..=(p1 + avg_r).min(buf_len - 1) {
                sum_a += self.phase_buf[k] - origin;
                n_a += 1;
            }
            let phase_a = sum_a / n_a as f32;

            // Phase at three-quarter point
            let mut sum_b = 0.0f32;
            let mut n_b = 0;
            for k in p2.saturating_sub(avg_r)..=(p2 + avg_r).min(buf_len - 1) {
                sum_b += self.phase_buf[k] - origin;
                n_b += 1;
            }
            let phase_b = sum_b / n_b as f32;

            // Scale: the phase change over half a bit should be ±π/4 for BPSK
            // (full bit = ±π/2, half bit = ±π/4)
            let delta = phase_b - phase_a;
            let soft_bit = (delta / (PI / 4.0)).clamp(-4.0, 4.0);
            soft.push(soft_bit);
        }

        soft
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

// Keep the old name as an alias for API compatibility
pub type ContinuousPacketExtractor = MatrixSyncExtractor;

// ============================================================================
// Windowed sync search (used by batch decoders)
// ============================================================================

pub fn find_sync_soft_format1(soft_bits: &[f32]) -> RxSync {
    let mut best = RxSync::default();
    let mut best_corr_abs = 0.0f32;

    if soft_bits.len() < PACKET_BITS {
        return best;
    }

    let max_start = soft_bits.len().saturating_sub(PACKET_BITS);

    let mut pat0 = [0.0f32; SYNC_BITS];
    for i in 0..SYNC_BITS {
        pat0[i] = if SYNC_PATTERN_HD43[i] == 1 { 1.0 } else { -1.0 };
    }

    for pos in 0..=max_start {
        let window = &soft_bits[pos..pos + PACKET_BITS];
        let sync_soft = deinterleave_sync_soft_f1(window);
        let corr = dot43(&sync_soft, &pat0);
        let corr_abs = corr.abs();

        if corr_abs > best_corr_abs {
            best_corr_abs = corr_abs;
            best.found = true;
            best.position = pos as i32;
            best.correlation = corr_abs / SYNC_BITS as f32;
            best.polarity = if corr >= 0.0 { 1 } else { -1 };
            best.sync_bits = count_sync_matches(&sync_soft, &pat0, best.polarity);
            best.sync_shift = 0;
            best.sync_rotation = 0;
            best.format_hint = 1;
        }
    }

    if best.correlation < 0.25 {
        best.found = false;
    }

    best
}

pub fn find_sync_soft(soft_bits: &[f32]) -> RxSync {
    let mut best = RxSync::default();
    let mut best_corr_abs = 0.0f32;

    if soft_bits.len() < PACKET_BITS {
        return best;
    }

    let max_start = soft_bits.len().saturating_sub(PACKET_BITS);

    let mut pat0 = [0.0f32; SYNC_BITS];
    for i in 0..SYNC_BITS {
        pat0[i] = if SYNC_PATTERN_HD43[i] == 1 { 1.0 } else { -1.0 };
    }

    let pat_label_0 = pat0;
    let pat_label_14 = rotate_left_43(&pat0, 29);
    let pat_label_29 = rotate_left_43(&pat0, 14);

    for pos in 0..=max_start {
        let window = &soft_bits[pos..pos + PACKET_BITS];

        // Format-1
        {
            let sync_soft = deinterleave_sync_soft_f1(window);
            let corr = dot43(&sync_soft, &pat_label_0);
            let corr_abs = corr.abs();
            if corr_abs > best_corr_abs {
                best_corr_abs = corr_abs;
                best.found = true;
                best.position = pos as i32;
                best.correlation = corr_abs / SYNC_BITS as f32;
                best.polarity = if corr >= 0.0 { 1 } else { -1 };
                best.sync_bits = count_sync_matches(&sync_soft, &pat_label_0, best.polarity);
                best.sync_shift = 0;
                best.sync_rotation = 0;
                best.format_hint = 1;
            }
        }

        // Format-2 (3 shifts)
        {
            let sync_soft = deinterleave_sync_soft_f2(window);
            for (label, rotation, pat) in [
                (0i32, 0i32, &pat_label_0),
                (14, 29, &pat_label_14),
                (29, 14, &pat_label_29),
            ] {
                let corr = dot43(&sync_soft, pat);
                let corr_abs = corr.abs();
                if corr_abs > best_corr_abs {
                    best_corr_abs = corr_abs;
                    best.found = true;
                    best.position = pos as i32;
                    best.correlation = corr_abs / SYNC_BITS as f32;
                    best.polarity = if corr >= 0.0 { 1 } else { -1 };
                    best.sync_bits = count_sync_matches(&sync_soft, pat, best.polarity);
                    best.sync_shift = label;
                    best.sync_rotation = rotation;
                    best.format_hint = 2;
                }
            }
        }
    }

    if best.correlation < 0.25 {
        best.found = false;
    }

    best
}

// ============================================================================
// Packet extraction helpers
// ============================================================================

pub fn extract_packet_soft_format1(soft_bits: &[f32], sync: &RxSync) -> Option<Vec<f32>> {
    if !sync.found {
        return None;
    }
    let start = sync.position as usize;
    let end = start + PACKET_BITS;
    if end > soft_bits.len() {
        return None;
    }

    let pol = sync.polarity as f32;
    let pkt: Vec<f32> = soft_bits[start..end].iter().map(|&b| b * pol).collect();
    Some(pkt)
}

// ============================================================================
// Utility functions
// ============================================================================

fn dot43(a: &[f32; 43], b: &[f32; 43]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..43 {
        s += a[i] * b[i];
    }
    s
}

fn rotate_left_43(pat: &[f32; 43], k: usize) -> [f32; 43] {
    let mut out = [0.0f32; 43];
    for i in 0..43 {
        out[i] = pat[(i + k) % 43];
    }
    out
}

fn count_sync_matches(sync_soft: &[f32; 43], pat: &[f32; 43], polarity: i32) -> i32 {
    let mut count = 0i32;
    let pol = polarity as f32;
    for i in 0..43 {
        let bit_val = sync_soft[i] * pol;
        let expected = pat[i];
        if (bit_val > 0.0 && expected > 0.0) || (bit_val < 0.0 && expected < 0.0) {
            count += 1;
        }
    }
    count
}

fn deinterleave_sync_soft_f1(window: &[f32]) -> [f32; 43] {
    let mut out = [0.0f32; 43];

    for (pos, &(typ, idx)) in fmt1::FORMAT1_TABLE.iter().enumerate() {
        if typ == b'S' {
            let logical = (idx - 1) as usize;
            if logical < 43 && pos < window.len() {
                out[logical] = window[pos];
            }
        }
    }

    out
}

fn deinterleave_sync_soft_f2(window: &[f32]) -> [f32; 43] {
    let mut out = [0.0f32; 43];

    for (pos, &(typ, idx)) in fmt2::FORMAT2_TABLE.iter().enumerate() {
        if typ.eq_ignore_ascii_case("s") {
            let logical = (idx - 1) as usize;
            if logical < 43 && pos < window.len() {
                out[logical] = window[pos];
            }
        }
    }

    out
}

fn unwrap_phases(phases: &[f32]) -> Vec<f32> {
    if phases.is_empty() {
        return Vec::new();
    }

    let mut unwrapped = vec![0.0f32; phases.len()];
    let mut offset = 0.0f32;

    unwrapped[0] = phases[0];

    for i in 1..phases.len() {
        let diff = phases[i] - phases[i - 1];
        if diff > PI {
            offset -= 2.0 * PI;
        } else if diff < -PI {
            offset += 2.0 * PI;
        }
        unwrapped[i] = phases[i] + offset;
    }

    unwrapped
}
