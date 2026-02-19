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

use num_complex::Complex32;
use rustfft::FftPlanner;

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
const CARRIER_HZ: f32 = 1496.1; // DJ5HG spec: 193*2000/258 = 1496.1 Hz
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
/// Buffers raw audio samples and periodically runs the full DJ5HG receiver
/// pipeline per Section 8: carrier recovery (SQ→FFT→CSY) followed by the
/// matrix bit-synchronisation timing search.
#[derive(Debug, Clone)]
pub struct MatrixSyncExtractor {
    /// Raw audio buffer — used for Hilbert-based frequency offset estimation.
    audio_buf: VecDeque<f32>,
    /// Unwrapped phase buffer — fed by the internal PhaseDemodState.
    phase_buf: VecDeque<f32>,
    /// Internal demodulator — converts audio to unwrapped phase in real time.
    demod: PhaseDemodState,
    /// Absolute sample index of the first element in both buffers.
    buf_start_index: u64,
    /// Absolute sample count (total samples pushed).
    total_samples: u64,
    /// Next absolute sample index at which to run an evaluation.
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
            audio_buf: VecDeque::with_capacity(TWO_PKT_SAMPLES + EVAL_STRIDE_SAMPLES),
            phase_buf: VecDeque::with_capacity(TWO_PKT_SAMPLES + EVAL_STRIDE_SAMPLES),
            demod: PhaseDemodState::new(),
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

    /// Compatibility shim — no-op.
    pub fn push_soft_bits(&mut self, _bits: &[f32]) -> Vec<PacketCandidate> {
        Vec::new()
    }

    /// Push raw audio samples. Returns any packet candidates found.
    ///
    /// This is the primary entry point. Raw audio is buffered here; the full
    /// DJ5HG pipeline (carrier recovery → matched filter → demod → sync search)
    /// runs inside evaluate() when enough data has accumulated.
    pub fn push_audio(&mut self, audio: &[f32]) -> Vec<PacketCandidate> {
        let mut out = Vec::new();

        // Demodulate the incoming chunk to unwrapped phase first.
        // Both buffers stay in lockstep — same sample count, same trim index.
        let phases = self.demod.push_audio(audio);

        let max_buf = 3 * PACKET_BITS * SAMPLES_PER_BIT;

        for (&s, &p) in audio.iter().zip(phases.iter()) {
            self.audio_buf.push_back(s);
            self.phase_buf.push_back(p);
            self.total_samples += 1;

            // Trim both buffers together so buf_start_index stays consistent.
            while self.audio_buf.len() > max_buf {
                self.audio_buf.pop_front();
                self.phase_buf.pop_front();
                self.buf_start_index += 1;
            }
        }

        if self.total_samples >= self.next_eval_at
            && self.phase_buf.len() >= TWO_PKT_SAMPLES
        {
            self.evaluate(&mut out);
            self.next_eval_at = self.total_samples + EVAL_STRIDE_SAMPLES as u64;
        }

        out
    }

    /// Backward-compat shim for callers that previously fed pre-demodulated
    /// phase samples. Those callers should be updated to call push_audio()
    /// with raw audio chunks, bypassing PhaseDemodState entirely.
    /// Until updated, this silently discards data and returns nothing.
    pub fn push_phase(&mut self, _phases: &[f32]) -> Vec<PacketCandidate> {
        Vec::new()
    }

    /// DJ5HG Section 8.2 — Carrier frequency detection and correction.
    ///
    /// Implements the SQ → FFT → CSY path from Figure 5:
    ///
    ///   1. Hilbert transform: real audio → complex analytic signal
    ///   2. Mix down by nominal carrier (1496.1 Hz) → complex baseband
    ///   3. Matched filter: boxcar lowpass at 1000 Hz on complex I+Q
    ///   4. Square signal: removes BPSK modulation, carrier appears at 2×offset
    ///   5. FFT: find peak → frequency offset = peak_freq/2, phase = peak_phase/2
    ///   6. Apply correction: multiply by exp(-j(2π·f_offset·t + φ))
    ///   7. Return real part of corrected signal + detected offset for logging
    ///
    /// If no clear peak is found (e.g. no signal yet), returns the signal
    /// demodulated at exactly 0 Hz offset — identical to previous behaviour.
    /// Hilbert-based MSK frequency offset estimator.
    ///
    /// For MSK, instantaneous frequency alternates between mark (2000 Hz) and
    /// space (1000 Hz) with a midpoint of 1500 Hz.  A frequency offset Δf shifts
    /// every sample's instantaneous frequency by Δf, so:
    ///
    ///   mean(f_inst) = 1500 + Δf   =>   Δf = mean(f_inst) - 1500
    ///
    /// Method:
    ///   1. FFT-based Hilbert → analytic signal
    ///   2. Instantaneous phase = atan2(imag, real)
    ///   3. Unwrap phase, differentiate → instantaneous frequency
    ///   4. Mean frequency − nominal midpoint = Δf
    ///
    /// Returns Δf in Hz.  Returns 0.0 if the audio is too short or silent.
    fn estimate_freq_offset(audio: &[f32]) -> f32 {
        let n = audio.len();
        if n < 64 {
            return 0.0;
        }

        // Quick silence check — avoid chasing noise on an empty band.
        let rms: f32 = (audio.iter().map(|&x| x * x).sum::<f32>() / n as f32).sqrt();
        if rms < 1e-5 {
            return 0.0;
        }

        // ── Step 1: FFT-based Hilbert transform → analytic signal ──
        let mut planner = FftPlanner::<f32>::new();
        let fft_fwd = planner.plan_fft_forward(n);
        let fft_inv = planner.plan_fft_inverse(n);

        let mut spec: Vec<Complex32> = audio
            .iter()
            .map(|&x| Complex32::new(x, 0.0))
            .collect();
        fft_fwd.process(&mut spec);

        // One-sided: zero DC imaginary, zero negatives, double positives.
        spec[0] = Complex32::new(spec[0].re, 0.0);
        if n % 2 == 0 {
            spec[n / 2] = Complex32::new(spec[n / 2].re, 0.0);
        }
        for k in 1..n / 2 {
            spec[k] *= 2.0;
        }
        for k in (n / 2 + 1)..n {
            spec[k] = Complex32::new(0.0, 0.0);
        }
        fft_inv.process(&mut spec);
        let inv_n = 1.0 / n as f32;
        // analytic[k].re = original sample, analytic[k].im = Hilbert transform
        let analytic: Vec<Complex32> = spec.iter().map(|c| c * inv_n).collect();

        // ── Step 2: Instantaneous phase ──
        let inst_phase: Vec<f32> = analytic.iter().map(|c| c.im.atan2(c.re)).collect();

        // ── Step 3: Unwrap and differentiate → instantaneous frequency ──
        // freq[k] = (φ[k] - φ[k-1]) * Fs / (2π)
        let mut freq_sum = 0.0f32;
        let mut freq_count = 0usize;
        let mut prev = inst_phase[0];

        for &phi in &inst_phase[1..] {
            let mut diff = phi - prev;
            // Wrap to [-π, π]
            while diff >  PI { diff -= 2.0 * PI; }
            while diff < -PI { diff += 2.0 * PI; }
            let f_inst = diff * SAMPLE_RATE / (2.0 * PI);
            // Only count samples in the expected MSK band (500–2500 Hz) to
            // suppress noise/silence samples from skewing the mean.
            if f_inst.abs() > 200.0 && f_inst.abs() < 3000.0 {
                freq_sum += f_inst;
                freq_count += 1;
            }
            prev = phi;
        }

        if freq_count < n / 4 {
            // Too few valid samples — probably silence or heavy noise.
            return 0.0;
        }

        let mean_freq = freq_sum / freq_count as f32;
        // MSK midpoint is 1500 Hz (average of 1000 and 2000).
        let delta_f = mean_freq - 1500.0;

        // Clamp to ±300 Hz — anything beyond that is noise, not offset.
        delta_f.clamp(-300.0, 300.0)
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

        // ── Hilbert-based frequency offset estimation ──
        // Estimate Δf from the raw audio window, then subtract the corresponding
        // linear phase ramp from the unwrapped phase window before demodulation.
        // This is correct for MSK: a frequency offset appears as a constant slope
        // added to every phase sample (drift rate = 2π·Δf / Fs rad/sample).
        let raw_window: Vec<f32> = self.audio_buf
            .range(window_start..window_start + TWO_PKT_SAMPLES)
            .copied()
            .collect();
        let delta_f = Self::estimate_freq_offset(&raw_window);

        if delta_f.abs() > 1.0 {
            log::debug!("[AFC] Frequency offset: {:.1} Hz", delta_f);
        }

        // Build frequency-corrected phase window.
        // phase_correction[i] = 2π · Δf · i / Fs
        // Subtract this from the stored unwrapped phase to remove the offset.
        let phase_slope = 2.0 * PI * delta_f / SAMPLE_RATE;
        let corrected_phases: Vec<f32> = self.phase_buf
            .range(window_start..window_start + TWO_PKT_SAMPLES)
            .enumerate()
            .map(|(i, &p)| p - phase_slope * i as f32)
            .collect();

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
            let soft = Self::demod_at_offset(&corrected_phases, tau);
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

    /// Demodulate frequency-corrected unwrapped phase samples to soft bits
    /// at a given sub-bit timing offset τ.
    ///
    /// Measures the phase SLOPE within each bit by comparing the value at
    /// the quarter-point (sample 6/24) to the three-quarter-point (sample 18/24).
    /// For MSK, phase ramps at +π/2 per bit for a '1' and −π/2 for a '0'.
    /// Over half a bit that slope produces ±π/4, so:
    ///   soft_bit = delta / (π/4)
    /// This is the original working formulation restored unchanged.
    fn demod_at_offset(phases: &[f32], tau: usize) -> Vec<f32> {
        let n = phases.len();
        let first = tau;

        if first + SAMPLES_PER_BIT > n {
            return Vec::new();
        }
        let n_bits = (n - first) / SAMPLES_PER_BIT;
        if n_bits < 1 {
            return Vec::new();
        }

        let q1 = SAMPLES_PER_BIT / 4;      // 6
        let q3 = 3 * SAMPLES_PER_BIT / 4;  // 18
        let avg_r = 1usize;                 // ±1 sample = 3-sample average

        let mut soft = Vec::with_capacity(n_bits);

        for b in 0..n_bits {
            let bit_start = first + b * SAMPLES_PER_BIT;
            let p1 = bit_start + q1;
            let p2 = bit_start + q3;

            if p2 + avg_r >= n {
                break;
            }

            let origin = phases[p1];

            // Phase slope at quarter-point (relative to origin to cancel DC)
            let mut sum_a = 0.0f32;
            let mut n_a = 0usize;
            for k in p1.saturating_sub(avg_r)..=(p1 + avg_r).min(n - 1) {
                sum_a += phases[k] - origin;
                n_a += 1;
            }
            let phase_a = sum_a / n_a as f32;

            // Phase slope at three-quarter-point
            let mut sum_b = 0.0f32;
            let mut n_b = 0usize;
            for k in p2.saturating_sub(avg_r)..=(p2 + avg_r).min(n - 1) {
                sum_b += phases[k] - origin;
                n_b += 1;
            }
            let phase_b = sum_b / n_b as f32;

            // Phase change over half a bit = ±π/4 for a full-amplitude MSK bit.
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
