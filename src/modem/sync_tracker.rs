//! Sync Tracker V4 - DJ5HG PSK2K with soft bit output
//!
//! Key features:
//! 1. Proper BPSK demodulation (carrier removal, phase detection)
//! 2. HD43 sync correlation on demodulated soft bits
//! 3. Three sync pattern rotations for partial packets (shifts 0, 14, 29)
//! 4. **Outputs packet soft bits directly** - no need to re-demodulate
//!
//! Reference: "The PSK2k V5 Codes" by Klaus von der Heide, DJ5HG

use std::collections::VecDeque;
use std::f32::consts::PI;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Constants
// ============================================================================

const SAMPLE_RATE: f32 = 48_000.0;
const CARRIER_HZ: f32 = 1496.1;
const BIT_RATE: f32 = 2000.0;
const SAMPLES_PER_BIT: usize = (SAMPLE_RATE / BIT_RATE) as usize; // 24
const PACKET_BITS: usize = 258;
const SYNC_BITS: usize = 43;

/// The 43-bit Hadamard sync pattern (DJ5HG spec section 3)
pub const SYNC_PATTERN_43: [i8; 43] = [
    0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0,
    0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0
];

pub const SYNC_SHIFTS: [u8; 3] = [0, 14, 29];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatHint {
    Format1,
    Format2,
    Unknown,
}

/// A detected sync peak with packet data
#[derive(Debug, Clone)]
pub struct SyncPeak {
    pub t_end_samples: u64,
    pub corr: f32,
    pub polarity: Option<i8>,
    pub format_hint: FormatHint,
    pub utc_ms: i64,
    pub slot_id: i64,
    pub sync_shift: u8,
    pub bit_position: usize,
    pub sync_bits_matched: i32,
    /// The 258 packet soft bits (polarity-corrected), ready for FEC decode
    pub packet_soft_bits: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct DecodeHint {
    pub t_end_samples: u64,
    pub t_end_offset: usize,
    pub polarity: Option<i8>,
    pub format_hint: FormatHint,
    pub slot_id: i64,
    pub sync_shift: u8,
}

#[derive(Debug, Clone)]
pub struct SyncTrackerConfig {
    pub sample_rate: u32,
    pub slot_period_ms: u32,
    pub corr_threshold: f32,
    pub min_sync_bits: i32,
}

impl Default for SyncTrackerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            slot_period_ms: 15000,
            corr_threshold: 0.30,
            min_sync_bits: 38,
        }
    }
}

// ============================================================================
// BPSK Demodulator
// ============================================================================

struct BpskDemodState {
    sample_index: u64,
    prev_i: f32,
    prev_q: f32,
    last_phase: f32,
    wrap_offset: f32,
    bit_buf: Vec<f32>,
}

impl Default for BpskDemodState {
    fn default() -> Self {
        Self {
            sample_index: 0,
            prev_i: 1.0,
            prev_q: 0.0,
            last_phase: 0.0,
            wrap_offset: 0.0,
            bit_buf: Vec::with_capacity(SAMPLES_PER_BIT),
        }
    }
}

impl BpskDemodState {
    fn demodulate(&mut self, audio: &[f32]) -> Vec<f32> {
        let mut soft_bits = Vec::with_capacity(audio.len() / SAMPLES_PER_BIT + 1);
        const ALPHA: f32 = 0.2;
        let avg_window = SAMPLES_PER_BIT / 4; // 6 samples

        for &sample in audio {
            let t = self.sample_index as f32 / SAMPLE_RATE;
            let lo_i = (2.0 * PI * CARRIER_HZ * t).cos();
            let lo_q = -(2.0 * PI * CARRIER_HZ * t).sin();

            let i = sample * lo_i;
            let q = sample * lo_q;

            self.prev_i = (1.0 - ALPHA) * self.prev_i + ALPHA * i;
            self.prev_q = (1.0 - ALPHA) * self.prev_q + ALPHA * q;

            let phase = self.prev_q.atan2(self.prev_i);
            let diff = phase - self.last_phase;
            if diff > PI {
                self.wrap_offset -= 2.0 * PI;
            } else if diff < -PI {
                self.wrap_offset += 2.0 * PI;
            }
            let unwrapped = phase + self.wrap_offset;
            self.last_phase = phase;

            self.bit_buf.push(unwrapped);
            self.sample_index += 1;

            if self.bit_buf.len() >= SAMPLES_PER_BIT {
                let phase_start: f32 = self.bit_buf[0..avg_window].iter().sum::<f32>()
                    / avg_window as f32;
                let phase_end: f32 = self.bit_buf[SAMPLES_PER_BIT - avg_window..SAMPLES_PER_BIT]
                    .iter()
                    .sum::<f32>()
                    / avg_window as f32;

                let delta_phi = phase_end - phase_start;
                let soft_bit = (delta_phi / (PI / 2.0)).clamp(-4.0, 4.0);
                soft_bits.push(soft_bit);
                self.bit_buf.clear();
            }
        }
        soft_bits
    }

    fn reset(&mut self) {
        self.sample_index = 0;
        self.prev_i = 1.0;
        self.prev_q = 0.0;
        self.last_phase = 0.0;
        self.wrap_offset = 0.0;
        self.bit_buf.clear();
    }
}

// ============================================================================
// Sync Tracker
// ============================================================================

pub struct SyncTracker {
    config: SyncTrackerConfig,
    demod: BpskDemodState,
    soft_bit_ring: VecDeque<f32>,
    window_bits: usize,
    audio_sample_count: u64,
    soft_bit_count: u64,
    bits_since_last_analysis: usize,
    sync_patterns_bipolar: [[f32; SYNC_BITS]; 3],
    last_analysis_ms: i64,
    min_analysis_interval_ms: i64,
    analyses_count: usize,
    peaks_found: usize,
}

impl SyncTracker {
    pub fn new(config: SyncTrackerConfig) -> Self {
        let window_bits = PACKET_BITS * 2;
        let sync_patterns_bipolar = Self::build_sync_patterns();

        log::info!(
            "SyncTracker V4: window={} bits, corr_thr={:.2}, sync_bits_min={}",
            window_bits,
            config.corr_threshold,
            config.min_sync_bits
        );

        Self {
            config,
            demod: BpskDemodState::default(),
            soft_bit_ring: VecDeque::with_capacity(window_bits + 100),
            window_bits,
            audio_sample_count: 0,
            soft_bit_count: 0,
            bits_since_last_analysis: 0,
            sync_patterns_bipolar,
            last_analysis_ms: 0,
            min_analysis_interval_ms: 129,
            analyses_count: 0,
            peaks_found: 0,
        }
    }

    fn build_sync_patterns() -> [[f32; SYNC_BITS]; 3] {
        let base: Vec<f32> = SYNC_PATTERN_43
            .iter()
            .map(|&b| if b == 1 { 1.0 } else { -1.0 })
            .collect();

        let mut patterns = [[0.0f32; SYNC_BITS]; 3];
        for i in 0..SYNC_BITS {
            patterns[0][i] = base[i];
            patterns[1][i] = base[(i + 14) % SYNC_BITS];
            patterns[2][i] = base[(i + 29) % SYNC_BITS];
        }
        patterns
    }

    pub fn update(&mut self, audio: &[f32], slot_period_ms: u32) -> Vec<SyncPeak> {
        self.audio_sample_count += audio.len() as u64;

        let rms: f32 =
            (audio.iter().map(|x| x * x).sum::<f32>() / audio.len().max(1) as f32).sqrt();
        if rms < 0.0005 {
            return Vec::new();
        }

        let soft_bits = self.demod.demodulate(audio);

        for &bit in &soft_bits {
            self.soft_bit_ring.push_back(bit);
            self.bits_since_last_analysis += 1;
            self.soft_bit_count += 1;

            while self.soft_bit_ring.len() > self.window_bits + PACKET_BITS {
                self.soft_bit_ring.pop_front();
            }
        }

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let enough_bits = self.soft_bit_ring.len() >= self.window_bits;
        let enough_time = now_ms - self.last_analysis_ms >= self.min_analysis_interval_ms;
        let enough_new = self.bits_since_last_analysis >= PACKET_BITS;

        if enough_bits && enough_time && enough_new {
            self.last_analysis_ms = now_ms;
            self.bits_since_last_analysis = 0;
            self.analyses_count += 1;

            if let Some(peak) = self.find_global_max(slot_period_ms) {
                self.peaks_found += 1;

                if self.analyses_count % 10 == 0 {
                    log::info!(
                        "[SYNC-V4] analyses={} peaks={} ({}%), corr={:.3}, bits={}/43",
                        self.analyses_count,
                        self.peaks_found,
                        self.peaks_found * 100 / self.analyses_count.max(1),
                        peak.corr,
                        peak.sync_bits_matched
                    );
                }

                return vec![peak];
            }
        }

        Vec::new()
    }

    fn find_global_max(&self, slot_period_ms: u32) -> Option<SyncPeak> {
        if self.soft_bit_ring.len() < self.window_bits {
            return None;
        }

        let bits: Vec<f32> = self.soft_bit_ring.iter().copied().collect();
        let n = bits.len();

        let mut best_corr = 0.0f32;
        let mut best_pos = 0usize;
        let mut best_polarity = 1i8;
        let mut best_shift_idx = 0usize;
        let mut best_sync_bits = 0i32;

        // For each candidate packet end position
        for pos in PACKET_BITS..=n {
            let packet = &bits[pos - PACKET_BITS..pos];

            // Extract sync bits at every 6th position
            let mut sync_soft = [0.0f32; SYNC_BITS];
            for k in 0..SYNC_BITS {
                sync_soft[k] = packet[k * 6];
            }

            // DC removal and normalization
            let mean: f32 = sync_soft.iter().sum::<f32>() / SYNC_BITS as f32;
            let centered: Vec<f32> = sync_soft.iter().map(|x| x - mean).collect();
            let window_energy: f32 = centered.iter().map(|x| x * x).sum::<f32>();
            let window_norm = window_energy.sqrt().max(1e-12);
            let pattern_norm = (SYNC_BITS as f32).sqrt();

            // Try all 3 sync rotations
            for (shift_idx, pattern) in self.sync_patterns_bipolar.iter().enumerate() {
                let dot: f32 = centered
                    .iter()
                    .zip(pattern.iter())
                    .map(|(&w, &p)| w * p)
                    .sum();

                let corr = dot / (window_norm * pattern_norm);
                let corr_abs = corr.abs();

                if corr_abs > best_corr {
                    best_corr = corr_abs;
                    best_pos = pos;
                    best_polarity = if corr >= 0.0 { 1 } else { -1 };
                    best_shift_idx = shift_idx;

                    // Count matching sync bits
                    let pol = best_polarity as f32;
                    best_sync_bits = sync_soft
                        .iter()
                        .zip(pattern.iter())
                        .filter(|(&soft, &pat)| {
                            let hard = if soft * pol >= 0.0 { 1.0 } else { -1.0 };
                            (hard - pat).abs() < 0.5
                        })
                        .count() as i32;
                }
            }
        }

        // Gate on correlation and sync bits
        if best_corr < self.config.corr_threshold {
            return None;
        }
        if best_sync_bits < self.config.min_sync_bits {
            return None;
        }

        // Extract the packet soft bits (polarity-corrected)
        let packet_start = best_pos - PACKET_BITS;
        let pol = best_polarity as f32;
        let packet_soft_bits: Vec<f32> = bits[packet_start..best_pos]
            .iter()
            .map(|&b| b * pol)
            .collect();

        let utc_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let slot_id = utc_ms / slot_period_ms as i64;

        let bits_from_end = n - best_pos;
        let samples_from_end = bits_from_end * SAMPLES_PER_BIT;
        let t_end_samples = self.audio_sample_count.saturating_sub(samples_from_end as u64);

        log::info!(
            "[SYNC-V4] PEAK: pos={}/{} corr={:.3} pol={} shift={} bits={}/43",
            best_pos,
            n,
            best_corr,
            best_polarity,
            SYNC_SHIFTS[best_shift_idx],
            best_sync_bits
        );

        Some(SyncPeak {
            t_end_samples,
            corr: best_corr,
            polarity: Some(best_polarity),
            format_hint: FormatHint::Unknown,
            utc_ms,
            slot_id,
            sync_shift: SYNC_SHIFTS[best_shift_idx],
            bit_position: best_pos,
            sync_bits_matched: best_sync_bits,
            packet_soft_bits,
        })
    }

    pub fn extraction_params(&self) -> (usize, usize) {
        // No longer needed for audio extraction - we output soft bits directly
        // But keep for compatibility
        let pre = (PACKET_BITS + 1) * SAMPLES_PER_BIT;
        let post = 0;
        (pre, post)
    }

    pub fn reset(&mut self) {
        self.demod.reset();
        self.soft_bit_ring.clear();
        self.audio_sample_count = 0;
        self.soft_bit_count = 0;
        self.bits_since_last_analysis = 0;
        self.last_analysis_ms = 0;
        self.analyses_count = 0;
        self.peaks_found = 0;
        log::info!("SyncTracker V4 reset");
    }
}
