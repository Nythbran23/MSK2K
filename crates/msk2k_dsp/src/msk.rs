// crates/msk2k_dsp/src/msk.rs
//
// MSK2K modem pieces.
//
// TX: modulate_48k(bits) -> Vec<f32>
// RX (stubs for now): demodulate_48k(audio)->baseband, find_sync(baseband)->RxSync,
//                     extract_packet_soft(baseband, sync)->Option<Vec<f32>>
//
// The TX modulator below is intended to match Python MSK2KModulator in msk2k_complete.py:
// - sample_rate = 48000
// - bit_rate    = 2000  => samples_per_bit = 24
// - carrier     = 1500 Hz
// - per-bit phase step = ±pi/2
// - per-sample phase uses (j + 0.5) center sampling
// - output normalized by peak abs

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RxSync {
    pub found: bool,
    pub correlation: f32,
    pub position: i32,
    pub sync_bits: i32,
    pub polarity: i32,
    pub sync_shift: i32,
    pub sync_rotation: i32,
}

// =====================
// TX: MSK modulator
// =====================

pub fn modulate_48k(bits: &[i32]) -> Vec<f32> {
    const SAMPLE_RATE: f32 = 48_000.0;
    const BIT_RATE: f32 = 2_000.0;
    const CARRIER_HZ: f32 = 1_350.0;

    let samples_per_bit = (SAMPLE_RATE / BIT_RATE) as usize; // 24
    if bits.is_empty() || samples_per_bit == 0 {
        return Vec::new();
    }

    let n_samples = bits.len() * samples_per_bit;
    let mut out = vec![0.0f32; n_samples];

    let two_pi = 2.0f32 * std::f32::consts::PI;
    let half = 0.5f32;

    let mut current_phase: f32 = 0.0;

    for (i, &b) in bits.iter().enumerate() {
        // Python: delta_phase = +pi/2 if bit else -pi/2
        let bit_is_one = b != 0;
        let delta_phase = if bit_is_one {
            std::f32::consts::PI * 0.5
        } else {
            -std::f32::consts::PI * 0.5
        };

        let phase_rate = delta_phase / (samples_per_bit as f32);

        for j in 0..samples_per_bit {
            let idx = i * samples_per_bit + j;

            // Python: phase[idx] = current_phase + phase_rate * (j + 0.5)
            let ph = current_phase + phase_rate * ((j as f32) + half);

            let t = (idx as f32) / SAMPLE_RATE;
            let carrier = two_pi * CARRIER_HZ * t;

            out[idx] = (carrier + ph).cos();
        }

        current_phase += delta_phase;
    }

    // Normalize to ±1 like Python generate_packet_audio()
    let mut peak = 0.0f32;
    for &v in &out {
        let a = v.abs();
        if a > peak {
            peak = a;
        }
    }
    if peak > 0.0 {
        for v in &mut out {
            *v /= peak;
        }
    }

    out
}

// =====================
// RX frontend (stubs)
// =====================

/// Demodulate 48 kHz MSK audio into a "soft" baseband stream.
/// In your Python goldens this is currently length 258 (for the test vector),
/// but in real use it could be longer.
pub fn demodulate_48k(_audio: &[f32]) -> Vec<f32> {
    // TODO: implement MSK demod path to match Python:
    // - matched filter / integrate-and-dump at 24 samples/bit
    // - produce soft values per bit
    Vec::new()
}

/// Find sync in the demodulated baseband soft stream.
pub fn find_sync(_baseband: &[f32]) -> RxSync {
    // TODO: implement sync correlation, polarity, rotation, etc.
    RxSync {
        found: false,
        correlation: 0.0,
        position: 0,
        sync_bits: 0,
        polarity: 1,
        sync_shift: 0,
        sync_rotation: 0,
    }
}

/// Extract the 258 "packet soft bits" (or whatever fixed size your Python uses)
/// given the baseband stream and sync result.
pub fn extract_packet_soft(_baseband: &[f32], _sync: &RxSync) -> Option<Vec<f32>> {
    // TODO: implement extraction windowing/rotation/polarity handling
    None
}
