//! Complete MSK2K Decoder
//!
//! Implements the full decode pipeline:
//! - Short signals: single-packet decode
//! - Long signals: fast-scan then accumulation

use crate::accumulator::{Accumulator, PhaseClustering};
use crate::callsign::CallsignCodec;
use crate::decode::{DecodedPacket, decode_packet_soft};
use crate::message::Message;
use crate::rx::{RxSync, demodulate_48k, extract_packet_soft, find_sync as rx_find_sync};

// MSK2K parameters
const SAMPLE_RATE: f32 = 48000.0;
const BIT_RATE: f32 = 2000.0;
const CARRIER_HZ: f32 = 1500.0;
const SAMPLES_PER_BIT: usize = 24;

// Thresholds
const LONG_AUDIO_THRESHOLD: f32 = 0.30;
const FAST_SCAN_CHUNK: f32 = 0.39;
const FAST_SCAN_STEP: f32 = 0.13;
const SYNC_THRESHOLD_FAST: usize = 30;
const SYNC_THRESHOLD_STRONG: usize = 40;
const CORR_THRESHOLD: f32 = 0.28;
const CANONICAL_THRESHOLD: f32 = 0.30;

/// Decode result with detailed metadata
#[derive(Debug, Clone)]
pub struct DecodeResult {
    pub success: bool,
    pub text: String,
    pub from_call: String,
    pub to_call: Option<String>,
    pub format: u8,
    pub sync_bits: String,
    pub sync_correlation: f32,
    pub method: String,
    pub error: Option<String>,
}

impl DecodeResult {
    pub fn failure(error: String, sync_bits: String, correlation: f32) -> Self {
        Self {
            success: false,
            text: String::new(),
            from_call: String::new(),
            to_call: None,
            format: 0,
            sync_bits,
            sync_correlation: correlation,
            method: String::new(),
            error: Some(error),
        }
    }
}

/// Internal helper for demodulation results
struct DemodResult {
    soft_bits: Vec<f32>,
    magnitude: Vec<f32>,
}

pub struct Decoder {
    codec: CallsignCodec,
    my_callsign: Option<String>,
    partner_callsign: Option<String>,
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            codec: CallsignCodec::new(),
            my_callsign: None,
            partner_callsign: None,
        }
    }

    pub fn set_my_callsign(&mut self, callsign: String) {
        self.my_callsign = Some(callsign);
    }

    pub fn set_partner_callsign(&mut self, callsign: String) {
        self.partner_callsign = Some(callsign);
    }

    pub fn decode(&self, audio: &[f32]) -> DecodeResult {
        let duration = audio.len() as f32 / SAMPLE_RATE;
        if duration >= LONG_AUDIO_THRESHOLD {
            self.decode_long_signal(audio)
        } else {
            self.decode_short_signal(audio)
        }
    }

    pub fn process_audio(&self, audio: &[f32]) -> Option<DecodedPacket> {
        if audio.is_empty() {
            return None;
        }
        let baseband = demodulate_48k(audio);
        let sync = rx_find_sync(&baseband);
        if sync.found {
            let soft_bits = extract_packet_soft(&baseband, &sync)?;
            return decode_packet_soft(&soft_bits, &sync);
        }
        None
    }

    fn decode_short_signal(&self, audio: &[f32]) -> DecodeResult {
        let demod = self.demodulate_internal(audio);
        let sync = rx_find_sync(&demod.soft_bits);

        if !sync.found || sync.sync_bits < SYNC_THRESHOLD_FAST as i32 {
            return DecodeResult::failure(
                format!("Sync failed ({}/43)", sync.sync_bits),
                format!("{}/43", sync.sync_bits),
                sync.correlation,
            );
        }

        if let Some(packet_soft) = extract_packet_soft(&demod.soft_bits, &sync) {
            if let Some(mut result) = self.try_decode_packet(&packet_soft) {
                result.sync_bits = format!("{}/43", sync.sync_bits);
                result.sync_correlation = sync.correlation;
                result.method = "single-packet".into();
                return result;
            }
        }

        DecodeResult::failure(
            "Decode failed".into(),
            format!("{}/43", sync.sync_bits),
            sync.correlation,
        )
    }

    fn decode_long_signal(&self, audio: &[f32]) -> DecodeResult {
        let demod = self.demodulate_internal(audio);
        let chunk_bits = (FAST_SCAN_CHUNK * BIT_RATE) as usize;
        let step_bits = (FAST_SCAN_STEP * BIT_RATE) as usize;

        let mut scan_start = 0;
        while scan_start + 258 <= demod.soft_bits.len() {
            let chunk_end = (scan_start + chunk_bits).min(demod.soft_bits.len());
            let chunk = &demod.soft_bits[scan_start..chunk_end];
            let sync = rx_find_sync(chunk);

            if sync.found && sync.sync_bits >= SYNC_THRESHOLD_FAST as i32 {
                if let Some(packet_soft) = extract_packet_soft(chunk, &sync) {
                    if let Some(mut result) = self.try_decode_packet(&packet_soft) {
                        result.method = "fast-scan".into();
                        return result;
                    }
                }
            }
            scan_start += step_bits;
        }
        self.decode_with_accumulation(&demod)
    }

    fn decode_with_accumulation(&self, demod: &DemodResult) -> DecodeResult {
        let mut acc = Accumulator::new();
        let baseband = &demod.soft_bits;
        let mut candidates = Vec::new();
        let mut clustering = PhaseClustering::new();
        let mut scan_pos = 0;

        while scan_pos + 258 <= baseband.len() {
            let chunk_size = 600.min(baseband.len() - scan_pos);
            let chunk = &baseband[scan_pos..scan_pos + chunk_size];
            let sync = rx_find_sync(chunk);

            if sync.found && sync.correlation >= CORR_THRESHOLD {
                let true_start =
                    (scan_pos as i32 + sync.position - (6 * sync.sync_rotation)) as i32;
                if true_start >= 0 && (true_start as usize + 258) <= baseband.len() {
                    let true_start_u = true_start as usize;
                    let packet_soft: Vec<f32> = baseband[true_start_u..true_start_u + 258]
                        .iter()
                        .map(|&x| x * sync.polarity as f32)
                        .collect();

                    let phase = (true_start_u % 258) as i32;
                    let weight = sync.correlation * sync.correlation;
                    clustering.add_candidate(phase, candidates.len(), weight);
                    candidates.push(packet_soft);
                }
            }
            scan_pos += 100;
        }

        if let Some((_phase, indices, _ratio)) = clustering.get_dominant_bin() {
            for &idx in &indices {
                acc.accumulate_soft_packet(&candidates[idx], 1.0, None, None);
            }

            let averaged = acc.get_averaged_soft();
            if let Some(mut result) = self.try_decode_packet(&averaged) {
                result.method = format!("accumulator ({} pings)", acc.num_pings());
                return result;
            }
        }

        DecodeResult::failure("Accumulation failed".into(), "0 pings".into(), 0.0)
    }

    fn try_decode_packet(&self, packet_soft: &[f32]) -> Option<DecodeResult> {
        let sync = RxSync {
            found: true,
            correlation: 1.0,
            position: 0,
            sync_bits: 43,
            polarity: 1,
            sync_shift: 0,
            sync_rotation: 0,
            format_hint: 1, // Default to format 1
        };

        let decoded = decode_packet_soft(packet_soft, &sync)?;

        let msg = if decoded.format == 1 {
            Message::from_format1_bits(&self.codec, &decoded.info_bits, &decoded.addr_bits, true, None)
                .or_else(|_| {
                    Message::from_format1_bits(
                        &self.codec,
                        &decoded.info_bits,
                        &decoded.addr_bits,
                        false,
                        self.my_callsign.as_deref(),
                    )
                })
        } else {
            let my = self.my_callsign.as_deref().unwrap_or("");
            let p = self.partner_callsign.as_deref().unwrap_or("");
            Message::from_format2_bits(&self.codec, &decoded.info_bits, &decoded.addr_bits, my, p)
        }
        .ok()?;

        Some(DecodeResult {
            success: true,
            text: msg.text,
            from_call: msg.from_call,
            to_call: msg.to_call,
            format: decoded.format,
            sync_bits: "43/43".into(),
            sync_correlation: 1.0,
            method: "".into(),
            error: None,
        })
    }

    fn demodulate_internal(&self, audio: &[f32]) -> DemodResult {
        let baseband = demodulate_48k(audio);
        let mut magnitude = vec![1.0; baseband.len()];
        for (i, chunk) in audio.chunks(SAMPLES_PER_BIT).enumerate() {
            if i < magnitude.len() {
                magnitude[i] = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
            }
        }
        DemodResult {
            soft_bits: baseband,
            magnitude,
        }
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}
