// src/engine/accumulator.rs

use log::{debug, info};
use msk2k_dsp::callsign::CallsignCodec;
use msk2k_dsp::decode::{decode_packet_soft, is_general_addr};
use msk2k_dsp::message::Message;
use msk2k_dsp::rx::{PacketCandidate, RxSync};

const PACKET_BITS: usize = 258;
const SAMPLES_PER_BIT: usize = 24; // 48k / 2000
const PACKET_SAMPLES: usize = PACKET_BITS * SAMPLES_PER_BIT;
const ENVELOPE_THRESHOLD: f32 = 0.25;

pub struct Accumulator {
    candidates: Vec<PacketCandidate>,
    my_call: String,
    their_call: Option<String>,
}

impl Accumulator {
    pub fn new(my_call: &str, their_call: Option<String>) -> Self {
        Self {
            candidates: Vec::new(),
            my_call: my_call.to_string(),
            their_call,
        }
    }

    /// Reset internal state for a new 15s slot
    pub fn clear(&mut self) {
        self.candidates.clear();
    }

    /// Update configuration
    pub fn update_config(&mut self, my_call: &str, their_call: Option<String>) {
        self.my_call = my_call.to_string();
        self.their_call = their_call;
    }

    /// Add a candidate. Pass *everything* found by the extractor.
    pub fn add(&mut self, cand: PacketCandidate) {
        self.candidates.push(cand);
    }

    /// Run the Deep Search accumulation logic.
    pub fn process(&self) -> Option<Message> {
        if self.candidates.len() < 2 {
            return None; 
        }

        // 1. Grid Consensus
        let mut virtual_hist = vec![0.0f32; PACKET_SAMPLES];
        let window = SAMPLES_PER_BIT * 4;

        for cand in &self.candidates {
            let shift_offset = cand.sync.sync_shift * SAMPLES_PER_BIT as i32;
            let v_idx = (cand.end_index as i32 - shift_offset).rem_euclid(PACKET_SAMPLES as i32) as usize;
            let weight = cand.sync.correlation.powi(2);
            
            for i in 0..=window { 
                let idx = (v_idx + i) % PACKET_SAMPLES;
                virtual_hist[idx] += weight;
            }
            for i in 1..=window { 
                let idx = (v_idx + PACKET_SAMPLES - i) % PACKET_SAMPLES;
                virtual_hist[idx] += weight;
            }
        }

        let (grid_center, _) = virtual_hist.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();

        // 2. Filter & Dedupe
        let mut aligned = Vec::new();
        for cand in &self.candidates {
            let shift_offset = cand.sync.sync_shift * SAMPLES_PER_BIT as i32;
            let v_idx = (cand.end_index as i32 - shift_offset).rem_euclid(PACKET_SAMPLES as i32) as usize;
            
            let diff = (v_idx as i32 - grid_center as i32).abs();
            let wrap_diff = (PACKET_SAMPLES as i32 - diff).abs();
            
            if diff.min(wrap_diff) <= (SAMPLES_PER_BIT as i32 * 12) {
                aligned.push(cand);
            }
        }

        aligned.sort_by(|a, b| b.sync.correlation.partial_cmp(&a.sync.correlation).unwrap());

        let mut final_cands = Vec::new();
        let mut kept_records: Vec<(i32, i32)> = Vec::new();

        for cand in aligned {
            let p_idx = cand.end_index as i32;
            let shift = cand.sync.sync_shift;
            
            // Dedupe echo: same time (+/- 50ms), same shift
            // FIX: Use simple tuple access to avoid type inference issues
            let is_echo = kept_records.iter().any(|rec| {
                let (kp, ks) = *rec; // Copy values out
                (kp - p_idx).abs() < 2400 && ks == shift
            });

            if !is_echo {
                final_cands.push(cand);
                kept_records.push((p_idx, shift));
            }
        }

        if final_cands.is_empty() { return None; }

        debug!("[Accum] Processing {} unique fragments", final_cands.len());

        // 3. Accumulate (Universal + Calibrated)
        let mut accum_llr = vec![0.0f32; PACKET_BITS];
        let mut accum_weight = vec![0.0f32; PACKET_BITS];
        let mut max_val = 0.0f32;

        for cand in final_cands {
            let pol = if cand.sync.polarity > 0 { 1.0 } else { -1.0 };
            // CALIBRATION FIX: S00->0, S14->84, S29->174
            let shift_rot = cand.sync.sync_shift * 6;
            
            let w_global = cand.sync.correlation.powi(2);

            // Envelope Logic
            let mut smoothed = vec![0.0f32; PACKET_BITS];
            let mut peak = 0.0f32;
            let mut peak_i = 0;
            for i in 0..PACKET_BITS {
                let mut sum = 0.0;
                for j in -2..=2 {
                    let k = (i as i32 + j).rem_euclid(PACKET_BITS as i32) as usize;
                    sum += cand.packet_soft[k].abs();
                }
                smoothed[i] = sum / 5.0;
                if smoothed[i] > peak { peak = smoothed[i]; peak_i = i; }
            }

            let thresh = peak * ENVELOPE_THRESHOLD;
            let mut mask = vec![false; PACKET_BITS];
            if peak > 0.0 {
                mask[peak_i] = true;
                // Simple gap fill
                let mut gap = 0; 
                for i in 1..PACKET_BITS {
                    let idx = (peak_i as i32 + i as i32).rem_euclid(PACKET_BITS as i32) as usize;
                    if smoothed[idx] > thresh { 
                        // Backfill gap
                        for g in 0..=gap { 
                            let b_idx = (idx as i32 - g as i32).rem_euclid(PACKET_BITS as i32) as usize;
                            mask[b_idx] = true; 
                        }
                        gap = 0; 
                    } else { 
                        gap += 1; 
                        if gap > 20 { break; } 
                    }
                }
                gap = 0;
                for i in 1..PACKET_BITS {
                    let idx = (peak_i as i32 - i as i32).rem_euclid(PACKET_BITS as i32) as usize;
                    if smoothed[idx] > thresh {
                        for g in 0..=gap {
                            let b_idx = (idx as i32 + g as i32).rem_euclid(PACKET_BITS as i32) as usize;
                            mask[b_idx] = true;
                        }
                        gap = 0;
                    } else {
                        gap += 1;
                        if gap > 20 { break; }
                    }
                }
            }

            for i in 0..PACKET_BITS {
                if mask[i] {
                    // Apply Calibrated Rotation
                    let target = (i as i32 - shift_rot).rem_euclid(PACKET_BITS as i32) as usize;
                    accum_llr[target] += cand.packet_soft[i] * w_global * pol;
                    accum_weight[target] += w_global;
                }
            }
        }

        // 4. Normalize
        let mut final_soft = vec![0.0f32; PACKET_BITS];
        for i in 0..PACKET_BITS {
            if accum_weight[i] > 0.0 {
                final_soft[i] = accum_llr[i] / accum_weight[i];
                if final_soft[i].abs() > max_val { max_val = final_soft[i].abs(); }
            }
        }
        if max_val > 0.0 {
            for x in &mut final_soft { *x /= max_val; }
        }

        // 5. Decode (Try Normal + Inverted)
        let codec = CallsignCodec::new();
        // Dummy Sync struct required by decoder (we fake a perfect lock)
        // Try all format hints just in case
        let hints = [(0, 1), (14, 2), (29, 2)];

        for invert in [false, true] {
            let pol_mod = if invert { -1.0 } else { 1.0 };
            let mut test_buf = vec![0.0f32; PACKET_BITS];
            for i in 0..PACKET_BITS { test_buf[i] = final_soft[i] * pol_mod; }

            for &(shift, fmt) in &hints {
                let ts = RxSync { 
                    found: true, correlation: 0.99, position: 0, sync_bits: 43, 
                    polarity: 1, sync_shift: shift, sync_rotation: 0, format_hint: fmt 
                };

                if let Some(pkt) = decode_packet_soft(&test_buf, &ts) {
                    let is_general = is_general_addr(&pkt.addr_bits);
                    
                    let msg_res = if pkt.format == 1 {
                        Message::from_format1_bits(&codec, &pkt.info_bits, &pkt.addr_bits, is_general, Some(&self.my_call))
                    } else if pkt.format == 2 {
                        if self.my_call.is_empty() { continue; }
                        Message::from_format2_bits(&codec, &pkt.info_bits, &pkt.addr_bits, &self.my_call, self.their_call.as_deref().unwrap_or(""))
                    } else {
                        continue;
                    };

                    if let Ok(m) = msg_res {
                        // Loopback protection
                        if m.from_call == self.my_call { continue; }
                        
                        info!("[Accum] 🎯 Recovered Message: '{}' (Inv: {})", m.text, invert);
                        return Some(m);
                    }
                }
            }
        }

        None
    }
}