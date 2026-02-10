// src/engine/accumulator.rs

use log::{debug, info};
use msk2k_dsp::callsign::CallsignCodec;
use msk2k_dsp::decode::{decode_packet_soft, is_general_addr};
use msk2k_dsp::message::Message;
use msk2k_dsp::rx::{PacketCandidate, RxSync};

const PACKET_BITS: usize = 258;
const SAMPLES_PER_BIT: usize = 24; 
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

    pub fn clear(&mut self) {
        self.candidates.clear();
    }

    pub fn update_config(&mut self, my_call: &str, their_call: Option<String>) {
        self.my_call = my_call.to_string();
        self.their_call = their_call;
    }

    pub fn add(&mut self, cand: PacketCandidate) {
        self.candidates.push(cand);
    }

    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    pub fn process(&self) -> Option<Message> {
        if self.candidates.len() < 2 { return None; }

        // ============================================================================
        // 🟢 DSP ACCUMULATION LOGIC - DO NOT MODIFY THIS SECTION
        // ============================================================================
        
        let mut virtual_hist = vec![0.0f32; PACKET_SAMPLES];
        let window = SAMPLES_PER_BIT * 4;

        for cand in &self.candidates {
            let shift_offset = cand.sync.sync_shift * SAMPLES_PER_BIT as i32;
            let v_idx = (cand.end_index as i32 - shift_offset).rem_euclid(PACKET_SAMPLES as i32) as usize;
            let weight = cand.sync.correlation.powi(2);
            for i in 0..=window { 
                virtual_hist[(v_idx + i) % PACKET_SAMPLES] += weight;
            }
            for i in 1..=window { 
                virtual_hist[(v_idx + PACKET_SAMPLES - i) % PACKET_SAMPLES] += weight;
            }
        }

        let (grid_center, _) = virtual_hist.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();

        let mut aligned = Vec::new();
        for cand in &self.candidates {
            let shift_offset = cand.sync.sync_shift * SAMPLES_PER_BIT as i32;
            let v_idx = (cand.end_index as i32 - shift_offset).rem_euclid(PACKET_SAMPLES as i32) as usize;
            let diff = (v_idx as i32 - grid_center as i32).abs();
            let wrap_diff = (PACKET_SAMPLES as i32 - diff).abs();
            if diff.min(wrap_diff) <= (SAMPLES_PER_BIT as i32 * 12) { aligned.push(cand); }
        }
        aligned.sort_by(|a, b| b.sync.correlation.partial_cmp(&a.sync.correlation).unwrap());

        let mut final_cands = Vec::new();
        let mut kept_records: Vec<(i32, i32)> = Vec::new();
        for cand in aligned {
            let p_idx = cand.end_index as i32;
            let shift = cand.sync.sync_shift;
            let is_echo = kept_records.iter().any(|(kp, ks)| (*kp - p_idx).abs() < 2400 && *ks == shift);
            if !is_echo { final_cands.push(cand); kept_records.push((p_idx, shift)); }
        }
        if final_cands.is_empty() { return None; }

        debug!("[Accum] Processing {} unique fragments", final_cands.len());

        let mut accum_llr = vec![0.0f32; PACKET_BITS];
        let mut accum_weight = vec![0.0f32; PACKET_BITS];
        
        for cand in final_cands {
            let pol = if cand.sync.polarity > 0 { 1.0 } else { -1.0 };
            let shift_rot = cand.sync.sync_shift * 6;
            let w_global = cand.sync.correlation.powi(2);

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
                let mut gap = 0; 
                for i in 1..PACKET_BITS {
                    let idx = (peak_i as i32 + i as i32).rem_euclid(PACKET_BITS as i32) as usize;
                    if smoothed[idx] > thresh { 
                        for g in 0..=gap { mask[(idx as i32 - g as i32).rem_euclid(PACKET_BITS as i32) as usize] = true; }
                        gap = 0; 
                    } else { gap += 1; if gap > 20 { break; } }
                }
                gap = 0;
                for i in 1..PACKET_BITS {
                    let idx = (peak_i as i32 - i as i32).rem_euclid(PACKET_BITS as i32) as usize;
                    if smoothed[idx] > thresh { for g in 0..=gap { mask[(idx as i32 + g as i32).rem_euclid(PACKET_BITS as i32) as usize] = true; } gap = 0; } else { gap += 1; if gap > 20 { break; } }
                }
            }
            for i in 0..PACKET_BITS {
                if mask[i] {
                    let target = (i as i32 - shift_rot).rem_euclid(PACKET_BITS as i32) as usize;
                    accum_llr[target] += cand.packet_soft[i] * w_global * pol;
                    accum_weight[target] += w_global;
                }
            }
        }

        let mut final_soft = vec![0.0f32; PACKET_BITS]; 
        let mut max_val = 0.0f32;
        for i in 0..PACKET_BITS {
            if accum_weight[i] > 0.0 {
                final_soft[i] = accum_llr[i] / accum_weight[i];
                if final_soft[i].abs() > max_val { max_val = final_soft[i].abs(); }
            }
        }
        if max_val > 0.0 { for x in &mut final_soft { *x /= max_val; } }

        // ============================================================================
        // 🟢 DECODE LOGIC - UPDATED FOR CQ+GRID SUPPORT
        // ============================================================================
        
        let codec = CallsignCodec::new();
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
                    let msg_res = if pkt.format == 1 {
                        let is_general = is_general_addr(&pkt.addr_bits);
                        let b55 = pkt.info_bits.get(55).unwrap_or(&0);
                        let b56 = pkt.info_bits.get(56).unwrap_or(&0);
                        
                        // 🟢 TYPE 11: CQ+GRID (7-char + grid)
                        if is_general && *b55 == 1 && *b56 == 1 {
                            match codec.decode_cq_with_grid(&pkt.info_bits) {
                                Ok(grid_text) => {
                                    let parts: Vec<&str> = grid_text.split_whitespace().collect();
                                    let clean_call = parts.get(0).unwrap_or(&"?").to_string();
                                    let grid = parts.get(1).unwrap_or(&"");
                                    let ui_text = format!("CQ {} {}", clean_call, grid);
                                    Ok(Message {
                                        from_call: clean_call,
                                        to_call: None,
                                        text: ui_text.clone(),
                                        content: msk2k_dsp::message::MessageContent::Format1 { text: ui_text },
                                        format: 1,
                                    })
                                }
                                Err(_) => continue, // CRC failed, try next hint
                            }
                        } else {
                            // 🟢 STANDARD FORMAT 1 (Type 01 or directed)
                            Message::from_format1_bits(&codec, &pkt.info_bits, &pkt.addr_bits, is_general, Some(&self.my_call))
                        }
                    } else if pkt.format == 2 {
                        if self.my_call.is_empty() { continue; }
                        Message::from_format2_bits(&codec, &pkt.info_bits, &pkt.addr_bits, &self.my_call, self.their_call.as_deref().unwrap_or(""))
                    } else {
                        continue;
                    };

                    if let Ok(m) = msg_res {
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
