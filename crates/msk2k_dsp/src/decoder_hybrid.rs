// decoder_hybrid.rs
//
// DJ5HG-faithful decoder: sync peak marks packet END, so take last 258 bits

use crate::callsign::CallsignCodec;
use crate::decode::{decode_packet_soft, is_general_addr, DecodedPacket};
use crate::message::Message;
use crate::rx::{demodulate_msk_soft, RxSync, PACKET_BITS};

#[derive(Clone)]
pub struct HybridDecoderConfig {
    pub corr_min: f32,
    pub sync_bits_min: usize,
    pub my_call: String,
    pub their_call: String,
}

impl Default for HybridDecoderConfig {
    fn default() -> Self {
        Self {
            corr_min: 0.30,
            sync_bits_min: 38,
            my_call: String::new(),
            their_call: String::new(),
        }
    }
}

pub struct HybridDecoder {
    cfg: HybridDecoderConfig,
    codec: CallsignCodec,
}

impl HybridDecoder {
    pub fn decode_audio(&self, audio: &[f32]) -> Option<Message> {
        self.decode_one(audio)
    }
    
    /// Decode with known polarity from sync_tracker
    pub fn decode_audio_with_polarity(&self, audio: &[f32], polarity: i8) -> Option<Message> {
        let soft_bits: Vec<f32> = demodulate_msk_soft(audio);
        
        if soft_bits.len() < PACKET_BITS {
            log::info!("[DEC] X Only {} soft bits (need {})", soft_bits.len(), PACKET_BITS);
            return None;
        }

        // DJ5HG spec 8.2: "the last 258 bits are read from the demodulated signal"
        // The sync peak marks the END of the packet
        let packet_start = soft_bits.len() - PACKET_BITS;
        let packet_raw: Vec<f32> = soft_bits[packet_start..].to_vec();
        
        // Apply polarity from sync_tracker
        let pol = polarity as f32;
        let packet_soft: Vec<f32> = packet_raw.iter().map(|&b| b * pol).collect();
        
        log::info!("[DEC] Extracted last {} bits, polarity={}", PACKET_BITS, polarity);
        
        self.try_decode_packet(&packet_soft, polarity)
    }

    pub fn new() -> Self {
        Self {
            cfg: HybridDecoderConfig::default(),
            codec: CallsignCodec::new(),
        }
    }

    pub fn from_config(cfg: HybridDecoderConfig) -> Self {
        Self {
            cfg,
            codec: CallsignCodec::new(),
        }
    }

    pub fn with_calls(mut self, my_call: impl Into<String>, their_call: impl Into<String>) -> Self {
        self.cfg.my_call = my_call.into();
        self.cfg.their_call = their_call.into();
        self
    }

    pub fn set_calls(&mut self, my_call: impl Into<String>, their_call: impl Into<String>) {
        self.cfg.my_call = my_call.into();
        self.cfg.their_call = their_call.into();
    }

    pub fn config(&self) -> &HybridDecoderConfig {
        &self.cfg
    }

    pub fn set_config(&mut self, cfg: HybridDecoderConfig) {
        self.cfg = cfg;
    }

    /// Main decode - tries both polarities if polarity unknown
    pub fn decode(&self, audio: &[f32]) -> Vec<Message> {
        let soft_bits: Vec<f32> = demodulate_msk_soft(audio);
        
        if soft_bits.len() < PACKET_BITS {
            log::info!("[DEC] X Only {} soft bits (need {})", soft_bits.len(), PACKET_BITS);
            return Vec::new();
        }

        // DJ5HG spec 8.2: "the last 258 bits are read from the demodulated signal"
        let packet_start = soft_bits.len() - PACKET_BITS;
        let packet_raw: Vec<f32> = soft_bits[packet_start..].to_vec();
        
        log::info!("[DEC] Extracted last {} bits from {} total", PACKET_BITS, soft_bits.len());
        
        // Try positive polarity first
        let packet_pos: Vec<f32> = packet_raw.iter().map(|&b| b).collect();
        if let Some(msg) = self.try_decode_packet(&packet_pos, 1) {
            return vec![msg];
        }
        
        // Try negative polarity
        let packet_neg: Vec<f32> = packet_raw.iter().map(|&b| -b).collect();
        if let Some(msg) = self.try_decode_packet(&packet_neg, -1) {
            return vec![msg];
        }
        
        log::info!("[DEC] X Neither polarity decoded");
        Vec::new()
    }

    pub fn decode_one(&self, audio: &[f32]) -> Option<Message> {
        self.decode(audio).into_iter().next()
    }

    fn try_decode_packet(&self, packet_soft: &[f32], polarity: i8) -> Option<Message> {
        // Build a minimal RxSync for decode_packet_soft
        // We'll try Format-1 first (sync_shift=0), then Format-2 shifts if needed
        
        for sync_shift in [0, 14, 29] {
            let sync = RxSync {
                found: true,
                correlation: 1.0,  // We trust sync_tracker's correlation
                position: 0,
                sync_bits: 43,
                polarity: polarity as i32,
                sync_shift,
                sync_rotation: 0,
                format_hint: if sync_shift == 0 { 1 } else { 2 },
            };
            
            if let Some(pkt) = decode_packet_soft(packet_soft, &sync) {
                log::info!(
                    "[DEC] FEC OK: format={} shift={} info={} addr={}", 
                    pkt.format, sync_shift, pkt.info_bits.len(), pkt.addr_bits.len()
                );
                
                if let Some(msg) = self.decoded_to_message(pkt) {
                    log::info!("[DEC] OK DECODED: '{}' from {} (pol={}, shift={})", 
                        msg.text, msg.from_call, polarity, sync_shift);
                    return Some(msg);
                }
            }
        }
        
        log::debug!("[DEC] FEC failed for polarity={}", polarity);
        None
    }

    fn decoded_to_message(&self, pkt: DecodedPacket) -> Option<Message> {
        match pkt.format {
            1 => {
                let is_general = is_general_addr(&pkt.addr_bits);
                log::info!("[DEC] Format-1 is_general={}", is_general);
                
                match Message::from_format1_bits(&self.codec, &pkt.info_bits, &pkt.addr_bits, is_general, Some(self.cfg.my_call.as_str())) {
                    Ok(msg) => Some(msg),
                    Err(e) => {
                        log::info!("[DEC] X Format-1 parse error: {}", e);
                        None
                    }
                }
            }
            2 => {
                if self.cfg.my_call.is_empty() || self.cfg.their_call.is_empty() {
                    log::info!("[DEC] X Format-2 needs my_call/their_call");
                    return None;
                }
                
                match Message::from_format2_bits(
                    &self.codec,
                    &pkt.info_bits,
                    &pkt.addr_bits,
                    &self.cfg.my_call,
                    &self.cfg.their_call,
                ) {
                    Ok(msg) => Some(msg),
                    Err(e) => {
                        log::info!("[DEC] X Format-2 parse error: {}", e);
                        None
                    }
                }
            }
            _ => {
                log::info!("[DEC] X Unknown format {}", pkt.format);
                None
            }
        }
    }
}
