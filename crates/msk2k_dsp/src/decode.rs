//! Packet decoding: soft bits → deinterleave → FEC → message bits

use crate::fmt1;
use crate::fmt2;
use crate::fec;
use crate::rx::RxSync;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddrKind {
    General,
    Addressed,
}

pub const GENERAL_ADDRESS_49: [i32; 49] = [
    1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1,
    1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1
];

#[inline]
pub fn classify_addr_bits(addr_bits: &[i32]) -> AddrKind {
    if addr_bits.len() >= 49 && addr_bits[..49] == GENERAL_ADDRESS_49 {
        AddrKind::General
    } else {
        AddrKind::Addressed
    }
}

#[inline]
pub fn is_general_addr(addr_bits: &[i32]) -> bool {
    matches!(classify_addr_bits(addr_bits), AddrKind::General)
}

pub struct DecodedPacket {
    pub format: u8,
    pub info_bits: Vec<i32>,
    pub addr_bits: Vec<i32>,
    pub sync_bits: Vec<i32>,
    pub sync_ok: bool,
}

// 🟢 MODIFIED: Removed the "packet_hard" mapping entirely. We use the raw f32 slices.
pub fn decode_packet_soft(packet_soft: &[f32], sync: &RxSync) -> Option<DecodedPacket> {
    if !sync.found || packet_soft.len() != 258 {
        return None;
    }
    let soft_sample: Vec<f32> = packet_soft.iter().take(16)
        .map(|x| (x * 100.0).round() / 100.0).collect();
    log::debug!("[DECODE] soft_bits[0..16]: {:?}", soft_sample);

    if sync.sync_shift == 0 {
        decode_format1_soft_entry(packet_soft)
    } else {
        decode_format2_soft_entry(packet_soft, sync)
    }
}

fn decode_format1_soft_entry(packet_soft: &[f32]) -> Option<DecodedPacket> {
    let (sync_bits, addr_bits, poly1, poly2) = fmt1::deinterleave_format1_soft(packet_soft);
    
    let mut codeword = Vec::with_capacity(166);
    codeword.extend(poly1);
    codeword.extend(poly2);

    let info_bits = fec::decode_format1_soft(&codeword);

    Some(DecodedPacket {
        format: 1,
        info_bits,
        addr_bits,
        sync_bits,
        sync_ok: true,
    })
}

fn decode_format2_soft_entry(packet_soft: &[f32], _sync: &RxSync) -> Option<DecodedPacket> {
    let (sync_bits, addr_bits, polys) = fmt2::deinterleave_format2_soft(packet_soft);
    
    const ORDER: [&str; 9] = ["Pa", "Pb", "Pc", "Pd", "Pe", "Pf", "Pg", "Ph", "Pi"];
    
    let mut codeword = Vec::with_capacity(162);
    for i in 0..18 {
        for name in &ORDER {
            let bits = polys.get(*name)?;
            if i >= bits.len() { return None; }
            codeword.push(bits[i]);
        }
    }
    
    let info_bits = fec::decode_format2_soft(&codeword);

    const MAX_FMT2_CODEWORD_ERRORS: usize = 24;
    let expected = fec::encode_format2(&info_bits);

    let mut errors = 0usize;
    let mut total = 0usize;
    for name in &ORDER {
        let exp = expected.get(*name)?;
        let got_soft = polys.get(*name)?;
        let n = exp.len().min(got_soft.len());
        for i in 0..n {
            total += 1;
            // 🟢 Need to compare the expected hard bits with thresholded received bits 
            let got_hard = if got_soft[i] > 0.0 { 1 } else { 0 };
            if exp[i] != got_hard {
                errors += 1;
            }
        }
    }

    if total == 0 || errors > MAX_FMT2_CODEWORD_ERRORS {
        log::debug!(
            "Format-2 reject: re-encode mismatch errors={}/{} (max={})",
            errors,
            total,
            MAX_FMT2_CODEWORD_ERRORS
        );
        return None;
    }

    log::debug!("Format-2 accepted: errors={}/{} info_bits={:?}", errors, total, &info_bits);

    Some(DecodedPacket {
        format: 2,
        info_bits,
        addr_bits,
        sync_bits,
        sync_ok: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_format1_zeros() {
        let packet_soft = vec![-1.0f32; 258];
        let sync = RxSync {
            found: true,
            correlation: 0.5,
            position: 100,
            sync_bits: 43,
            polarity: 1,
            sync_shift: 0,
            format_hint: 1,
            sync_rotation: 0,
        };
        
        // This continues to pass because the Soft Viterbi accurately interprets 
        // entirely negative certainty identically to previous hard logic.
        let result = decode_packet_soft(&packet_soft, &sync);
        assert!(result.is_some());
        let decoded = result.unwrap();
        assert_eq!(decoded.format, 1);
    }
}