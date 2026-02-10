// crates/msk2k_dsp/src/callsign.rs

use std::collections::HashMap;

/// Callsign encoder/decoder for MSK2K protocol
pub struct CallsignCodec {
    /// Base-37 alphabet: /ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
    alphabet: Vec<char>,
    /// Base-42 alphabet for Format 1 Text/Grid encoding
    alphabet_b42: Vec<char>,
    /// Prime numbers for parity generation (prime -> num_bits)
    primes: HashMap<u32, usize>,
    /// Parity selection by callsign length
    parity_selection: HashMap<usize, Vec<u32>>,
    /// Length codes (4 bits for 3-9 chars, 1 bit for 10 chars)
    length_codes: HashMap<usize, Vec<i32>>,
}

impl CallsignCodec {
    pub fn new() -> Self {
        let alphabet_str = "/ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let alphabet: Vec<char> = alphabet_str.chars().collect();
        
        // 🟢 CORRECT ALIGNMENT: Standard 37 + 5 unique chars = 42
        // We add: Space, Dot, Comma, Question, Dash
        let alphabet_b42: Vec<char> = "/ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?-".chars().collect();

        let mut primes = HashMap::new();
        primes.insert(7, 3);
        primes.insert(13, 4);
        primes.insert(23, 5);
        primes.insert(29, 5);
        primes.insert(31, 5);
        primes.insert(59, 6);
        primes.insert(61, 6);

        let mut parity_selection = HashMap::new();
        parity_selection.insert(3, vec![7, 13, 23, 29, 31, 59, 61]);
        parity_selection.insert(4, vec![7, 13, 29, 31, 59, 61]);
        parity_selection.insert(5, vec![7, 13, 29, 31, 61]);
        parity_selection.insert(6, vec![7, 13, 29, 31]);
        parity_selection.insert(7, vec![7, 29, 31]);
        parity_selection.insert(8, vec![7, 31]);
        parity_selection.insert(9, vec![7]);
        parity_selection.insert(10, vec![]);

        let mut length_codes = HashMap::new();
        length_codes.insert(3, vec![0, 1, 1, 1]);
        length_codes.insert(4, vec![1, 0, 1, 1]);
        length_codes.insert(5, vec![0, 0, 1, 1]);
        length_codes.insert(6, vec![1, 1, 0, 1]);
        length_codes.insert(7, vec![0, 1, 0, 1]);
        length_codes.insert(8, vec![1, 0, 0, 1]);
        length_codes.insert(9, vec![0, 0, 0, 1]);
        length_codes.insert(10, vec![0]);

        Self {
            alphabet,
            alphabet_b42,
            primes,
            parity_selection,
            length_codes,
        }
    }

    /// Packs 4 Maidenhead indices into a 16-bit integer
    pub fn pack_grid(indices: &[usize; 4]) -> u16 {
        let f1 = indices[0] as u16; // A-R (0-17)
        let f2 = indices[1] as u16; // A-R (0-17)
        let n1 = indices[2] as u16; // 0-9
        let n2 = indices[3] as u16; // 0-9
        (f1 * 18 * 100) + (f2 * 100) + (n1 * 10) + n2
    }

    /// Unpacks a 16-bit integer back into a 4-figure Maidenhead string
    pub fn unpack_grid_to_string(grid_val: u16) -> String {
        let alphabet: Vec<char> = "ABCDEFGHIJKLMNOPQR".chars().collect();
        let f1 = (grid_val / 1800) as usize;
        let f2 = ((grid_val % 1800) / 100) as usize;
        let n1 = ((grid_val % 100) / 10) as usize;
        let n2 = (grid_val % 10) as usize;

        format!("{}{}{}{}", alphabet[f1.min(17)], alphabet[f2.min(17)], n1.min(9), n2.min(9))
    }

    // ============================================================================
    // 🟢 NEW: 7-CHARACTER CALLSIGN + GRID (FITS IN STANDARD 71-BIT FORMAT 1)
    // ============================================================================
    
    /// Encode 7-char callsign + 16-bit grid into 71 bits total
    /// Layout: 22 bits (call part 1) + 17 bits (call part 2) + 16 bits (grid) + 2 type + 14 CRC = 71 bits
    pub fn encode_cq_with_grid(&self, call: &str, indices: &[usize; 4]) -> Result<Vec<i32>, String> {
        let call_trimmed = call.trim().to_uppercase();
        if call_trimmed.len() > 7 {
            return Err("Callsign limited to 7 chars in Grid Mode".into());
        }
        
        let call_padded = format!("{: <7}", call_trimmed); // Pad to 7 chars
        let grid_val = Self::pack_grid(indices);
        
        // Split 7-char call: first 4 + last 3 (base-42)
        let z1 = self.string_to_base42(&call_padded[0..4]); // 42^4 needs 22 bits
        let z2 = self.string_to_base42(&call_padded[4..7]); // 42^3 needs 17 bits
        
        // Pack into 55 data bits
        let mut bits = self.int_to_bits(z1, 22);           // bits 0-21
        bits.extend(self.int_to_bits(z2, 17));             // bits 22-38
        bits.extend(self.int_to_bits(grid_val as u64, 16)); // bits 39-54
        
        // Add type code 11 (bits 55-56)
        bits.push(1); 
        bits.push(1); 
        
        // Add 14-bit CRC (bits 57-70) - total 71 bits
        let data_value = self.bits_to_int(&bits[0..55]);
        let parity = data_value % 16381; // Use 14-bit prime
        bits.extend(self.int_to_bits(parity, 14));
        
        Ok(bits) // 71 bits total
    }
    
    /// Decode a 71-bit CQ+grid packet back to "CALL GRID"
    pub fn decode_cq_with_grid(&self, bits: &[i32]) -> Result<String, String> {
        if bits.len() < 71 { 
            return Err("Insufficient bits".into()); 
        }

        // Check type code (bits 55-56 should be [1,1])
        if bits[55] != 1 || bits[56] != 1 {
            return Err("Not a CQ+grid message".into());
        }
        
        // 🟢 VALIDATE CRC (bits 57-70 is 14 bits, prime r=16381)
        let data_value = self.bits_to_int(&bits[0..55]);
        let expected_parity = data_value % 16381;
        let received_parity = self.bits_to_int(&bits[57..71]);
        
        if expected_parity != received_parity {
            return Err("CRC failed".into());
        }
        
        // 🟢 CRC PASSED - DECODE
        let z1 = self.bits_to_int(&bits[0..22]);
        let z2 = self.bits_to_int(&bits[22..39]);
        let grid_val = self.bits_to_int(&bits[39..55]) as u16;
        
        let call1 = self.base42_decode_to_string(z1, 4);
        let call2 = self.base42_decode_to_string(z2, 3);
        let grid_str = Self::unpack_grid_to_string(grid_val);
        
        let callsign = format!("{}{}", call1, call2).trim().to_string();
        Ok(format!("{} {}", callsign, grid_str))
    }

    // ============================================================================
    // 🔴 OLD: 8-CHARACTER + GRID (62 bits, COMMENTED OUT FOR SAFETY)
    // ============================================================================
    
    /*
    /// OLD MSK2K Custom: Encode 8-char callsign + 16-bit Grid (62 bits total - NO CRC)
    pub fn encode_cq_with_grid_old(&self, call: &str, indices: &[usize; 4]) -> Result<Vec<i32>, String> {
        let call_padded = format!("{: <8}", call.trim().to_uppercase());
        if call_padded.len() > 8 {
            return Err("Callsign limited to 8 chars in Grid Mode".into());
        }

        let grid_val = Self::pack_grid(indices);
        let z1 = self.string_to_base42(&call_padded[0..5]); 
        let z2_call = self.string_to_base42(&call_padded[5..8]); 

        let mut bits = self.int_to_bits(z1, 27);
        let z2_combined = (z2_call * 1024) + (grid_val as u64 >> 6); 
        bits.extend(self.int_to_bits(z2_combined, 27));

        bits.push(1);
        bits.push(1);
        bits.extend(self.int_to_bits(grid_val as u64 & 0x3F, 6));

        Ok(bits) // 62 bits - NEEDS 15 MORE FOR CRC
    }

    /// OLD Decode a 62-bit Type 11 packet back to "CALL GRID" (NO CRC CHECK)
    pub fn decode_cq_with_grid_old(&self, bits: &[i32]) -> String {
        if bits.len() < 56 { 
            return "ERROR".to_string(); 
        }

        let z1 = self.bits_to_int(&bits[0..27]);
        let z2_combined = self.bits_to_int(&bits[27..54]);
        let grid_low_bits = self.bits_to_int(&bits[56..62]); 

        let z2_call = z2_combined >> 10;
        let grid_high_bits = (z2_combined & 0x3FF) << 6; 
        
        let grid_val = (grid_high_bits | grid_low_bits) as u16;

        let call1 = self.base42_decode_to_string(z1, 5);
        let call2 = self.base42_decode_to_string(z2_call, 3);
        
        let grid_str = Self::unpack_grid_to_string(grid_val);

        format!("{} {}", format!("{}{}", call1, call2).trim(), grid_str)
    }
    */

    // ============================================================================
    // HELPER FUNCTIONS (UNCHANGED)
    // ============================================================================

    fn string_to_base42(&self, s: &str) -> u64 {
        let mut z: u64 = 0;
        for ch in s.chars() {
            let digit = self.alphabet_b42.iter().position(|&c| c == ch).unwrap_or(26) as u64;
            z = z * 42 + digit;
        }
        z
    }

    fn base42_decode_to_string(&self, mut value: u64, length: usize) -> String {
        let mut chars = Vec::with_capacity(length);
        let alpha_len = self.alphabet_b42.len();

        for _ in 0..length {
            let idx = (value % 42) as usize;
            let safe_idx = idx.min(alpha_len - 1);
            chars.push(self.alphabet_b42[safe_idx]);
            value /= 42;
        }
        chars.reverse();
        chars.into_iter().collect()
    }

    fn is_plausible_callsign_minimal(cs: &str) -> bool {
        let cs = cs.trim();
        if cs.len() < 3 || cs.len() > 10 { return false; }
        if !cs.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '/') { return false; }
        let slash_count = cs.chars().filter(|&c| c == '/').count();
        if slash_count > 2 { return false; }
        if cs.starts_with('/') || cs.ends_with('/') { return false; }
        if cs.contains("//") { return false; }
        true
    }

    pub fn encode_callsign(&self, callsign: &str) -> Result<Vec<i32>, String> {
        let call = callsign.trim().to_uppercase();
        let call_len = call.len();
        if call_len < 3 || call_len > 10 { return Err(format!("Callsign length must be 3-10 characters, got {}", call_len)); }

        let mut z: u64 = 0;
        for ch in call.chars() {
            let digit = match self.alphabet.iter().position(|&c| c == ch) {
                Some(d) => d,
                None => return Err(format!("Invalid callsign character: '{}'", ch)),
            };
            z = z * 37 + digit as u64;
        }

        if call_len == 10 { return self.encode_10char_callsign(&call); }
        let bit_len = match call_len { 3 => 16, 4 => 21, 5 => 27, 6 => 32, 7 => 37, 8 => 42, 9 => 47, _ => 0 };
        let mut total_bits = self.int_to_bits(z, bit_len);

        if let Some(primes) = self.parity_selection.get(&call_len) {
            for &prime in primes {
                let remainder = (z % prime as u64) as u32;
                total_bits.extend(self.int_to_bits(remainder as u64, self.primes[&prime]));
            }
        }

        while total_bits.len() < 50 { total_bits.push(0); }
        total_bits.truncate(50);
        total_bits.extend(&self.length_codes[&call_len]);
        Ok(total_bits[..54].to_vec())
    }

    fn encode_10char_callsign(&self, call: &str) -> Result<Vec<i32>, String> {
        let call1 = &call[..6];
        let call2 = &call[6..];
        let mut z1: u64 = 0;
        for ch in call1.chars() { z1 = z1 * 37 + self.alphabet.iter().position(|&c| c == ch).unwrap() as u64; }
        let mut z2: u64 = 0;
        for ch in call2.chars() { z2 = z2 * 37 + self.alphabet.iter().position(|&c| c == ch).unwrap() as u64; }
        let mut bits = self.int_to_bits(z1, 32);
        bits.extend(self.int_to_bits(z2, 21));
        bits.push(0);
        Ok(bits)
    }

    pub fn decode_callsign(&self, bits: &[i32]) -> String {
        if bits.len() != 54 { return "ERROR".to_string(); }
        if bits[53] == 0 { return self.decode_10char_callsign(bits); }
        let length_code: Vec<i32> = bits[50..54].to_vec();
        let mut call_len = None;
        for (&length, code) in &self.length_codes {
            if code == &length_code { call_len = Some(length); break; }
        }
        let call_len = match call_len { Some(len) => len, None => return "ERROR".to_string() };
        let bit_len = match call_len { 3 => 16, 4 => 21, 5 => 27, 6 => 32, 7 => 37, 8 => 42, 9 => 47, _ => 0 };
        let z = self.bits_to_int(&bits[..bit_len]);
        let decoded = self.base37_decode(z, call_len).trim().to_string();
        if !Self::is_plausible_callsign_minimal(&decoded) { return "ERROR".to_string(); }
        decoded
    }

    fn decode_10char_callsign(&self, bits: &[i32]) -> String {
        let z1 = self.bits_to_int(&bits[..32]);
        let z2 = self.bits_to_int(&bits[32..53]);
        let call1 = self.base37_decode(z1, 6);
        let call2 = self.base37_decode(z2, 4);
        let decoded = format!("{}{}", call1, call2).trim().to_string();
        if !Self::is_plausible_callsign_minimal(&decoded) { return "ERROR".to_string(); }
        decoded
    }

    pub fn generate_private_address(&self, callsign: &str) -> Result<Vec<i32>, String> {
        let full_code = self.encode_callsign(callsign)?;
        Ok(full_code[..49].to_vec())
    }

    pub fn decode_private_address(&self, addr_bits: &[i32]) -> String {
        if addr_bits.len() < 49 { return "ERROR".to_string(); }
        let length_code_options: &[(usize, [i32; 4])] = &[ (3, [0,1,1,1]), (4, [1,0,1,1]), (5, [0,0,1,1]), (6, [1,1,0,1]), (7, [0,1,0,1]), (8, [1,0,0,1]), (9, [0,0,0,1]) ];
        for &(_, ref lcode) in length_code_options {
            for bit49 in [0, 1] {
                let mut reconstructed = vec![0; 54];
                reconstructed[..49].copy_from_slice(&addr_bits[..49]);
                reconstructed[49] = bit49;
                reconstructed[50..54].copy_from_slice(lcode);
                let res = self.decode_callsign(&reconstructed);
                if res != "ERROR" { return res; }
            }
        }
        "UNKNOWN".to_string()
    }

    pub fn generate_parity(&self, source_bits: &[i32], r: u32, num_bits: usize) -> Vec<i32> {
        let value = self.bits_to_int(source_bits);
        self.int_to_bits(value % r as u64, num_bits)
    }

    pub fn int_to_bits(&self, value: u64, num_bits: usize) -> Vec<i32> {
        let mut bits = Vec::with_capacity(num_bits);
        for i in (0..num_bits).rev() { bits.push(((value >> i) & 1) as i32); }
        bits
    }

    pub fn bits_to_int(&self, bits: &[i32]) -> u64 {
        let mut value: u64 = 0;
        for &bit in bits { value = (value << 1) | (bit as u64); }
        value
    }

    fn base37_decode(&self, mut value: u64, length: usize) -> String {
        let mut chars = Vec::with_capacity(length);
        for _ in 0..length {
            chars.push(self.alphabet[(value % 37) as usize]);
            value /= 37;
        }
        chars.reverse();
        chars.into_iter().collect()
    }
}

impl Default for CallsignCodec { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_6char() {
        let codec = CallsignCodec::new();
        let callsign = "KA1ABC";

        let encoded = codec.encode_callsign(callsign).unwrap();
        assert_eq!(encoded.len(), 54);

        let decoded = codec.decode_callsign(&encoded);
        assert_eq!(decoded, callsign);
    }

    #[test]
    fn test_encode_decode_4char() {
        let codec = CallsignCodec::new();
        let callsign = "W1AW";

        let encoded = codec.encode_callsign(callsign).unwrap();
        assert_eq!(encoded.len(), 54);

        let decoded = codec.decode_callsign(&encoded);
        assert_eq!(decoded, callsign);
    }

    #[test]
    fn test_cq_with_grid_7char() {
        let codec = CallsignCodec::new();
        let grid_indices = [8, 14, 8, 3]; // IO83
        
        let encoded = codec.encode_cq_with_grid("GW4WND", &grid_indices).unwrap();
        assert_eq!(encoded.len(), 71); // Must be 71 bits (55 data + 2 type + 14 CRC)
        
        let decoded = codec.decode_cq_with_grid(&encoded).unwrap();
        assert!(decoded.contains("GW4WND"));
        assert!(decoded.contains("IO83"));
    }

    #[test]
    fn test_parity_generation() {
        let codec = CallsignCodec::new();

        let source_bits = vec![1, 0, 1, 0, 1];
        let parity = codec.generate_parity(&source_bits, 7, 3);

        assert_eq!(parity.len(), 3);
        assert_eq!(parity, vec![0, 0, 0]);
    }

    #[test]
    fn test_parity_with_large_value() {
        let codec = CallsignCodec::new();

        let source_bits = vec![1; 20];
        let parity = codec.generate_parity(&source_bits, 32749, 15);

        assert_eq!(parity.len(), 15);
    }
}
