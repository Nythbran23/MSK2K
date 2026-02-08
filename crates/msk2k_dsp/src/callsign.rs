// crates/msk2k_dsp/src/callsign.rs
//
// Callsign encoding/decoding for MSK2K
// Based on PSK2kSourceEncoder from Python implementation

use std::collections::HashMap;

/// Callsign encoder/decoder for MSK2K protocol
pub struct CallsignCodec {
    /// Base-37 alphabet: /ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
    alphabet: Vec<char>,
    /// Prime numbers for parity generation (prime -> num_bits)
    primes: HashMap<u32, usize>,
    /// Parity selection by callsign length
    parity_selection: HashMap<usize, Vec<u32>>,
    /// Length codes (4 bits for 3-9 chars, 1 bit for 10 chars)
    length_codes: HashMap<usize, Vec<i32>>,
}

impl CallsignCodec {
    pub fn new() -> Self {
        let alphabet = "/ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".chars().collect();

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
            primes,
            parity_selection,
            length_codes,
        }
    }

    /// Minimal plausibility filter for decoded callsigns:
    /// - max 2 slashes
    /// - cannot start or end with '/'
    /// - cannot contain "//"
    /// - only allow A–Z 0–9 /
    fn is_plausible_callsign_minimal(cs: &str) -> bool {
        let cs = cs.trim();
        if cs.len() < 3 || cs.len() > 10 {
            return false;
        }

        // Only allow A–Z, 0–9 and '/'
        if !cs
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '/')
        {
            return false;
        }

        let slash_count = cs.chars().filter(|&c| c == '/').count();
        if slash_count > 2 {
            return false;
        }

        if cs.starts_with('/') || cs.ends_with('/') {
            return false;
        }

        if cs.contains("//") {
            return false;
        }

        true
    }

    /// Encode callsign to 54-bit representation
    pub fn encode_callsign(&self, callsign: &str) -> Result<Vec<i32>, String> {
        let call = callsign.trim().to_uppercase();
        let call_len = call.len();

        if call_len < 3 || call_len > 10 {
            return Err(format!(
                "Callsign length must be 3-10 characters, got {}",
                call_len
            ));
        }

        // Convert callsign to base-37 number (reject invalid characters)
        let mut z: u64 = 0;
        for ch in call.chars() {
            let digit = match self.alphabet.iter().position(|&c| c == ch) {
                Some(d) => d,
                None => return Err(format!("Invalid callsign character: '{}'", ch)),
            };
            z = z * 37 + digit as u64;
        }

        if call_len == 10 {
            // Special case: split into two parts
            return self.encode_10char_callsign(&call);
        }

        // Normal case: encode as single number
        let bit_len = match call_len {
            3 => 16,
            4 => 21,
            5 => 27,
            6 => 32,
            7 => 37,
            8 => 42,
            9 => 47,
            _ => return Err(format!("Invalid callsign length: {}", call_len)),
        };

        let mut total_bits = self.int_to_bits(z, bit_len);

        // Generate parity bits
        if let Some(primes) = self.parity_selection.get(&call_len) {
            for &prime in primes {
                let remainder = (z % prime as u64) as u32;
                let num_bits = self.primes[&prime];
                total_bits.extend(self.int_to_bits(remainder as u64, num_bits));
            }
        }

        // Pad to 50 bits (length code will go at 50-53)
        while total_bits.len() < 50 {
            total_bits.push(0);
        }
        total_bits.truncate(50);

        // Add length code at positions 50-53
        total_bits.extend(&self.length_codes[&call_len]);

        Ok(total_bits[..54].to_vec())
    }

    fn encode_10char_callsign(&self, call: &str) -> Result<Vec<i32>, String> {
        if call.len() != 10 {
            return Err("Expected 10-character callsign".to_string());
        }

        let call1 = &call[..6];
        let call2 = &call[6..];

        let mut z1: u64 = 0;
        for ch in call1.chars() {
            let digit = match self.alphabet.iter().position(|&c| c == ch) {
                Some(d) => d,
                None => return Err(format!("Invalid callsign character: '{}'", ch)),
            };
            z1 = z1 * 37 + digit as u64;
        }

        let mut z2: u64 = 0;
        for ch in call2.chars() {
            let digit = match self.alphabet.iter().position(|&c| c == ch) {
                Some(d) => d,
                None => return Err(format!("Invalid callsign character: '{}'", ch)),
            };
            z2 = z2 * 37 + digit as u64;
        }

        // 32 bits + 21 bits + 1 length bit = 54 bits
        let mut bits = self.int_to_bits(z1, 32);
        bits.extend(self.int_to_bits(z2, 21));
        bits.push(0); // Length code for 10 chars

        Ok(bits)
    }

    /// Decode 54-bit array back to callsign
    pub fn decode_callsign(&self, bits: &[i32]) -> String {
        if bits.len() != 54 {
            return "ERROR".to_string();
        }

        // Check if 10-character callsign (bit 53 == 0)
        if bits[53] == 0 {
            return self.decode_10char_callsign(bits);
        }

        // Determine length from 4-bit code at 50-53
        let length_code: Vec<i32> = bits[50..54].to_vec();

        let mut call_len = None;
        for (&length, code) in &self.length_codes {
            if code == &length_code {
                call_len = Some(length);
                break;
            }
        }

        let call_len = match call_len {
            Some(len) => len,
            None => return "ERROR".to_string(),
        };

        // Extract callsign bits
        let bit_len = match call_len {
            3 => 16,
            4 => 21,
            5 => 27,
            6 => 32,
            7 => 37,
            8 => 42,
            9 => 47,
            _ => return "ERROR".to_string(),
        };

        let call_bits = &bits[..bit_len];
        let z = self.bits_to_int(call_bits);

        let decoded = self.base37_decode(z, call_len).trim().to_string();

        // Apply minimal plausibility check (reject ////// etc)
        if !Self::is_plausible_callsign_minimal(&decoded) {
            return "ERROR".to_string();
        }

        decoded
    }

    fn decode_10char_callsign(&self, bits: &[i32]) -> String {
        let bits1 = &bits[..32];
        let bits2 = &bits[32..53];

        let z1 = self.bits_to_int(bits1);
        let z2 = self.bits_to_int(bits2);

        let call1 = self.base37_decode(z1, 6);
        let call2 = self.base37_decode(z2, 4);

        let decoded = format!("{}{}", call1, call2).trim().to_string();

        // Apply minimal plausibility check
        if !Self::is_plausible_callsign_minimal(&decoded) {
            return "ERROR".to_string();
        }

        decoded
    }

    /// Generate 49-bit private address from callsign
    /// Per DJ5HG spec Section 7.2: erase the last 5 bits of the 54-bit code
    /// (verified against spec example: DJ5HG address = first 49 bits of 54-bit code)
    pub fn generate_private_address(&self, callsign: &str) -> Result<Vec<i32>, String> {
        let full_code = self.encode_callsign(callsign)?;
        Ok(full_code[..49].to_vec())
    }

    /// Decode callsign from 49-bit private address
    /// Per DJ5HG spec Section 7.2: the address is the first 49 bits of the 54-bit code.
    /// The last 5 bits were erased. We reconstruct by trying all possible length codes.
    pub fn decode_private_address(&self, addr_bits: &[i32]) -> String {
        if addr_bits.len() < 49 {
            log::warn!("[CALL] decode_private_address: addr_bits too short: {}", addr_bits.len());
            return "ERROR".to_string();
        }

        // The 54-bit code is: addr[0..49] + erased[49..54]
        // The erased bits include part of the parity and the length code.
        // We need to try different length codes to find a valid callsign.
        //
        // Length codes (bits 50-53 of 54-bit code):
        //   3: 0111  4: 1011  5: 0011  6: 1101  7: 0101  8: 1001  9: 0001  10: 0 (special)
        //
        // Since bits 49-53 were erased, bit 49 is unknown parity, bits 50-53 are length code.
        
        // Try each possible length code
        let length_code_options: &[(usize, [i32; 4])] = &[
            (3, [0, 1, 1, 1]),
            (4, [1, 0, 1, 1]),
            (5, [0, 0, 1, 1]),
            (6, [1, 1, 0, 1]),
            (7, [0, 1, 0, 1]),
            (8, [1, 0, 0, 1]),
            (9, [0, 0, 0, 1]),
        ];

        for &(_call_len, ref lcode) in length_code_options {
            // Try with bit 49 = 0 and = 1
            for bit49 in [0i32, 1i32] {
                let mut reconstructed = vec![0i32; 54];
                reconstructed[..49].copy_from_slice(&addr_bits[..49]);
                reconstructed[49] = bit49;
                reconstructed[50] = lcode[0];
                reconstructed[51] = lcode[1];
                reconstructed[52] = lcode[2];
                reconstructed[53] = lcode[3];

                let result = self.decode_callsign(&reconstructed);
                if result != "ERROR" {
                    // Verify: re-encode and check first 49 bits match
                    if let Ok(re_encoded) = self.encode_callsign(&result) {
                        if re_encoded.len() >= 49 && re_encoded[..49] == addr_bits[..49] {
                            return result;
                        }
                    }
                }
            }
        }

        // Also try 10-char decode (length code bit 53 = 0)
        let mut reconstructed = vec![0i32; 54];
        reconstructed[..49].copy_from_slice(&addr_bits[..49]);
        reconstructed[53] = 0;
        let result = self.decode_callsign(&reconstructed);
        if result != "ERROR" {
            if let Ok(re_encoded) = self.encode_callsign(&result) {
                if re_encoded.len() >= 49 && re_encoded[..49] == addr_bits[..49] {
                    return result;
                }
            }
        }

        "UNKNOWN".to_string()
    }
    /// Generate parity bits using residual code
    ///
    /// This matches Python's generate_parity() function exactly.
    /// Used for both Format-1 and Format-2 parity generation.
    pub fn generate_parity(&self, source_bits: &[i32], r: u32, num_bits: usize) -> Vec<i32> {
        let value = self.bits_to_int(source_bits);
        let remainder = (value % r as u64) as u64;
        self.int_to_bits(remainder, num_bits)
    }

    // Helper functions

    fn int_to_bits(&self, value: u64, num_bits: usize) -> Vec<i32> {
        let mut bits = Vec::with_capacity(num_bits);
        for i in (0..num_bits).rev() {
            bits.push(((value >> i) & 1) as i32);
        }
        bits
    }

    fn bits_to_int(&self, bits: &[i32]) -> u64 {
        let mut value: u64 = 0;
        for &bit in bits {
            value = (value << 1) | (bit as u64);
        }
        value
    }

    fn base37_decode(&self, mut value: u64, length: usize) -> String {
        let mut chars = Vec::with_capacity(length);
        for _ in 0..length {
            let idx = (value % 37) as usize;
            chars.push(self.alphabet[idx]);
            value /= 37;
        }
        chars.reverse();
        chars.into_iter().collect()
    }
}

impl Default for CallsignCodec {
    fn default() -> Self {
        Self::new()
    }
}

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
    fn test_parity_generation() {
        let codec = CallsignCodec::new();

        // Test case: binary value 21 (0b10101)
        let source_bits = vec![1, 0, 1, 0, 1];
        let parity = codec.generate_parity(&source_bits, 7, 3);

        assert_eq!(parity.len(), 3);
        // 21 % 7 = 0, so parity should be [0, 0, 0]
        assert_eq!(parity, vec![0, 0, 0]);
    }

    #[test]
    fn test_parity_with_large_value() {
        let codec = CallsignCodec::new();

        // Test with 32749 (Format-2 parity prime)
        let source_bits = vec![1; 20]; // Some arbitrary bits
        let parity = codec.generate_parity(&source_bits, 32749, 15);

        assert_eq!(parity.len(), 15);
    }
}