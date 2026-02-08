use std::collections::HashMap;

/// Convolutional encoder matching PSK2k documentation
const FMT1_K: usize = 13;
const FMT2_K: usize = 10;

// Polynomials as integers (matching Python)
const FMT1_POLY_A_INT: u32 = 0b1101101010001;
const FMT1_POLY_B_INT: u32 = 0b1000110111111;

const FMT2_POLYS_INT: [u32; 9] = [
    0b1111001001, // Pa
    0b1010111101, // Pb
    0b1101100111, // Pc
    0b1101010111, // Pd
    0b1111001001, // Pe
    0b1010111101, // Pf
    0b1101100111, // Pg
    0b1110111001, // Ph
    0b1010011011, // Pi
];

// Keep array versions for encoder
const FMT1_POLY_A: [i32; FMT1_K] = [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1];
const FMT1_POLY_B: [i32; FMT1_K] = [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1];

const FMT2_POLYS: [[i32; FMT2_K]; 9] = [
    [1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 1, 0, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 1, 0, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 1],
];

const FMT2_NAMES: [&str; 9] = ["Pa", "Pb", "Pc", "Pd", "Pe", "Pf", "Pg", "Ph", "Pi"];


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_format2_roundtrip_trace() {
        // Same test input as Python: alternating 1,0,1,0,...
        let info_bits = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        
        eprintln!("\n{}", "=".repeat(80));
        eprintln!("RUST FORMAT-2 ROUNDTRIP TRACE");
        eprintln!("{}", "=".repeat(80));
        eprintln!("\n🎯 TEST INPUT: {:?}", &info_bits[..]);
        
        // STEP 1: Encode
        eprintln!("\n{}", "=".repeat(80));
        eprintln!("STEP 1: TAIL-BITING CONVOLUTIONAL ENCODE");
        eprintln!("{}", "=".repeat(80));
        let poly_dict = encode_format2(&info_bits);
        
        eprintln!("\n📝 ENCODED STREAMS:");
        for name in ["Pa", "Pb", "Pc", "Pd", "Pe", "Pf", "Pg", "Ph", "Pi"] {
            if let Some(stream) = poly_dict.get(name) {
                eprintln!("   {}: {:?}", name, stream);
            }
        }
        
        // STEP 2: Build time-major codeword (as done in decode.rs)
        eprintln!("\n{}", "=".repeat(80));
        eprintln!("STEP 2: BUILD TIME-MAJOR CODEWORD");
        eprintln!("{}", "=".repeat(80));
        let mut codeword_time_major = Vec::with_capacity(162);
        for i in 0..18 {
            for name in ["Pa", "Pb", "Pc", "Pd", "Pe", "Pf", "Pg", "Ph", "Pi"] {
                codeword_time_major.push(poly_dict[name][i]);
            }
        }
        
        eprintln!("🔢 TIME-MAJOR CODEWORD: {} bits", codeword_time_major.len());
        eprintln!("   First 27 bits (3 timesteps): {:?}", &codeword_time_major[0..27]);
        
        // STEP 3: Decode
        eprintln!("\n{}", "=".repeat(80));
        eprintln!("STEP 3: VITERBI DECODE");
        eprintln!("{}", "=".repeat(80));
        let decoded = decode_format2(&codeword_time_major);
        
        eprintln!("📤 DECODED: {:?}", &decoded);
        
        // STEP 4: Verification
        eprintln!("\n{}", "=".repeat(80));
        eprintln!("VERIFICATION");
        eprintln!("{}", "=".repeat(80));
        eprintln!("Original:  {:?}", &info_bits[..]);
        eprintln!("Decoded:   {:?}", &decoded);
        
        let mut mismatches = Vec::new();
        for i in 0..18 {
            if decoded[i] != info_bits[i] {
                mismatches.push(i);
            }
        }
        
        if mismatches.is_empty() {
            eprintln!("\n✅ PERFECT MATCH!");
        } else {
            eprintln!("\n❌ MISMATCHES at positions: {:?}", mismatches);
            for i in &mismatches {
                eprintln!("   Position {}: expected {}, got {}", i, info_bits[*i], decoded[*i]);
            }
        }
        
        eprintln!("\n{}", "=".repeat(80));
        eprintln!("TEST COMPLETE");
        eprintln!("{}", "=".repeat(80));
        
        // Assert for test framework
        assert_eq!(decoded.len(), 18, "Decoded length should be 18");
        for i in 0..18 {
            assert_eq!(decoded[i], info_bits[i], "Mismatch at position {}", i);
        }
    } // <- This closes the function
} // <- This closes mod tests
    


fn convolve_polynomial(info_bits: &[i32], poly: &[i32]) -> Vec<i32> {
    let n = info_bits.len();
    let k = poly.len();
    let out_len = n + k - 1;
    let mut out = vec![0i32; out_len];

    for t in 0..out_len {
        let i_min = if t + 1 >= k { t + 1 - k } else { 0 };
        let i_max = if t < n { t } else { n - 1 };
        let mut acc = 0i32;
        for i in i_min..=i_max {
            let tap = poly[t - i];
            if tap != 0 {
                acc ^= info_bits[i] & 1;
            }
        }
        out[t] = acc & 1;
    }
    out
}

pub fn encode_format1(info_bits: &[i32]) -> (Vec<i32>, Vec<i32>) {
    if info_bits.len() != 71 {
        panic!("Format1 requires 71 info bits, got {}", info_bits.len());
    }
    let tail_len = 2 * (FMT1_K - 1);
    let mut info_with_tail = Vec::with_capacity(info_bits.len() + tail_len);
    info_with_tail.extend(info_bits.iter().map(|x| x & 1));
    info_with_tail.extend(std::iter::repeat(0i32).take(tail_len));

    let enc1_full = convolve_polynomial(&info_with_tail, &FMT1_POLY_A);
    let enc2_full = convolve_polynomial(&info_with_tail, &FMT1_POLY_B);

    let poly1 = enc1_full[..83].to_vec();
    let poly2 = enc2_full[..83].to_vec();
    (poly1, poly2)
}

/// AMENDED: Tail-biting Format 2 Encoder (Rate 1/9)
/// This correctly initializes the state using the last bits of the message.
pub fn encode_format2(info_bits: &[i32]) -> HashMap<String, Vec<i32>> {
    if info_bits.len() != 18 {
        panic!("Format2 requires 18 info bits, got {}", info_bits.len());
    }

    let k_minus_1 = FMT2_K - 1; // 9 bits
    let mut wrapped = vec![0i32; 18 + k_minus_1];

    // Tail-biting: Wrap the end of the message to the start
    wrapped[..k_minus_1].copy_from_slice(&info_bits[18 - k_minus_1..]);
    wrapped[k_minus_1..].copy_from_slice(info_bits);

    let mut out_map: HashMap<String, Vec<i32>> = HashMap::new();

    for (idx, name) in FMT2_NAMES.iter().enumerate() {
        let poly = &FMT2_POLYS[idx];
        let mut stream = Vec::with_capacity(18);

        for i in 0..18 {
            let mut acc = 0i32;
            for j in 0..FMT2_K {
                if poly[j] == 1 {
                    // Convolution using the tail-biting buffer
                    acc ^= wrapped[i + k_minus_1 - j];
                }
            }
            stream.push(acc & 1);
        }
        out_map.insert(name.to_string(), stream);
    }
    out_map
}

/// Helper for Transmitter: returns the 162 bits flattened and interleaved
/// as required for the Format 2 packet structure.
pub fn encode_format2_flat(info_bits: &[i32]) -> Vec<i32> {
    let streams = encode_format2(info_bits);
    let mut flat = Vec::with_capacity(162);

    // Interleave bits: Pa[0], Pb[0], Pc[0]... Pi[0], Pa[1]...
    for i in 0..18 {
        for name in &FMT2_NAMES {
            if let Some(s) = streams.get(*name) {
                flat.push(s[i]);
            }
        }
    }
    flat
}

// ============================================================================
// VITERBI DECODER (Unchanged per request)
// ============================================================================

const FMT1_NUM_STATES: usize = 1 << (FMT1_K - 1);
const FMT2_NUM_STATES: usize = 1 << (FMT2_K - 1);

fn next_state_and_output(state: usize, input_bit: i32, k: usize, poly_int: u32) -> (usize, i32) {
    let new_state = ((input_bit as usize & 1) << (k - 2)) | (state >> 1);
    let shift_reg = ((input_bit as usize & 1) << (k - 1)) | state;
    let and_result = (shift_reg as u32) & poly_int;
    let output = and_result.count_ones() as i32 & 1;
    (new_state, output)
}

fn hamming_distance(a: &[i32], b: &[i32]) -> u32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y) as u32).sum()
}

fn viterbi_decode_format1_internal(poly1: &[i32], poly2: &[i32]) -> Vec<i32> {
    if poly1.len() != 83 || poly2.len() != 83 {
        panic!("Format1 Viterbi expects 83 bits from each poly");
    }

    let num_outputs = 83;
    let num_states = FMT1_NUM_STATES;
    let mut path_metrics = vec![vec![u32::MAX; num_states]; num_outputs + 1];
    let mut survivors = vec![vec![(0usize, 0i32); num_states]; num_outputs];
    path_metrics[0][0] = 0;

    for t in 0..num_outputs {
        let received = [poly1[t], poly2[t]];
        for state in 0..num_states {
            if path_metrics[t][state] == u32::MAX { continue; }
            for input_bit in 0..2 {
                let (next_state_a, out_a) = next_state_and_output(state, input_bit, FMT1_K, FMT1_POLY_A_INT);
                let (_next_state_b, out_b) = next_state_and_output(state, input_bit, FMT1_K, FMT1_POLY_B_INT);
                let next_state = next_state_a;
                let expected = [out_a, out_b];
                let branch_metric = hamming_distance(&expected, &received);
                let new_metric = path_metrics[t][state].saturating_add(branch_metric);
                if new_metric < path_metrics[t + 1][next_state] {
                    path_metrics[t + 1][next_state] = new_metric;
                    survivors[t][next_state] = (state, input_bit);
                }
            }
        }
    }

    let mut info_bits = Vec::with_capacity(num_outputs);
    let mut state = 0usize;
    for t in (0..num_outputs).rev() {
        let (prev_state, input_bit) = survivors[t][state];
        info_bits.push(input_bit);
        state = prev_state;
    }
    info_bits.reverse();
    info_bits.truncate(71);
    info_bits
}

fn viterbi_decode_format2_internal(encoded_streams: &[Vec<i32>; 9]) -> Vec<i32> {
    for (i, stream) in encoded_streams.iter().enumerate() {
        if stream.len() != 18 { panic!("Format2 Viterbi expects 18 bits from poly {}", i); }
    }

    let num_outputs = 18;
    let num_states = FMT2_NUM_STATES;
    let total_steps = num_outputs * 2; 
    let mut path_metrics = vec![vec![u32::MAX; num_states]; total_steps + 1];
    let mut survivors = vec![vec![(0usize, 0i32); num_states]; total_steps];

    for state in 0..num_states { path_metrics[0][state] = 0; }

    for wrap in 0..2 {
        for step in 0..num_outputs {
            let actual_step = wrap * num_outputs + step;
            let mut received = [0i32; 9];
            for i in 0..9 { received[i] = encoded_streams[i][step]; }

            for state in 0..num_states {
                if path_metrics[actual_step][state] == u32::MAX { continue; }
                for input_bit in 0..2 {
                    let mut expected = [0i32; 9];
                    let mut next_state = 0;
                    for i in 0..9 {
                        let (ns, out) = next_state_and_output(state, input_bit, FMT2_K, FMT2_POLYS_INT[i]);
                        expected[i] = out;
                        next_state = ns;
                    }
                    let branch_metric = hamming_distance(&expected, &received);
                    let new_metric = path_metrics[actual_step][state].saturating_add(branch_metric);
                    if new_metric < path_metrics[actual_step + 1][next_state] {
                        path_metrics[actual_step + 1][next_state] = new_metric;
                        survivors[actual_step][next_state] = (state, input_bit);
                    }
                }
            }
        }
    }

    let mut best_state = 0;
    let mut best_metric = u32::MAX;
    for state in 0..num_states {
        if path_metrics[total_steps][state] < best_metric {
            best_metric = path_metrics[total_steps][state];
            best_state = state;
        }
    }

    let mut info_bits = Vec::with_capacity(num_outputs);
    let mut state = best_state;
    for t in (num_outputs..total_steps).rev() {
        let (prev_state, input_bit) = survivors[t][state];
        info_bits.push(input_bit);
        state = prev_state;
    }
    info_bits.reverse();
    info_bits
}

pub fn decode_format1(codeword: &[i32]) -> Vec<i32> {
    if codeword.len() != 166 { panic!("Format1 decode requires 166-bit codeword"); }
    let poly1 = &codeword[0..83];
    let poly2 = &codeword[83..166];
    viterbi_decode_format1_internal(poly1, poly2)
}

pub fn decode_format2(codeword: &[i32]) -> Vec<i32> {
    if codeword.len() != 162 { panic!("Format2 decode requires 162-bit codeword"); }
    
    // Codeword comes from decode.rs in TIME-MAJOR format:
    //   [Pa[0], Pb[0], ..., Pi[0], Pa[1], Pb[1], ..., Pi[1], ..., Pa[17], ..., Pi[17]]
    // 
    // Viterbi expects STREAM-MAJOR format:
    //   [[Pa[0..17]], [Pb[0..17]], ..., [Pi[0..17]]]
    // 
    // De-interleave: extract every 9th bit for each stream
    
    let mut streams: [Vec<i32>; 9] = [
        Vec::with_capacity(18), Vec::with_capacity(18), Vec::with_capacity(18),
        Vec::with_capacity(18), Vec::with_capacity(18), Vec::with_capacity(18),
        Vec::with_capacity(18), Vec::with_capacity(18), Vec::with_capacity(18),
    ];
    
    // Extract each stream: stream[i] gets every 9th bit starting at offset i
    for stream_idx in 0..9 {
        for time_step in 0..18 {
            let bit_position = time_step * 9 + stream_idx;
            streams[stream_idx].push(codeword[bit_position]);
        }
    }
    
    viterbi_decode_format2_internal(&streams)
}