use std::collections::HashMap;

const FMT1_K: usize = 13;
const FMT2_K: usize = 10;

const FMT1_POLY_A_INT: u32 = 0b1101101010001;
const FMT1_POLY_B_INT: u32 = 0b1000110111111;

const FMT2_POLYS_INT: [u32; 9] = [
    0b1111001001, 0b1010111101, 0b1101100111, 0b1101010111, 0b1111001001,
    0b1010111101, 0b1101100111, 0b1110111001, 0b1010011011,
];

const FMT1_POLY_A: [i32; FMT1_K] = [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1];
const FMT1_POLY_B: [i32; FMT1_K] = [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1];

const FMT2_POLYS: [[i32; FMT2_K]; 9] = [
    [1, 1, 1, 1, 0, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 1, 0, 1, 1, 0, 0, 1, 1, 1], [1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 1, 0, 1, 1, 0, 0, 1, 1, 1], [1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 1],
];

const FMT2_NAMES: [&str; 9] = ["Pa", "Pb", "Pc", "Pd", "Pe", "Pf", "Pg", "Ph", "Pi"];

fn convolve_polynomial(info_bits: &[i32], poly: &[i32]) -> Vec<i32> {
    let n = info_bits.len(); let k = poly.len();
    let out_len = n + k - 1; let mut out = vec![0i32; out_len];
    for t in 0..out_len {
        let i_min = if t + 1 >= k { t + 1 - k } else { 0 };
        let i_max = if t < n { t } else { n - 1 };
        let mut acc = 0i32;
        for i in i_min..=i_max {
            let tap = poly[t - i];
            if tap != 0 { acc ^= info_bits[i] & 1; }
        }
        out[t] = acc & 1;
    }
    out
}

pub fn encode_format1(info_bits: &[i32]) -> (Vec<i32>, Vec<i32>) {
    if info_bits.len() != 71 { panic!("Format1 requires 71 info bits"); }
    let tail_len = 2 * (FMT1_K - 1);
    let mut info_with_tail = Vec::with_capacity(info_bits.len() + tail_len);
    info_with_tail.extend(info_bits.iter().map(|x| x & 1));
    info_with_tail.extend(std::iter::repeat(0i32).take(tail_len));

    let enc1_full = convolve_polynomial(&info_with_tail, &FMT1_POLY_A);
    let enc2_full = convolve_polynomial(&info_with_tail, &FMT1_POLY_B);
    (enc1_full[..83].to_vec(), enc2_full[..83].to_vec())
}

pub fn encode_format2(info_bits: &[i32]) -> HashMap<String, Vec<i32>> {
    if info_bits.len() != 18 { panic!("Format2 requires 18 info bits"); }
    let k_minus_1 = FMT2_K - 1; 
    let mut wrapped = vec![0i32; 18 + k_minus_1];
    wrapped[..k_minus_1].copy_from_slice(&info_bits[18 - k_minus_1..]);
    wrapped[k_minus_1..].copy_from_slice(info_bits);

    let mut out_map: HashMap<String, Vec<i32>> = HashMap::new();
    for (idx, name) in FMT2_NAMES.iter().enumerate() {
        let poly = &FMT2_POLYS[idx];
        let mut stream = Vec::with_capacity(18);
        for i in 0..18 {
            let mut acc = 0i32;
            for j in 0..FMT2_K {
                if poly[j] == 1 { acc ^= wrapped[i + k_minus_1 - j]; }
            }
            stream.push(acc & 1);
        }
        out_map.insert(name.to_string(), stream);
    }
    out_map
}

pub fn encode_format2_flat(info_bits: &[i32]) -> Vec<i32> {
    let streams = encode_format2(info_bits);
    let mut flat = Vec::with_capacity(162);
    for i in 0..18 {
        for name in &FMT2_NAMES {
            if let Some(s) = streams.get(*name) { flat.push(s[i]); }
        }
    }
    flat
}

// ============================================================================
// VITERBI DECODERS
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

// 🟢 NEW: Euclidean Soft Metric
fn soft_metric(received: f32, expected_bit: i32) -> f32 {
    let expected_val = if expected_bit == 1 { 1.0 } else { -1.0 };
    let diff = received - expected_val;
    diff * diff
}

fn viterbi_decode_format1_internal(poly1: &[i32], poly2: &[i32]) -> Vec<i32> {
    if poly1.len() != 83 || poly2.len() != 83 { panic!("Format1 Viterbi expects 83 bits"); }
    let num_outputs = 83; let num_states = FMT1_NUM_STATES;
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
                let expected = [out_a, out_b];
                let branch_metric = hamming_distance(&expected, &received);
                let new_metric = path_metrics[t][state].saturating_add(branch_metric);
                if new_metric < path_metrics[t + 1][next_state_a] {
                    path_metrics[t + 1][next_state_a] = new_metric;
                    survivors[t][next_state_a] = (state, input_bit);
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
    info_bits.reverse(); info_bits.truncate(71); info_bits
}

// 🟢 NEW: Soft Viterbi format 1
fn viterbi_decode_format1_soft(poly1: &[f32], poly2: &[f32]) -> Vec<i32> {
    let num_outputs = 83; let num_states = FMT1_NUM_STATES;
    let mut path_metrics = vec![vec![f32::MAX; num_states]; num_outputs + 1];
    let mut survivors = vec![vec![(0usize, 0i32); num_states]; num_outputs];
    path_metrics[0][0] = 0.0;

    for t in 0..num_outputs {
        let received_1 = poly1[t];
        let received_2 = poly2[t];
        for state in 0..num_states {
            if path_metrics[t][state] == f32::MAX { continue; }
            for input_bit in 0..2 {
                let (next_state_a, out_a) = next_state_and_output(state, input_bit, FMT1_K, FMT1_POLY_A_INT);
                let (_next_state_b, out_b) = next_state_and_output(state, input_bit, FMT1_K, FMT1_POLY_B_INT);
                
                let bm1 = soft_metric(received_1, out_a);
                let bm2 = soft_metric(received_2, out_b);
                let branch_metric = bm1 + bm2;
                
                let new_metric = path_metrics[t][state] + branch_metric;
                if new_metric < path_metrics[t + 1][next_state_a] {
                    path_metrics[t + 1][next_state_a] = new_metric;
                    survivors[t][next_state_a] = (state, input_bit);
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
    info_bits.reverse(); info_bits.truncate(71); info_bits
}

fn viterbi_decode_format2_internal(encoded_streams: &[Vec<i32>; 9]) -> Vec<i32> {
    let num_outputs = 18; let num_states = FMT2_NUM_STATES; let total_steps = num_outputs * 2; 
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
                        expected[i] = out; next_state = ns;
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
    let mut best_state = 0; let mut best_metric = u32::MAX;
    for state in 0..num_states {
        if path_metrics[total_steps][state] < best_metric {
            best_metric = path_metrics[total_steps][state]; best_state = state;
        }
    }
    let mut info_bits = Vec::with_capacity(num_outputs);
    let mut state = best_state;
    for t in (num_outputs..total_steps).rev() {
        let (prev_state, input_bit) = survivors[t][state];
        info_bits.push(input_bit); state = prev_state;
    }
    info_bits.reverse(); info_bits
}

// 🟢 NEW: Soft Viterbi format 2
fn viterbi_decode_format2_soft(encoded_streams: &[Vec<f32>; 9]) -> Vec<i32> {
    let num_outputs = 18; let num_states = FMT2_NUM_STATES; let total_steps = num_outputs * 2; 
    let mut path_metrics = vec![vec![f32::MAX; num_states]; total_steps + 1];
    let mut survivors = vec![vec![(0usize, 0i32); num_states]; total_steps];
    for state in 0..num_states { path_metrics[0][state] = 0.0; }

    for wrap in 0..2 {
        for step in 0..num_outputs {
            let actual_step = wrap * num_outputs + step;
            let mut received = [0.0f32; 9];
            for i in 0..9 { received[i] = encoded_streams[i][step]; }

            for state in 0..num_states {
                if path_metrics[actual_step][state] == f32::MAX { continue; }
                for input_bit in 0..2 {
                    let mut next_state = 0;
                    let mut branch_metric = 0.0f32;
                    for i in 0..9 {
                        let (ns, out) = next_state_and_output(state, input_bit, FMT2_K, FMT2_POLYS_INT[i]);
                        branch_metric += soft_metric(received[i], out);
                        next_state = ns;
                    }
                    let new_metric = path_metrics[actual_step][state] + branch_metric;
                    if new_metric < path_metrics[actual_step + 1][next_state] {
                        path_metrics[actual_step + 1][next_state] = new_metric;
                        survivors[actual_step][next_state] = (state, input_bit);
                    }
                }
            }
        }
    }
    let mut best_state = 0; let mut best_metric = f32::MAX;
    for state in 0..num_states {
        if path_metrics[total_steps][state] < best_metric {
            best_metric = path_metrics[total_steps][state]; best_state = state;
        }
    }
    let mut info_bits = Vec::with_capacity(num_outputs);
    let mut state = best_state;
    for t in (num_outputs..total_steps).rev() {
        let (prev_state, input_bit) = survivors[t][state];
        info_bits.push(input_bit); state = prev_state;
    }
    info_bits.reverse(); info_bits
}

pub fn decode_format1(codeword: &[i32]) -> Vec<i32> {
    if codeword.len() != 166 { panic!("Format1 decode requires 166-bit codeword"); }
    viterbi_decode_format1_internal(&codeword[0..83], &codeword[83..166])
}

// 🟢 NEW: Entrypoint for Format 1 Soft Decoding
pub fn decode_format1_soft(codeword: &[f32]) -> Vec<i32> {
    if codeword.len() != 166 { panic!("Format1 soft decode requires 166-bit codeword"); }
    viterbi_decode_format1_soft(&codeword[0..83], &codeword[83..166])
}

pub fn decode_format2(codeword: &[i32]) -> Vec<i32> {
    if codeword.len() != 162 { panic!("Format2 decode requires 162-bit codeword"); }
    let mut streams: [Vec<i32>; 9] = [
        Vec::with_capacity(18), Vec::with_capacity(18), Vec::with_capacity(18),
        Vec::with_capacity(18), Vec::with_capacity(18), Vec::with_capacity(18),
        Vec::with_capacity(18), Vec::with_capacity(18), Vec::with_capacity(18),
    ];
    for stream_idx in 0..9 {
        for time_step in 0..18 {
            let bit_position = time_step * 9 + stream_idx;
            streams[stream_idx].push(codeword[bit_position]);
        }
    }
    viterbi_decode_format2_internal(&streams)
}

// 🟢 NEW: Entrypoint for Format 2 Soft Decoding
pub fn decode_format2_soft(codeword: &[f32]) -> Vec<i32> {
    if codeword.len() != 162 { panic!("Format2 soft decode requires 162-bit codeword"); }
    let mut streams: [Vec<f32>; 9] = [
        Vec::with_capacity(18), Vec::with_capacity(18), Vec::with_capacity(18),
        Vec::with_capacity(18), Vec::with_capacity(18), Vec::with_capacity(18),
        Vec::with_capacity(18), Vec::with_capacity(18), Vec::with_capacity(18),
    ];
    for stream_idx in 0..9 {
        for time_step in 0..18 {
            let bit_position = time_step * 9 + stream_idx;
            streams[stream_idx].push(codeword[bit_position]);
        }
    }
    viterbi_decode_format2_soft(&streams)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_format2_roundtrip_trace() {
        let info_bits = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let poly_dict = encode_format2(&info_bits);
        let mut codeword_time_major = Vec::with_capacity(162);
        for i in 0..18 {
            for name in ["Pa", "Pb", "Pc", "Pd", "Pe", "Pf", "Pg", "Ph", "Pi"] {
                codeword_time_major.push(poly_dict[name][i]);
            }
        }
        let decoded = decode_format2(&codeword_time_major);
        for i in 0..18 {
            assert_eq!(decoded[i], info_bits[i], "Mismatch at position {}", i);
        }
    }
}