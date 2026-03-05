use msk2k_dsp::{fec, fmt2, msk, rx};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

fn main() {
    let snr_range = -10..2; // Focused on the critical failure boundary
    let trials_per_snr = 250;

    println!("SNR(dB) | Hard BER | Soft BER | Gain  | Avg Corr");
    println!("--------------------------------------------------");

    for snr_db in snr_range {
        // Now returns 3 values, including the average correlation
        let (hard_errors, soft_errors, avg_corr) = run_trial(snr_db as f32, trials_per_snr);

        let hard_ber = hard_errors as f32 / (trials_per_snr * 18) as f32;
        let soft_ber = soft_errors as f32 / (trials_per_snr * 18) as f32;

        let gain = if soft_ber > 0.0 {
            hard_ber / soft_ber
        } else {
            0.0
        };

        println!(
            "{:>7} | {:>8.4} | {:>8.4} | {:>4.1}x | {:>8.4}",
            snr_db, hard_ber, soft_ber, gain, avg_corr
        );
    }
}

fn run_trial(snr_db: f32, count: usize) -> (usize, usize, f32) {
    let mut rng = thread_rng();
    let mut hard_err = 0;
    let mut soft_err = 0;
    
    // 🟢 Initialize the missing tracking variables here
    let mut total_corr = 0.0f32;
    let mut sync_count = 0.0f32;

    for _ in 0..count {
        let info_bits: Vec<i32> = (0..18).map(|_| rng.gen_range(0..2)).collect();

        let sync = fmt2::SYNC_PATTERN_FORMAT2.to_vec();
        let addr = vec![0i32; 49]; 
        let poly_streams = fec::encode_format2(&info_bits);
        let mut tx_packet = fmt2::interleave_format2(&sync, &addr, &poly_streams);
        
        let mut padded_tx = vec![0i32; 10];
        padded_tx.append(&mut tx_packet);
        padded_tx.extend(vec![0i32; 10]);

        let audio = msk::modulate_48k(&padded_tx);

        let snr_linear = 10.0f32.powf(snr_db / 10.0);
        let noise_std = (0.5 / snr_linear).sqrt();
        let normal = Normal::new(0.0, noise_std).unwrap();

        let noisy_audio: Vec<f32> = audio.iter().map(|&s| s + normal.sample(&mut rng)).collect();

        let soft_bits_raw = rx::demodulate_msk_soft(&noisy_audio);
        let sync_info = rx::find_sync(&soft_bits_raw);

        if !sync_info.found {
            hard_err += 9;
            soft_err += 9;
            continue;
        }

        // 🟢 Accumulate the metrics for successfully found packets
        total_corr += sync_info.correlation;
        sync_count += 1.0;

        let start = sync_info.position as usize;
        if start + 258 > soft_bits_raw.len() {
            hard_err += 9;
            soft_err += 9;
            continue;
        }

        let pol = sync_info.polarity as f32;
        let packet_soft: Vec<f32> = soft_bits_raw[start..start + 258]
            .iter()
            .map(|&b| b * pol)
            .collect();

        // --- Path A: Current Hard Viterbi ---
        let packet_hard: Vec<i32> = packet_soft
            .iter()
            .map(|&s| if s > 0.0 { 1 } else { 0 })
            .collect();
        let (_, _, polys_hard) = fmt2::deinterleave_format2(&packet_hard);

        let mut codeword_hard = Vec::with_capacity(162);
        for i in 0..18 {
            for name in &["Pa", "Pb", "Pc", "Pd", "Pe", "Pf", "Pg", "Ph", "Pi"] {
                codeword_hard.push(polys_hard.get(*name).unwrap()[i]);
            }
        }
        let decoded_hard = fec::decode_format2(&codeword_hard);
        hard_err += count_errors(&info_bits, &decoded_hard);

        // --- Path B: Proposed Soft Viterbi ---
        let (_, _, polys_soft) = fmt2::deinterleave_format2_soft(&packet_soft);

        let mut codeword_soft = Vec::with_capacity(162);
        for i in 0..18 {
            for name in &["Pa", "Pb", "Pc", "Pd", "Pe", "Pf", "Pg", "Ph", "Pi"] {
                codeword_soft.push(polys_soft.get(*name).unwrap()[i]);
            }
        }
        
        let decoded_soft = fec::decode_format2_soft(&codeword_soft);
        soft_err += count_errors(&info_bits, &decoded_soft);
    }
    
    // 🟢 Calculate the average before returning
    let avg_corr = if sync_count > 0.0 { total_corr / sync_count } else { 0.0 };
    (hard_err, soft_err, avg_corr)
}

fn count_errors(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}