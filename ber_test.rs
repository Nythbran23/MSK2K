use msk2k_dsp::{fec, msk, rx}; // Uses your project's modules
use rand::prelude::*;
use rand_distr::{Normal, Distribution};

fn main() {
    let snr_range = -5..10; // SNR in dB to test
    let trials_per_snr = 1000;

    println!("SNR(dB) | Hard BER | Soft BER | Gain");
    println!("------------------------------------");

    for snr_db in snr_range {
        let (hard_errors, soft_errors) = run_trial(snr_db as f32, trials_per_snr);
        
        let hard_ber = hard_errors as f32 / (trials_per_snr * 18) as f32;
        let soft_ber = soft_errors as f32 / (trials_per_snr * 18) as f32;
        
        println!("{:>7} | {:>8.4} | {:>8.4} | {:.1}x", 
                 snr_db, hard_ber, soft_ber, hard_ber / soft_ber.max(1e-6));
    }
}

fn run_trial(snr_db: f32, count: usize) -> (usize, usize) {
    let mut rng = thread_rng();
    let mut hard_err = 0;
    let mut soft_err = 0;

    // 1. Generate Target Signal (18-bit Format 2 message)
    let info_bits: Vec<i32> = (0..18).map(|_| rng.gen_range(0..2)).collect();
    let encoded = fec::encode_format2_flat(&info_bits);
    let audio = msk::modulate_48k(&encoded);

    // 2. Add AWGN (Additive White Gaussian Noise)
    let snr_linear = 10.0f32.powf(snr_db / 10.0);
    let noise_std = (1.0 / (2.0 * snr_linear)).sqrt();
    let normal = Normal::new(0.0, noise_std).unwrap();

    for _ in 0..count {
        let noisy_audio: Vec<f32> = audio.iter()
            .map(|&s| s + normal.sample(&mut rng))
            .collect();

        // 3. Extract Soft Bits from Demodulator
        let soft_bits = rx::demodulate_msk_soft(&noisy_audio);

        // Path A: Current Hard Viterbi
        let hard_bits: Vec<i32> = soft_bits.iter().map(|&s| if s > 0.0 {1} else {0}).collect();
        let decoded_hard = fec::decode_format2(&hard_bits);
        hard_err += count_errors(&info_bits, &decoded_hard);

        // Path B: Proposed Soft Viterbi
        let decoded_soft = fec::viterbi_decode_format2_soft(&soft_bits); // You'll implement this
        soft_err += count_errors(&info_bits, &decoded_soft);
    }
    (hard_err, soft_err)
}

fn count_errors(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}