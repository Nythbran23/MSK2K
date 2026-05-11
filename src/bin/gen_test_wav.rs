// src/bin/gen_test_wav.rs
//
// Generates a synthetic FSK441 WAV file of a known message for decoder testing.
// Ports MSHV's gen441() / abc441() functions directly.
//
// Usage: cargo run --bin gen_test_wav -- "G4ABC GW4WND IO82 -17" test.wav

use std::f32::consts::TAU;

// FSK441 charset — index = nc = 16*d0 + 4*d1 + d2
const CHARSET: &[u8; 48] = b" 123456789.,?/# $ABCD FGHIJKLMNOPQRSTUVWXY 0EZ*!";

const SAMPLE_RATE: u32  = 11025;
const NSPD:        usize = 25;
const TONE_BASE:   f32   = 882.0;
const TONE_STEP:   f32   = 441.0;

fn char_to_nc(c: char) -> Option<usize> {
    CHARSET.iter().position(|&b| b as char == c)
}

fn nc_to_dits(nc: usize) -> (u8, u8, u8) {
    ((nc / 16) as u8, ((nc / 4) % 4) as u8, (nc % 4) as u8)
}

fn encode_message(msg: &str) -> Vec<u8> {
    // Convert message string to sequence of tone indices (0-3)
    let mut tones: Vec<u8> = Vec::new();
    for c in msg.chars() {
        let uc = c.to_ascii_uppercase();
        if let Some(nc) = char_to_nc(uc) {
            let (d0, d1, d2) = nc_to_dits(nc);
            tones.push(d0);
            tones.push(d1);
            tones.push(d2);
        } else {
            // Unknown char -> encode as space (nc=0)
            tones.push(0); tones.push(0); tones.push(0);
        }
    }
    tones
}

fn generate_fsk441(tones: &[u8], repeats: usize, snr_db: f32) -> Vec<f32> {
    let nspd   = NSPD;
    let dt     = 1.0_f32 / SAMPLE_RATE as f32;
    let n_dits = tones.len();
    let n_samp = n_dits * nspd;

    // Noise amplitude from SNR
    // Signal amplitude = 1.0, noise RMS = 10^(-snr/20)
    let noise_amp = 10.0_f32.powf(-snr_db / 20.0);

    let mut samples = Vec::with_capacity(n_samp * repeats + SAMPLE_RATE as usize / 2);

    // 250ms silence at start
    for _ in 0..(SAMPLE_RATE / 4) as usize {
        samples.push(rand_gauss() * noise_amp);
    }

    let mut rng_state: u64 = 12345;

    for _ in 0..repeats {
        let mut phi = 0.0f32;

        for &tone_idx in tones {
            let freq  = TONE_BASE + tone_idx as f32 * TONE_STEP;
            let dpha  = TAU * freq * dt;
            for _ in 0..nspd {
                phi += dpha;
                let signal = phi.sin();
                let noise  = lcg_gauss(&mut rng_state) * noise_amp;
                samples.push(signal + noise);
            }
        }

        // 250ms gap between repeats (simulates the 30s TX/RX cycle structure
        // compressed to just a gap for testing)
        for _ in 0..(SAMPLE_RATE / 4) as usize {
            samples.push(lcg_gauss(&mut rng_state) * noise_amp);
        }
    }

    samples
}

// Simple LCG pseudo-random Gaussian via Box-Muller
fn lcg_gauss(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let u1 = (*state >> 33) as f32 / u32::MAX as f32 + 1e-10;
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let u2 = (*state >> 33) as f32 / u32::MAX as f32;
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

fn rand_gauss() -> f32 {
    let mut s: u64 = 98765;
    lcg_gauss(&mut s)
}

fn write_wav(path: &str, samples: &[f32]) {
    let spec = hound::WavSpec {
        channels:        1,
        sample_rate:     SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format:   hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("Cannot create WAV");
    for &s in samples {
        let s16 = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(s16).unwrap();
    }
    writer.finalize().unwrap();
    println!("Written: {} ({} samples, {:.1}s)",
        path, samples.len(), samples.len() as f32 / SAMPLE_RATE as f32);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let message = if args.len() > 1 { &args[1] } else { "G4ABC GW4WND IO82 -17" };
    let out     = if args.len() > 2 { &args[2] } else { "test_fsk441.wav" };
    let snr_db: f32 = if args.len() > 3 {
        args[3].parse().unwrap_or(20.0)
    } else { 20.0 };

    println!("Message : {}", message);
    println!("SNR     : {:.0} dB", snr_db);
    println!("Output  : {}", out);

    let tones   = encode_message(message);
    println!("Dits    : {} ({} chars x 3)", tones.len(), message.len());
    println!("Tones   : {:?}", &tones[..tones.len().min(12)]);

    // Generate 3 repeats of the message (simulates a TX burst)
    let samples = generate_fsk441(&tones, 3, snr_db);
    write_wav(out, &samples);

    println!();
    println!("Now test with:");
    println!("  cargo run --bin fsk441rx -- --wav {} --verbose", out);
}
