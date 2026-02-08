use num_complex::Complex32;
use rustfft::{FftPlanner, num_traits::Zero};

pub fn fft_mag_dbfs(samples: &[f32]) -> Vec<f32> {
    // Simple: take real samples, run FFT, return magnitude in dB (not calibrated).
    let n = samples.len().max(1);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buf: Vec<Complex32> = samples.iter().map(|&x| Complex32::new(x, 0.0)).collect();
    // If caller gives empty slice, ensure buffer length is 1
    if buf.is_empty() {
        buf.push(Complex32::zero());
    }
    fft.process(&mut buf);
    buf.iter()
        .map(|c| {
            let mag = (c.re * c.re + c.im * c.im).sqrt().max(1e-12);
            20.0 * mag.log10()
        })
        .collect()
}

pub mod accumulator;
pub mod callsign;
pub mod decode;
pub mod decoder;
pub mod decoder_hybrid;
pub mod fec;
pub mod fmt1;
pub mod fmt2;
pub mod message;
pub mod msk;
pub mod rx;
