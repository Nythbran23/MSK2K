fn main() {
    let samples: Vec<f32> = (0..256)
        .map(|i| (2.0 * std::f32::consts::PI * 10.0 * (i as f32) / 256.0).sin())
        .collect();

    let mags = msk2k_dsp::fft_mag_dbfs(&samples);

    println!("FFT bins: {}", mags.len());
    println!("First 8 mags: {:?}", &mags[..8.min(mags.len())]);
}