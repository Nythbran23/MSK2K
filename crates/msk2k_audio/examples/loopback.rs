//! Audio loopback test
//!
//! Captures audio from the default input device and plays it back
//! through the default output device with optional resampling.
//!
//! Usage: cargo run --example loopback [--resample]

use cpal::traits::DeviceTrait;
use msk2k_audio::{
    needs_resampling, AudioConfig, AudioInputBuilder, AudioOutputBuilder, AudioResampler,
    DeviceManager, MSK2K_SAMPLE_RATE,
};
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let use_resampling = std::env::args().any(|arg| arg == "--resample");

    let manager = DeviceManager::new()?;

    let input_device = manager.default_input_device()?;
    let output_device = manager.default_output_device()?;

    println!("Input device: {}", input_device.name()?);
    println!("Output device: {}", output_device.name()?);

    // Input at 48kHz (common sound card rate)
    let input_rate = if use_resampling {
        48000
    } else {
        MSK2K_SAMPLE_RATE
    };
    let output_rate = MSK2K_SAMPLE_RATE;

    let input_config = AudioConfig::new(input_rate, 1, 1024);
    let output_config = AudioConfig::new(output_rate, 1, 1024);

    println!("\nConfiguration:");
    println!("  Input:  {}", input_config);
    println!("  Output: {}", output_config);

    // Create channels
    let (input_tx, mut input_rx) = mpsc::unbounded_channel();
    let (output_tx, output_rx) = mpsc::unbounded_channel();

    // Start input stream
    let mut audio_input = AudioInputBuilder::new()
        .device(input_device)
        .config(input_config)
        .build()?;

    audio_input.start(input_tx)?;
    println!("\nAudio input started");

    // Start output stream
    let mut audio_output = AudioOutputBuilder::new()
        .device(output_device)
        .config(output_config)
        .build()?;

    audio_output.start(output_rx)?;
    println!("Audio output started");

    // Create resampler if needed
    let mut resampler = if needs_resampling(input_rate, output_rate) {
        println!("Resampling enabled: {}Hz → {}Hz", input_rate, output_rate);
        Some(AudioResampler::new_fixed_in(input_rate, output_rate, 1024)?)
    } else {
        println!("No resampling needed");
        None
    };

    println!("\nLoopback active - press Ctrl+C to stop");
    println!("Speak into your microphone and you should hear yourself!");

    // Process loop
    let mut sample_count = 0u64;
    while let Some(samples) = input_rx.recv().await {
        sample_count += samples.len() as u64;

        let output_samples = if let Some(ref mut r) = resampler {
            r.process(&samples)?
        } else {
            samples
        };

        // Send to output (ignore if channel is full/closed)
        let _ = output_tx.send(output_samples);

        // Print stats every second
        if sample_count % (input_rate as u64) < 1024 {
            println!(
                "Processed {:.1}s of audio",
                sample_count as f64 / input_rate as f64
            );
        }
    }

    Ok(())
}
