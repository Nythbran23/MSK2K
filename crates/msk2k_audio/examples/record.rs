//! Record audio to a WAV file
//!
//! Usage: cargo run --example record [duration_seconds] [output.wav]

use cpal::traits::DeviceTrait;
use hound::{WavSpec, WavWriter};
use msk2k_audio::{AudioConfig, AudioInputBuilder, DeviceManager, MSK2K_SAMPLE_RATE};
use std::path::PathBuf;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // Parse arguments
    let args: Vec<String> = std::env::args().collect();
    let duration_secs = args
        .get(1)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(5.0);
    let output_path = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("recording.wav"));

    println!("Recording {} seconds to {:?}", duration_secs, output_path);

    let manager = DeviceManager::new()?;
    let device = manager.default_input_device()?;

    println!("Using input device: {}", device.name()?);

    // Configure for MSK2K sample rate
    let config = AudioConfig::new(MSK2K_SAMPLE_RATE, 1, 1024);
    println!("Configuration: {}", config);

    // Create WAV writer
    let spec = WavSpec {
        channels: 1,
        sample_rate: MSK2K_SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = WavWriter::create(&output_path, spec)?;

    // Create audio input
    let (tx, mut rx) = mpsc::unbounded_channel();

    let mut audio_input = AudioInputBuilder::new()
        .device(device)
        .config(config)
        .build()?;

    audio_input.start(tx)?;
    println!("\nRecording started...");

    // Calculate total samples to record
    let total_samples = (duration_secs * MSK2K_SAMPLE_RATE as f32) as usize;
    let mut recorded_samples = 0;

    // Record loop
    while let Some(samples) = rx.recv().await {
        // Write samples to WAV file
        for &sample in &samples {
            writer.write_sample(sample)?;
            recorded_samples += 1;

            if recorded_samples >= total_samples {
                break;
            }
        }

        // Print progress
        let progress = (recorded_samples as f32 / total_samples as f32) * 100.0;
        print!("\rProgress: {:.1}%", progress);

        if recorded_samples >= total_samples {
            break;
        }
    }

    println!("\n\nRecording complete!");
    println!("Finalizing WAV file...");

    writer.finalize()?;

    println!("Saved to: {:?}", output_path);
    println!("Total samples: {}", recorded_samples);
    println!(
        "Duration: {:.2}s",
        recorded_samples as f32 / MSK2K_SAMPLE_RATE as f32
    );

    Ok(())
}
