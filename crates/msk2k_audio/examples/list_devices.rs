//! List available audio devices
//!
//! Usage: cargo run --example list_devices

use msk2k_audio::DeviceManager;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let manager = DeviceManager::new()?;

    println!("=== Audio Input Devices ===");
    match manager.list_input_devices() {
        Ok(devices) => {
            if devices.is_empty() {
                println!("No input devices found");
            } else {
                for (i, device) in devices.iter().enumerate() {
                    println!("{}. {}", i + 1, device);
                }
            }
        }
        Err(e) => println!("Error listing input devices: {}", e),
    }

    println!("\n=== Audio Output Devices ===");
    match manager.list_output_devices() {
        Ok(devices) => {
            if devices.is_empty() {
                println!("No output devices found");
            } else {
                for (i, device) in devices.iter().enumerate() {
                    println!("{}. {}", i + 1, device);
                }
            }
        }
        Err(e) => println!("Error listing output devices: {}", e),
    }

    Ok(())
}
