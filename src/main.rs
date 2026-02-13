// src/main.rs

// 1. HIDE CONSOLE WINDOW (Windows Only)
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod modem;
mod engine;
mod gui;
mod settings;
mod proto;
mod qso;

fn main() {
    // Set up logging: file + stderr
    // Always log to msk2k.log next to the executable (critical for Windows GUI builds)
    let log_path = std::env::current_exe()
        .ok()
        .and_then(|mut p| { p.pop(); p.push("msk2k.log"); Some(p) })
        .unwrap_or_else(|| std::path::PathBuf::from("msk2k.log"));

    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path);

    // Use RUST_LOG if set, otherwise default to info
    let env = env_logger::Env::default().default_filter_or("info");

    match file {
        Ok(f) => {
            env_logger::Builder::from_env(env)
                .target(env_logger::Target::Pipe(Box::new(f)))
                .init();
            log::info!("Logging to: {}", log_path.display());
        }
        Err(_) => {
            // Fall back to stderr if file can't be opened
            env_logger::Builder::from_env(env).init();
        }
    }

    if let Err(e) = gui::app::run_gui() {
        log::error!("GUI exited with error: {e:?}");
        eprintln!("GUI exited with error: {e:?}");
    }
}
