// src/main.rs

// 1. HIDE CONSOLE WINDOW (Windows Only)
// This attribute tells the compiler: "When compiling for Windows in Release mode, 
// use the GUI subsystem, not the Console subsystem."
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod modem;
mod engine;
mod gui;
mod settings;
mod proto;
mod qso;

use eframe::egui;

fn main() -> Result<(), eframe::Error> {
    env_logger::init();

    // 2. CONFIGURE WINDOW OPTIONS
    let options = eframe::NativeOptions {
        // Force the app to always use Dark Mode (fixes the white/black conflict)
        default_theme: eframe::Theme::Dark,
        
        // Set a nice default size
        initial_window_size: Some(egui::vec2(1000.0, 600.0)),
        
        // Keep other defaults
        ..Default::default()
    };

    // 3. LAUNCH THE APP
    eframe::run_native(
        "MSK2K",
        options,
        Box::new(|cc| Box::new(gui::app::Msk2kApp::new(cc))),
    )
}