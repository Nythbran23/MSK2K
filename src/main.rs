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
    env_logger::init();

    // We call your existing run_gui() function because Msk2kApp is private
    if let Err(e) = gui::app::run_gui() {
        eprintln!("GUI exited with error: {e:?}");
    }
}