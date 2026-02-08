// src/main.rs

mod modem;   
mod engine;
mod gui;
mod settings;
mod proto;
mod qso;

fn main() {
    env_logger::init();

    if let Err(e) = gui::app::run_gui() {
        eprintln!("GUI exited with error: {e:?}");
    }
}