// crates/msk2k_dsp/src/core/mod.rs
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
pub mod rx; // only if this is DSP-level (no audio/tokio). If it uses audio/tokio, remove it.