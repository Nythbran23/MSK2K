// src/modem/mod.rs
#[allow(dead_code)]
pub mod audio_ring;
pub mod rx;
pub mod sync_tracker;  // kept for reference but no longer used in the RX path
pub mod tx;
pub use rx::{run_receiver, RxAudioCfg, RxConfigUpdate, RxDecoded};
pub use tx::run_transmitter_task;

/// Runtime → TX worker requests.
/// Keep this minimal and stable; UI/engine can evolve without changing DSP.
///
/// IMPORTANT:
/// TX never blocks RX processing.
/// RX capture may pause, but decode continues until buffer drains.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum TxRequest {
    ApplyAudio {
        output_device: Option<String>,
        output_level: f32, // Ignored by new TX logic, kept for compat
        sample_rate: u32,
        buffer_size: usize,

        // Needed for Format-2 address packing
        my_call: String,
        their_call: String,
    },

    /// Transmit rendered message.
    /// Callsigns are always included so Format-2 builder has context.
    Text {
        rendered: String,
        slot_len_ms: u32,
        my_call: String,
        their_call: String,
    },

    /// NEW GRID PATH: Modem will use these bits directly as source for FEC 
    RawBits {
        bits: Vec<i32>,
        slot_len_ms: u32,
        my_call: String,
        their_call: String,
    },

    /// Stop TX immediately.
    /// RX decode MUST continue draining its buffer.
    Stop,
}