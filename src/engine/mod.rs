// src/engine/mod.rs
pub mod bus;
pub mod runtime;
pub mod report_calc;
pub mod accumulator;

// Re-export the “public API” that GUI code expects.
pub use bus::{EngineHandle, SlotParity, SlotPeriod, UiCmd, UiEvent};

// Convenience start function (GUI can call crate::engine::start_engine()).
// Your existing code calls runtime::start() too, so both exist.
pub fn start_engine() -> EngineHandle {
    runtime::start()
}

// Back-compat: your current file calls runtime::start() already.
pub fn start() -> EngineHandle {
    runtime::start()
}