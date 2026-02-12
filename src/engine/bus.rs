// src/engine/bus.rs

// 🟢 FIX: Import QsoRecord from the correct location
use crate::qso::adif::QsoRecord;
// RxEnvelope might still be needed depending on other files, keeping it safe
use crate::proto::RxEnvelope; 
use tokio::runtime::Runtime;
use tokio::sync::mpsc;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SlotPeriod {
    S15,
    S30,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SlotParity {
    Odd,
    Even,
}

#[derive(Debug, Clone)]
pub enum UiCmd {
    SetInputDevice(Option<String>),
    SetOutputDevice(Option<String>),
    SetSlotPeriod(SlotPeriod),
    SetSlotParity(SlotParity),
    SetAutoQso(bool),
    ApplyAudio,
    ConfigureHamlib { enabled: bool, address: String },
    
    ConfigureLauncher { 
        enable_launcher: bool, 
        rig_model: String, 
        serial_port: String, 
        baud_rate: u32 
    },
    Listen {
        my_call: String,
        their_call: String,
        auto_slots: bool,
    },
    StartCq {
        my_call: String,
        auto_slots: bool,
    },

    StartCqWithGrid { 
        my_call: String, 
        grid_indices: [usize; 4], 
        auto_slots: bool 
    },
    
    CallStation {
        my_call: String,
        their_call: String,
    },
    AnswerCq {
        my_call: String,
        their_call: String,
        rpt: i16,
        rx_slot: Option<u8>,
        grid: Option<String>, // 🟢 NEW: Grid from CQ
    },
    ColdCall {
        my_call: String,
        their_call: String,
    },
    SendReport {
        my_call: String,
        their_call: String,
        rpt: String,
    },
    SendRReport {
        my_call: String,
        their_call: String,
        rrpt: String,
    },
    SendRr {
        my_call: String,
        their_call: String,
    },
    Send73 {
        my_call: String,
        their_call: String,
    },
    Stop,
    SetBand(String),
    PublicTx {
        my_call: String,
        text: String,
    },
}

#[derive(Debug, Clone)]
pub enum UiEvent {
    RxText {
        text: String,
        snr: Option<f32>,
        utc_ms: i64,
        rx_slot: u8,
    },
    TxText {
        text: String,
    },
    State(String),
    Info(String),
    QsoLogged {
        record: QsoRecord,
    },
    TxSlotChanged {
        slot: SlotParity,
    },
    TheirCallChanged {
        callsign: String,
        grid: Option<String>, // 🟢 NEW: Grid if available
    },
    ConfigLoaded {
        my_call: String,
        input_device: Option<String>,
        output_device: Option<String>,
    },
}

pub struct EngineHandle {
    pub cmds: mpsc::UnboundedSender<UiCmd>,
    pub events: mpsc::UnboundedReceiver<UiEvent>,
    pub _rt: Runtime,
}