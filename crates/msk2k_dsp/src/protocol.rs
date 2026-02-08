// crates/msk2k_app/src/proto.rs
//
// Protocol-level envelopes used between UI <-> QSO engine <-> modem runtime.
// Future-proof: supports QTF/period + RX metrics (snr, freq offset, etc).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Fmt1,
    Fmt2,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Period {
    Ms2500,
    Ms15000,
    Ms30000,
    Ms60000,
}

#[derive(Debug, Clone)]
pub struct RxMetrics {
    /// SNR estimate in dB if available (receiver-measured, not in-band).
    pub snr_db: Option<f32>,
    /// Estimated frequency offset in Hz if available (receiver-measured).
    pub freq_offset_hz: Option<f32>,
    /// Any sync/correlation metric (receiver internal).
    pub sync_quality: Option<f32>,
    /// Which decode method/path produced this (e.g. "hybrid", "fft", etc.).
    pub method: Option<String>,
    /// Monotonic-ish timestamp for log ordering (ms since start or epoch; your choice).
    pub ts_ms: u64,
    /// Optional "ping" duration (ms) or accumulation window length.
    pub ping_ms: Option<u32>,
}

impl Default for RxMetrics {
    fn default() -> Self {
        Self {
            snr_db: None,
            freq_offset_hz: None,
            sync_quality: None,
            method: None,
            ts_ms: 0,
            ping_ms: None,
        }
    }
}

/// Rich payload meaning (UI/engine operates on this, not raw strings).
/// Keep this enum stable; you can add variants over time without rewriting the app.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Payload {
    // --- Core QSO primitives (minimum viable workflow) ---
    Cq {
        from: String,
        // Future: direction / QTF, period encoding, etc.
        qtf_deg: Option<u16>,
        period: Option<Period>,
    },
    Call {
        from: String,
        to: String,
    },
    Report {
        from: String,
        to: String,
        rpt: i16, // typically two-digit (e.g., 26)
        // Future: add extra report fields if you want
    },
    RReport {
        from: String,
        to: String,
        rpt: i16,
    },
    Rr {
        from: String,
        to: String,
    },
    SeventyThree {
        from: String,
        to: String,
    },

    // --- Future capability (present in design, not necessarily implemented yet) ---
    FreeText {
        from: String,
        to: String,
        text: String, // could be limited to 10 chars later
    },
    Contest {
        from: String,
        to: String,
        roger: bool,
        rpt: Option<i16>,
        qtf_deg: Option<u16>,
        period: Option<Period>,
        locator: Option<String>,
        pwr_w: Option<u16>,
        ant_gain_db: Option<u8>,
        serial: Option<u16>,
    },
}

/// What the RX pipeline delivers to the engine/UI.
/// `raw` is the human-readable string you actually decoded (useful for logs).
#[derive(Debug, Clone)]
pub struct RxEnvelope {
    pub payload: Payload,
    pub format: Format,
    pub raw: String,
    pub metrics: RxMetrics,
}

/// What the engine requests the modem to transmit.
#[derive(Debug, Clone)]
pub struct TxEnvelope {
    pub payload: Payload,
    pub format: Format,
    pub raw: String, // what will be sent (for logs and debugging)
}

/// Helpers to build canonical on-air strings.
/// Keep these here so UI/engine/modem agree on exact formatting.
pub fn render_payload(payload: &Payload) -> String {
    match payload {
        Payload::Cq { from, qtf_deg, period } => {
            // Keep CQ minimal for now; we retain qtf/period in the payload for future use.
            // You can later append them according to your MSK2K scheme.
            let mut s = format!("CQ {}", from);
            if let Some(qtf) = qtf_deg {
                s.push_str(&format!(" QTF{}", qtf));
            }
            if let Some(p) = period {
                let ms = match p {
                    Period::Ms2500 => 2500,
                    Period::Ms15000 => 15000,
                    Period::Ms30000 => 30000,
                    Period::Ms60000 => 60000,
                };
                s.push_str(&format!(" P{}", ms));
            }
            s
        }
        Payload::Call { from, to } => format!("{} de {}", to, from),
        Payload::Report { from, to, rpt } => format!("{} de {} {}", to, from, rpt),
        Payload::RReport { from, to, rpt } => format!("{} de {} R{}", to, from, rpt),
        Payload::Rr { from, to } => format!("{} de {} RR", to, from),
        Payload::SeventyThree { from, to } => format!("{} de {} 73", to, from),
        Payload::FreeText { from, to, text } => format!("{} de {} {}", to, from, text),
        Payload::Contest {
            from,
            to,
            roger,
            rpt,
            qtf_deg,
            period,
            locator,
            pwr_w,
            ant_gain_db,
            serial,
        } => {
            // Placeholder canonicalization. You can adjust once your MSK2K contest format is finalized.
            let mut parts = vec![format!("{} de {}", to, from)];
            if *roger {
                parts.push("R".to_string());
            }
            if let Some(r) = rpt {
                parts.push(format!("{}", r));
            }
            if let Some(qtf) = qtf_deg {
                parts.push(format!("QTF{}", qtf));
            }
            if let Some(p) = period {
                let ms = match p {
                    Period::Ms2500 => 2500,
                    Period::Ms15000 => 15000,
                    Period::Ms30000 => 30000,
                    Period::Ms60000 => 60000,
                };
                parts.push(format!("P{}", ms));
            }
            if let Some(loc) = locator {
                parts.push(format!("LOC{}", loc));
            }
            if let Some(pwr) = pwr_w {
                parts.push(format!("{}W", pwr));
            }
            if let Some(g) = ant_gain_db {
                parts.push(format!("G{}dB", g));
            }
            if let Some(sn) = serial {
                parts.push(format!("#{}", sn));
            }
            parts.join(" ")
        }
    }
}