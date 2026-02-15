// src/proto.rs
//
// Protocol types for QSO state machine.
// These are the "logical" message types used by the QSO engine,
// separate from the DSP encoding details.

use msk2k_dsp::message::Message;
#[allow(dead_code)]
/// Message format (maps to PSK2K Format-1 or Format-2)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Fmt1,  // Full messages: CQ, calls with reports
    Fmt2,  // Short messages: R-reports, RR, 73
}

/// 🟢 NEW: Container for the output of render_payload.
/// This allows us to send either a standard string or raw pre-packed bits.
#[derive(Debug, Clone)]
pub enum Rendered {
    Text(String),
    Bits(Vec<i32>),
}

/// Logical payload types for QSO protocol.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum Payload {
    /// CQ call (Format-1, general address)
    Cq {
        from: String,
        qtf_deg: Option<i16>,
        period: Option<u32>,
    },

    /// 🟢 Grid CQ: Holds the raw 56-bit vector generated in runtime.rs
    CqWithGrid {
        from: String,
        grid_bits: Vec<i32>,
    },
    
    /// Cold call - calling a station without report (Format-1, private)
    Call {
        from: String,
        to: String,
    },
    
    /// Call with report (Format-1, private)
    CallWithReport {
        from: String,
        to: String,
        rpt: i16,
    },
    
    /// Report only (Format-2)
    Report {
        from: String,
        to: String,
        rpt: i16,
    },
    
    /// Roger + Report (Format-2)
    RReport {
        from: String,
        to: String,
        rpt: i16,
    },
    
    /// Roger Roger (Format-2)
    Rr {
        from: String,
        to: String,
    },
    
    /// 73 - end of QSO (Format-2)
    SeventyThree {
        from: String,
        to: String,
    },
    
    /// Free text (Format-1)
    Text {
        from: String,
        to: Option<String>,
        text: String,
    },
}
#[allow(dead_code)]
impl Payload {
    pub fn from_call(&self) -> &str {
        match self {
            Payload::Cq { from, .. } => from,
            Payload::CqWithGrid { from, .. } => from,
            Payload::Call { from, .. } => from,
            Payload::CallWithReport { from, .. } => from,
            Payload::Report { from, .. } => from,
            Payload::RReport { from, .. } => from,
            Payload::Rr { from, .. } => from,
            Payload::SeventyThree { from, .. } => from,
            Payload::Text { from, .. } => from,
        }
    }
    
    pub fn to_call(&self) -> Option<&str> {
        match self {
            Payload::Cq { .. } => None,
            Payload::Call { to, .. } => Some(to),
            Payload::CallWithReport { to, .. } => Some(to),
            Payload::Report { to, .. } => Some(to),
            Payload::RReport { to, .. } => Some(to),
            Payload::Rr { to, .. } => Some(to),
            Payload::SeventyThree { to, .. } => Some(to),
            Payload::Text { to, .. } => to.as_deref(),
            Payload::CqWithGrid { .. } => None,
        }
    }
    
    pub fn format(&self) -> Format {
        match self {
            Payload::Cq { .. } => Format::Fmt1,
            Payload::CqWithGrid { .. } => Format::Fmt1,
            Payload::Call { .. } => Format::Fmt1,
            Payload::CallWithReport { .. } => Format::Fmt1,
            Payload::Report { .. } => Format::Fmt2,
            Payload::RReport { .. } => Format::Fmt2,
            Payload::Rr { .. } => Format::Fmt2,
            Payload::SeventyThree { .. } => Format::Fmt2,
            Payload::Text { .. } => Format::Fmt1,
        }
    }
}
#[allow(dead_code)]
/// Envelope for received messages
#[derive(Debug, Clone)]
pub struct RxEnvelope {
    pub payload: Payload,
    pub format: Format,
    pub snr: Option<f32>,
    pub utc_ms: i64,
    pub rx_slot: u8,
}

/// Envelope for messages to transmit
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TxEnvelope {
    pub payload: Payload,
    pub format: Format,
    pub raw: String,
}

/// 🟢 UPDATED: Render a payload to a Rendered enum.
/// This allows the modem to accept raw bits for Grid mode, bypassing the standard string encoder.
// src/proto.rs

pub fn render_payload(payload: &Payload) -> Rendered {
    match payload {
        Payload::CqWithGrid { from: _, grid_bits } => {
            // 🟢 Send the 56-bit Source Encoded block.
            // This contains the Call, Grid, and Signature.
            // It is NOT yet FEC encoded or Interleaved.
            Rendered::Bits(grid_bits.clone()) 
        }

        Payload::Cq { from, .. } => {
            // Standard text. The modem will perform Source Encoding (Base-37) on this.
            Rendered::Text(format!("CQ de {}", from))
        }
        
        // ... rest of match arms (Text variants)
        Payload::Call { from, to } => Rendered::Text(format!("{} de {}", to, from)),
        Payload::CallWithReport { from, to, rpt } => Rendered::Text(format!("{} de {} {}", to, from, rpt)),
        Payload::Report { rpt, .. } => Rendered::Text(format!("R{}", rpt)),
        Payload::RReport { rpt, .. } => Rendered::Text(format!("R{}", rpt)),
        Payload::Rr { .. } => Rendered::Text("RR".to_string()),
        Payload::SeventyThree { .. } => Rendered::Text("73".to_string()),
        Payload::Text { from, to, text } => {
             if let Some(to) = to { Rendered::Text(format!("{} de {} {}", to, from, text)) }
             else { Rendered::Text(format!("CQ de {} {}", from, text)) }
        }
    }
}

pub fn message_to_payload(msg: &Message) -> Option<Payload> {
    let from = msg.from_call.clone();
    let to = msg.to_call.clone();
    
    match msg.format {
        1 => {
            let text = &msg.text;
            if text.to_uppercase().starts_with("CQ") {
                return Some(Payload::Cq {
                    from,
                    qtf_deg: None,
                    period: None,
                });
            }
            if let Some(to) = to {
                let rpt = extract_report_from_text(text);
                if let Some(rpt) = rpt {
                    return Some(Payload::CallWithReport { from, to, rpt });
                } else {
                    return Some(Payload::Call { from, to });
                }
            }
            Some(Payload::Text { from, to, text: text.clone() })
        }
        2 => {
            let to = to?;
            match msg.text.as_str() {
                "R26" => Some(Payload::RReport { from, to, rpt: 26 }),
                "R27" => Some(Payload::RReport { from, to, rpt: 27 }),
                "R28" => Some(Payload::RReport { from, to, rpt: 28 }),
                "R29" => Some(Payload::RReport { from, to, rpt: 29 }),
                "R36" => Some(Payload::RReport { from, to, rpt: 36 }),
                "R37" => Some(Payload::RReport { from, to, rpt: 37 }),
                "RR" => Some(Payload::Rr { from, to }),
                "73" => Some(Payload::SeventyThree { from, to }),
                _ => None
            }
        }
        _ => None
    }
}

fn extract_report_from_text(text: &str) -> Option<i16> {
    let valid_reports = [26i16, 27, 28, 29, 36, 37];
    for word in text.split_whitespace().rev() {
        if word.chars().any(|c| c.is_alphabetic()) && word.chars().any(|c| c.is_numeric()) {
            continue;
        }
        if let Ok(num) = word.parse::<i16>() {
            if valid_reports.contains(&num) { return Some(num); }
        }
        if word.starts_with('R') && word.len() == 3 {
            if let Ok(num) = word[1..].parse::<i16>() {
                if valid_reports.contains(&num) { return Some(num); }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_render_cq() {
        let p = Payload::Cq { 
            from: "GW4WND".into(), 
            qtf_deg: None, 
            period: None 
        };
        assert_eq!(render_payload(&p), "CQ de GW4WND");
    }
    
    #[test]
    fn test_render_call_with_report() {
        let p = Payload::CallWithReport {
            from: "GW4WND".into(),
            to: "SM2CEW".into(),
            rpt: 27,
        };
        assert_eq!(render_payload(&p), "SM2CEW de GW4WND 27");
    }
    
    #[test]
    fn test_render_rreport() {
        let p = Payload::RReport {
            from: "GW4WND".into(),
            to: "SM2CEW".into(),
            rpt: 27,
        };
        assert_eq!(render_payload(&p), "R27");
    }
    
    #[test]
    fn test_render_rr() {
        let p = Payload::Rr {
            from: "GW4WND".into(),
            to: "SM2CEW".into(),
        };
        assert_eq!(render_payload(&p), "RR");
    }
    
    #[test]
    fn test_render_73() {
        let p = Payload::SeventyThree {
            from: "GW4WND".into(),
            to: "SM2CEW".into(),
        };
        assert_eq!(render_payload(&p), "73");
    }
    
    #[test]
    fn test_extract_report() {
        assert_eq!(extract_report_from_text("SM2CEW de DJ5HG 27"), Some(27));
        assert_eq!(extract_report_from_text("SM2CEW de DJ5HG"), None);
        assert_eq!(extract_report_from_text("CQ de DJ5HG"), None);
    }
}
