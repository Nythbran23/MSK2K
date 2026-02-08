// src/proto.rs
//
// Protocol types for QSO state machine.
// These are the "logical" message types used by the QSO engine,
// separate from the DSP encoding details.

use msk2k_dsp::message::Message;

/// Message format (maps to PSK2K Format-1 or Format-2)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Fmt1,  // Full messages: CQ, calls with reports
    Fmt2,  // Short messages: R-reports, RR, 73
}

/// Logical payload types for QSO protocol.
/// 
/// Region-1 ladder:
/// 1. CQ de A                    (Fmt1, general)
/// 2. A de B 27                  (Fmt1, private - call with report)
/// 3. B de A 27                  (Fmt1, private - report back)
/// 4. R27                        (Fmt2 - roger + report)
/// 5. RR                         (Fmt2 - roger roger)
/// 6. 73                         (Fmt2 - end)
/// 7. 73                         (Fmt2 - end)
#[derive(Debug, Clone, PartialEq)]
pub enum Payload {
    /// CQ call (Format-1, general address)
    Cq {
        from: String,
        qtf_deg: Option<i16>,
        period: Option<u32>,
    },
    
    /// Cold call - calling a station without report (Format-1, private)
    Call {
        from: String,
        to: String,
    },
    
    /// Call with report (Format-1, private)
    /// Used when answering CQ or sending initial report
    CallWithReport {
        from: String,
        to: String,
        rpt: i16,
    },
    
    /// Report only (Format-2)
    /// This is the "R27" style message
    Report {
        from: String,
        to: String,
        rpt: i16,
    },
    
    /// Roger + Report (Format-2)
    /// Acknowledges receipt and sends report back
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

impl Payload {
    /// Get the "from" callsign
    pub fn from_call(&self) -> &str {
        match self {
            Payload::Cq { from, .. } => from,
            Payload::Call { from, .. } => from,
            Payload::CallWithReport { from, .. } => from,
            Payload::Report { from, .. } => from,
            Payload::RReport { from, .. } => from,
            Payload::Rr { from, .. } => from,
            Payload::SeventyThree { from, .. } => from,
            Payload::Text { from, .. } => from,
        }
    }
    
    /// Get the "to" callsign if present
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
        }
    }
    
    /// Determine the format for this payload
    pub fn format(&self) -> Format {
        match self {
            Payload::Cq { .. } => Format::Fmt1,
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
#[derive(Debug, Clone)]
pub struct TxEnvelope {
    pub payload: Payload,
    pub format: Format,
    /// Pre-rendered text for TX (what actually gets sent to modem)
    pub raw: String,
}

/// Render a payload to the text string expected by the TX modem.
/// 
/// The modem tx.rs expects specific formats:
/// - CQ: "CQ de CALL"
/// - Call: "TO de FROM"
/// - Call+Report: "TO de FROM 27"
/// - Format-2 short: "R27", "RR", "73" (modem adds callsigns)
pub fn render_payload(payload: &Payload) -> String {
    match payload {
        Payload::Cq { from, .. } => {
            format!("CQ de {}", from)
        }
        
        Payload::Call { from, to } => {
            format!("{} de {}", to, from)
        }
        
        Payload::CallWithReport { from, to, rpt } => {
            format!("{} de {} {}", to, from, rpt)
        }
        
        // Format-2 messages: just the short code
        // The TX modem will add callsigns from context
        Payload::Report { rpt, .. } => {
            format!("R{}", rpt)
        }
        
        Payload::RReport { rpt, .. } => {
            format!("R{}", rpt)
        }
        
        Payload::Rr { .. } => {
            "RR".to_string()
        }
        
        Payload::SeventyThree { .. } => {
            "73".to_string()
        }
        
        Payload::Text { from, to, text } => {
            if let Some(to) = to {
                format!("{} de {} {}", to, from, text)
            } else {
                format!("CQ de {} {}", from, text)
            }
        }
    }
}

/// Convert a DSP Message to a protocol Payload.
/// 
/// This bridges the gap between what the decoder produces and what
/// the QSO state machine expects.
pub fn message_to_payload(msg: &Message) -> Option<Payload> {
    let from = msg.from_call.clone();
    let to = msg.to_call.clone();
    
    match msg.format {
        1 => {
            // Format-1: CQ or Call (with optional report)
            let text = &msg.text;
            
            // Check for CQ
            if text.to_uppercase().starts_with("CQ") {
                return Some(Payload::Cq {
                    from,
                    qtf_deg: None,
                    period: None,
                });
            }
            
            // Check for call with report
            // Format: "TO de FROM RPT" where RPT is like "27"
            if let Some(to) = to {
                // Try to extract report from text
                let rpt = extract_report_from_text(text);
                
                if let Some(rpt) = rpt {
                    return Some(Payload::CallWithReport { from, to, rpt });
                } else {
                    return Some(Payload::Call { from, to });
                }
            }
            
            // Fallback to text
            Some(Payload::Text { from, to, text: text.clone() })
        }
        
        2 => {
            // Format-2: Short messages
            let to = to?; // Format-2 always has a destination
            
            match msg.text.as_str() {
                "R26" => Some(Payload::RReport { from, to, rpt: 26 }),
                "R27" => Some(Payload::RReport { from, to, rpt: 27 }),
                "R28" => Some(Payload::RReport { from, to, rpt: 28 }),
                "R29" => Some(Payload::RReport { from, to, rpt: 29 }),
                "R36" => Some(Payload::RReport { from, to, rpt: 36 }),
                "R37" => Some(Payload::RReport { from, to, rpt: 37 }),
                "RR" => Some(Payload::Rr { from, to }),
                "73" => Some(Payload::SeventyThree { from, to }),
                _ => {
                    log::warn!("Unknown Format-2 message type: {}", msg.text);
                    None
                }
            }
        }
        
        _ => {
            log::warn!("Unknown message format: {}", msg.format);
            None
        }
    }
}

/// Extract numeric report from text like "SM2CEW de DJ5HG 27"
fn extract_report_from_text(text: &str) -> Option<i16> {
    // Valid reports: 26, 27, 28, 29, 36, 37
    let valid_reports = [26i16, 27, 28, 29, 36, 37];
    
    for word in text.split_whitespace().rev() {
        // Skip if it looks like a callsign (contains letters and numbers)
        if word.chars().any(|c| c.is_alphabetic()) && word.chars().any(|c| c.is_numeric()) {
            continue;
        }
        
        // Try to parse as number
        if let Ok(num) = word.parse::<i16>() {
            if valid_reports.contains(&num) {
                return Some(num);
            }
        }
        
        // Check for R-prefix style (shouldn't be in Format-1 but handle it)
        if word.starts_with('R') && word.len() == 3 {
            if let Ok(num) = word[1..].parse::<i16>() {
                if valid_reports.contains(&num) {
                    return Some(num);
                }
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
