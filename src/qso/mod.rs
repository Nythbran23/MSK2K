// src/qso/mod.rs
//
// Pure, deterministic QSO/session state machine.
// No egui, no audio, no tokio. Easy to unit test.
//
// Region-1 Meteor Scatter Ladder (3 scenarios):
// ==============================================
//
// SCENARIO ONE - A calls CQ:
// A: CQ                              ->
//                                    <-  B: A de B 27
// A: R27                             ->
//                                    <-  B: RR
// A: 73 (x5)                         ->
//                                    <-  B: 73 (x5)
//
// SCENARIO TWO - A receives cold call (no report):
//                                    <-  B: A de B (no report)
// A: B de A 27                       ->
//                                    <-  B: R27
// A: RR                              ->
//                                    <-  B: 73 (x5)
// A: 73 (x5)                         ->
//
// SCENARIO THREE - A calls specific station:
// A: B de A (no report)              ->
//                                    <-  B: A de B 27
// A: R27                             ->
//                                    <-  B: RR
// A: 73 (x5)                         ->
//                                    <-  B: 73 (x5)

pub mod adif;

#[allow(unused_imports)]
use crate::proto::{render_payload, Format, Payload, RxEnvelope, TxEnvelope, Rendered};
use adif::QsoRecord;
use std::time::{SystemTime, UNIX_EPOCH};

/// Get current UTC time in milliseconds
fn utc_ms_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// QSO protocol states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QsoState {
    Idle,
    Listening,
    CallingCq,
    CallingStn,
    SendingReport,
    SendingRReport,
    SendingRr,
    Sending73,
    Done,
}

impl std::fmt::Display for QsoState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QsoState::Idle => write!(f, "IDLE"),
            QsoState::Listening => write!(f, "LISTEN"),
            QsoState::CallingCq => write!(f, "CALLING_CQ"),
            QsoState::CallingStn => write!(f, "CALLING_STN"),
            QsoState::SendingReport => write!(f, "SEND_RPT"),
            QsoState::SendingRReport => write!(f, "SEND_RRPT"),
            QsoState::SendingRr => write!(f, "SEND_RR"),
            QsoState::Sending73 => write!(f, "SEND_73"),
            QsoState::Done => write!(f, "DONE"),
        }
    }
}

/// User/UI intents that drive state changes
#[derive(Debug, Clone)]
pub enum Intent {
    Listen,
    Cq,
    Call { their: String },
    AnswerCq { their: String, rpt: i16, grid: Option<String> }, // 🟢 NEW: Grid from CQ
    Abort,
}

/// Events emitted by the engine
#[derive(Debug, Clone)]
pub enum EngineEvent {
    StateChanged(QsoState),
    Rx(RxEnvelope),
    Tx(TxEnvelope),
    Info(String),
    QsoComplete { 
        their: String,
        record: Option<QsoRecord>,
    },
    TheirCallChanged { callsign: String, grid: Option<String> }, // 🟢 NEW: Grid
}

/// Action to take after processing
#[derive(Debug, Clone)]
pub enum Action {
    None,
    Transmit(TxEnvelope),
}

/// The QSO state machine engine
pub struct QsoEngine {
    pub my_call: String,
    pub their_call: Option<String>,
    pub their_grid: Option<String>, // 🟢 NEW: Store their grid from CQ
    pub state: QsoState,
    pub my_report: i16,
    pub their_report: Option<i16>,
    pub last_rx: Option<RxEnvelope>,
    pub last_tx: Option<TxEnvelope>,
    pub tx_repeat_count: u8,
    pub max_repeats: u8,
    pub qso_start_utc_ms: Option<i64>,
    pub band: String,
}

impl QsoEngine {
    pub fn new(my_call: String) -> Self {
        Self {
            my_call,
            their_call: None,
            their_grid: None, // 🟢 NEW: No grid initially
            state: QsoState::Idle,
            my_report: 27,
            their_report: None,
            last_rx: None,
            last_tx: None,
            tx_repeat_count: 0,
            max_repeats: 5,
            qso_start_utc_ms: None,
            band: "2M".to_string(),
        }
    }

    pub fn set_my_call(&mut self, my: String) {
        self.my_call = my;
    }

    pub fn set_their_call(&mut self, their: Option<String>) {
        self.their_call = their;
    }
    
    pub fn set_my_report(&mut self, rpt: i16) {
        self.my_report = rpt;
    }
    
    pub fn set_band(&mut self, band: String) {
        self.band = band;
    }
    
    fn mark_qso_start(&mut self) {
        if self.qso_start_utc_ms.is_none() {
            self.qso_start_utc_ms = Some(utc_ms_now());
            log::info!("📝 QSO started at {}", self.qso_start_utc_ms.unwrap());
        }
    }
    
    pub fn make_qso_record(&self) -> Option<QsoRecord> {
        let start = self.qso_start_utc_ms?;
        let their = self.their_call.as_ref()?;
        let end = utc_ms_now();
        
        Some(QsoRecord::new(
            their.clone(),
            self.my_call.clone(),
            start,
            end,
            self.band.clone(),
            None,
            self.my_report,
            self.their_report,
            self.their_grid.clone(), // 🟢 NEW: Include grid in record
        ))
    }

    pub fn on_intent(&mut self, intent: Intent) -> (Action, Vec<EngineEvent>) {
        let mut ev = vec![];
        let mut action = Action::None;

        match intent {
            Intent::Listen => {
                self.reset_qso();
                self.transition(QsoState::Listening, &mut ev);
            }
            Intent::Abort => {
                self.reset_qso();
                self.transition(QsoState::Idle, &mut ev);
            }
            Intent::Cq => {
                self.reset_qso();
                let payload = Payload::Cq {
                    from: self.my_call.clone(),
                    qtf_deg: None,
                    period: None,
                };
                action = Action::Transmit(self.make_tx(payload, &mut ev));
                self.transition(QsoState::CallingCq, &mut ev);
            }
            Intent::Call { their } => {
                self.reset_qso();
                self.their_call = Some(their.clone());
                self.mark_qso_start();
                ev.push(EngineEvent::TheirCallChanged { callsign: their.clone(), grid: None }); // 🟢 No grid for cold calls
                
                let payload = Payload::Call {
                    from: self.my_call.clone(),
                    to: their,
                };
                action = Action::Transmit(self.make_tx(payload, &mut ev));
                self.transition(QsoState::CallingStn, &mut ev);
            }
            Intent::AnswerCq { their, rpt, grid } => {
                self.reset_qso();
                self.their_call = Some(their.clone());
                self.their_grid = grid.clone(); // 🟢 NEW: Store grid from CQ
                self.my_report = rpt;
                self.mark_qso_start();
                ev.push(EngineEvent::TheirCallChanged { callsign: their.clone(), grid }); // 🟢 NEW: Emit grid
                
                let payload = Payload::CallWithReport {
                    from: self.my_call.clone(),
                    to: their,
                    rpt,
                };
                action = Action::Transmit(self.make_tx(payload, &mut ev));
                self.transition(QsoState::SendingReport, &mut ev);
            }
        }
        (action, ev)
    }

    pub fn on_rx(&mut self, rx: RxEnvelope) -> (Action, Vec<EngineEvent>) {
        let mut ev = vec![EngineEvent::Rx(rx.clone())];
        self.last_rx = Some(rx.clone());
        let from_call = rx.payload.from_call().to_string();
        
        let is_from_partner = self.their_call
            .as_ref()
            .map(|their| normalize_call(&from_call) == normalize_call(their))
            .unwrap_or(false);

        match (&self.state, &rx.payload) {
            (QsoState::Listening, Payload::Call { from, to }) if self.is_me(to) => {
                self.their_call = Some(from.clone());
                self.mark_qso_start();
                ev.push(EngineEvent::TheirCallChanged { callsign: from.clone(), grid: None }); // 🟢 No grid from calls
                let payload = Payload::CallWithReport {
                    from: self.my_call.clone(),
                    to: from.clone(),
                    rpt: self.my_report,
                };
                let action = Action::Transmit(self.make_tx(payload, &mut ev));
                self.transition(QsoState::SendingReport, &mut ev);
                return (action, ev);
            }
            (QsoState::Listening, Payload::CallWithReport { from, to, rpt }) if self.is_me(to) => {
                self.their_call = Some(from.clone());
                self.their_report = Some(*rpt);
                self.mark_qso_start();
                ev.push(EngineEvent::TheirCallChanged { callsign: from.clone(), grid: None }); // 🟢 No grid from calls
                let payload = Payload::RReport {
                    from: self.my_call.clone(),
                    to: from.clone(),
                    rpt: self.my_report,
                };
                let action = Action::Transmit(self.make_tx(payload, &mut ev));
                self.transition(QsoState::SendingRReport, &mut ev);
                return (action, ev);
            }
            (QsoState::CallingCq, Payload::CallWithReport { from, to, rpt }) if self.is_me(to) => {
                self.their_call = Some(from.clone());
                self.their_report = Some(*rpt);
                self.mark_qso_start();
                ev.push(EngineEvent::TheirCallChanged { callsign: from.clone(), grid: None }); // 🟢 No grid from calls
                let payload = Payload::RReport {
                    from: self.my_call.clone(),
                    to: from.clone(),
                    rpt: self.my_report,
                };
                let action = Action::Transmit(self.make_tx(payload, &mut ev));
                self.transition(QsoState::SendingRReport, &mut ev);
                return (action, ev);
            }
            (QsoState::CallingStn, Payload::CallWithReport { from, to, rpt }) 
                if self.is_me(to) && is_from_partner => {
                self.their_report = Some(*rpt);
                let payload = Payload::RReport {
                    from: self.my_call.clone(),
                    to: from.clone(),
                    rpt: self.my_report,
                };
                let action = Action::Transmit(self.make_tx(payload, &mut ev));
                self.transition(QsoState::SendingRReport, &mut ev);
                return (action, ev);
            }
            (QsoState::SendingReport, Payload::RReport { from, to, rpt }) 
                if self.is_me(to) && is_from_partner => {
                self.their_report = Some(*rpt);
                let payload = Payload::Rr {
                    from: self.my_call.clone(),
                    to: from.clone(),
                };
                let action = Action::Transmit(self.make_tx(payload, &mut ev));
                self.transition(QsoState::SendingRr, &mut ev);
                return (action, ev);
            }
            (QsoState::SendingRReport, Payload::Rr { from, to }) 
                if self.is_me(to) && is_from_partner => {
                self.tx_repeat_count = 0;
                let payload = Payload::SeventyThree {
                    from: self.my_call.clone(),
                    to: from.clone(),
                };
                let action = Action::Transmit(self.make_tx(payload, &mut ev));
                self.transition(QsoState::Sending73, &mut ev);
                return (action, ev);
            }
            (QsoState::SendingRr, Payload::SeventyThree { from, to }) 
                if self.is_me(to) && is_from_partner => {
                self.tx_repeat_count = 0;
                let payload = Payload::SeventyThree {
                    from: self.my_call.clone(),
                    to: from.clone(),
                };
                let action = Action::Transmit(self.make_tx(payload, &mut ev));
                self.transition(QsoState::Sending73, &mut ev);
                return (action, ev);
            }
            (QsoState::Sending73, Payload::SeventyThree { from, to }) 
                if self.is_me(to) && is_from_partner => {
                let their = from.clone();
                let record = self.make_qso_record();
                ev.push(EngineEvent::QsoComplete { their, record });
                self.transition(QsoState::Done, &mut ev);
                return (Action::None, ev);
            }
            // 🟢 NEW: Early termination - if they're calling CQ to someone else, they've moved on
            (QsoState::Sending73, Payload::Cq { from, .. }) 
                if is_from_partner => {
                ev.push(EngineEvent::Info("Partner calling CQ - terminating QSO early".into()));
                let their = from.clone();
                let record = self.make_qso_record();
                ev.push(EngineEvent::QsoComplete { their, record });
                self.transition(QsoState::Done, &mut ev);
                return (Action::None, ev);
            }
            // 🟢 NEW: Early termination - if they're calling someone specific, they've moved on
            (QsoState::Sending73, Payload::Call { from, .. }) 
                if is_from_partner => {
                ev.push(EngineEvent::Info("Partner calling someone else - terminating QSO early".into()));
                let their = from.clone();
                let record = self.make_qso_record();
                ev.push(EngineEvent::QsoComplete { their, record });
                self.transition(QsoState::Done, &mut ev);
                return (Action::None, ev);
            }
            _ => {}
        }
        (Action::None, ev)
    }

    pub fn next_tx(&mut self) -> Option<Payload> {
        match self.state {
            QsoState::CallingCq => Some(Payload::Cq { from: self.my_call.clone(), qtf_deg: None, period: None }),
            QsoState::CallingStn => Some(Payload::Call { from: self.my_call.clone(), to: self.their_call.clone()? }),
            QsoState::SendingReport => Some(Payload::CallWithReport { from: self.my_call.clone(), to: self.their_call.clone()?, rpt: self.my_report }),
            QsoState::SendingRReport => Some(Payload::RReport { from: self.my_call.clone(), to: self.their_call.clone()?, rpt: self.my_report }),
            QsoState::SendingRr => Some(Payload::Rr { from: self.my_call.clone(), to: self.their_call.clone()? }),
            QsoState::Sending73 => {
                self.tx_repeat_count += 1;
                if self.tx_repeat_count > self.max_repeats { return None; }
                Some(Payload::SeventyThree { from: self.my_call.clone(), to: self.their_call.clone()? })
            }
            _ => None,
        }
    }
    
    pub fn check_complete(&mut self) -> Option<EngineEvent> {
        if self.state == QsoState::Sending73 && self.tx_repeat_count > self.max_repeats {
            let their = self.their_call.clone().unwrap_or_default();
            let record = self.make_qso_record();
            self.state = QsoState::Done;
            return Some(EngineEvent::QsoComplete { their, record });
        }
        None
    }

    fn is_me(&self, call: &str) -> bool {
        normalize_call(call) == normalize_call(&self.my_call)
    }

    fn transition(&mut self, next: QsoState, ev: &mut Vec<EngineEvent>) {
        if self.state != next {
            self.state = next;
            ev.push(EngineEvent::StateChanged(next));
        }
    }

    fn make_tx(&mut self, payload: Payload, ev: &mut Vec<EngineEvent>) -> TxEnvelope {
        let format = payload.format();
        
        // 🟢 Convert Rendered enum to String for the UI envelope
        let raw = match render_payload(&payload) {
            Rendered::Text(s) => s,
            Rendered::Bits(_) => format!("CQ de {} [GRID]", payload.from_call()),
        };

        let tx = TxEnvelope { payload, format, raw };
        self.last_tx = Some(tx.clone());
        ev.push(EngineEvent::Tx(tx.clone()));
        tx
    }
    
    fn reset_qso(&mut self) {
        self.their_call = None;
        self.their_grid = None; // 🟢 NEW: Clear grid
        self.their_report = None;
        self.tx_repeat_count = 0;
        self.qso_start_utc_ms = None;
    }
}

fn normalize_call(s: &str) -> String {
    s.trim().to_uppercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // =========================================================================
    // SCENARIO ONE: A calls CQ
    // =========================================================================
    #[test]
    fn test_scenario_one_cq_caller() {
        let mut a = QsoEngine::new("GW4WND".into());
        
        // A: CQ
        let _ = a.on_intent(Intent::Cq);
        assert_eq!(a.state, QsoState::CallingCq);
        
        // A: receives "GW4WND de DK5HJ 26"
        let rx = RxEnvelope {
            payload: Payload::CallWithReport {
                from: "DK5HJ".into(),
                to: "GW4WND".into(),
                rpt: 26,
            },
            format: Format::Fmt1,
            snr: None,
            utc_ms: 0,
            rx_slot: 1,
        };
        let _ = a.on_rx(rx);
        assert_eq!(a.state, QsoState::SendingRReport);
        assert_eq!(a.their_call, Some("DK5HJ".into()));
        
        // A: next_tx returns R-report
        let tx = a.next_tx();
        assert!(matches!(tx, Some(Payload::RReport { .. })));
        
        // A: receives RR
        let rx = RxEnvelope {
            payload: Payload::Rr {
                from: "DK5HJ".into(),
                to: "GW4WND".into(),
            },
            format: Format::Fmt2,
            snr: None,
            utc_ms: 0,
            rx_slot: 1,
        };
        let _ = a.on_rx(rx);
        assert_eq!(a.state, QsoState::Sending73);
        
        // A: next_tx returns 73 (x5)
        for i in 1..=5 {
            let tx = a.next_tx();
            assert!(matches!(tx, Some(Payload::SeventyThree { .. })), "iteration {}", i);
        }
        
        // After 5, should return None
        let tx = a.next_tx();
        assert!(tx.is_none());
    }
    
    #[test]
    fn test_scenario_one_responder() {
        let mut b = QsoEngine::new("DK5HJ".into());
        b.my_report = 26;
        
        // B: listening, answers CQ
        let _ = b.on_intent(Intent::Listen);
        let _ = b.on_intent(Intent::AnswerCq { their: "GW4WND".into(), rpt: 26, grid: None });
        assert_eq!(b.state, QsoState::SendingReport);
        
        // B: next_tx returns CallWithReport
        let tx = b.next_tx();
        assert!(matches!(tx, Some(Payload::CallWithReport { .. })));
        
        // B: receives R-report
        let rx = RxEnvelope {
            payload: Payload::RReport {
                from: "GW4WND".into(),
                to: "DK5HJ".into(),
                rpt: 27,
            },
            format: Format::Fmt2,
            snr: None,
            utc_ms: 0,
            rx_slot: 0,
        };
        let _ = b.on_rx(rx);
        assert_eq!(b.state, QsoState::SendingRr);
        
        // B: next_tx returns RR
        let tx = b.next_tx();
        assert!(matches!(tx, Some(Payload::Rr { .. })));
        
        // B: receives 73
        let rx = RxEnvelope {
            payload: Payload::SeventyThree {
                from: "GW4WND".into(),
                to: "DK5HJ".into(),
            },
            format: Format::Fmt2,
            snr: None,
            utc_ms: 0,
            rx_slot: 0,
        };
        let _ = b.on_rx(rx);
        assert_eq!(b.state, QsoState::Sending73);
        
        // B: sends 73 x5
        for _ in 1..=5 {
            let tx = b.next_tx();
            assert!(matches!(tx, Some(Payload::SeventyThree { .. })));
        }
    }
    
    // =========================================================================
    // SCENARIO TWO: A receives cold call (no report)
    // =========================================================================
    #[test]
    fn test_scenario_two_receives_call() {
        let mut a = QsoEngine::new("GW4WND".into());
        a.my_report = 27;
        
        // A: listening
        let _ = a.on_intent(Intent::Listen);
        assert_eq!(a.state, QsoState::Listening);
        
        // A: receives "GW4WND de DK5HJ" (no report)
        let rx = RxEnvelope {
            payload: Payload::Call {
                from: "DK5HJ".into(),
                to: "GW4WND".into(),
            },
            format: Format::Fmt1,
            snr: None,
            utc_ms: 0,
            rx_slot: 1,
        };
        let _ = a.on_rx(rx);
        assert_eq!(a.state, QsoState::SendingReport);
        assert_eq!(a.their_call, Some("DK5HJ".into()));
        
        // A: next_tx returns CallWithReport
        let tx = a.next_tx();
        assert!(matches!(tx, Some(Payload::CallWithReport { rpt: 27, .. })));
    }
    
    // =========================================================================
    // SCENARIO THREE: A calls specific station
    // =========================================================================
    #[test]
    fn test_scenario_three_cold_call() {
        let mut a = QsoEngine::new("GW4WND".into());
        a.my_report = 27;
        
        // A: calls specific station
        let _ = a.on_intent(Intent::Call { their: "DK5HJ".into() });
        assert_eq!(a.state, QsoState::CallingStn);
        
        // A: next_tx returns Call (no report)
        let tx = a.next_tx();
        assert!(matches!(tx, Some(Payload::Call { .. })));
        
        // A: receives "GW4WND de DK5HJ 26"
        let rx = RxEnvelope {
            payload: Payload::CallWithReport {
                from: "DK5HJ".into(),
                to: "GW4WND".into(),
                rpt: 26,
            },
            format: Format::Fmt1,
            snr: None,
            utc_ms: 0,
            rx_slot: 1,
        };
        let _ = a.on_rx(rx);
        assert_eq!(a.state, QsoState::SendingRReport);
        
        // Rest follows same as scenario one...
    }
    
    #[test]
    fn test_scenario_three_receives_cold_call() {
        let mut b = QsoEngine::new("DK5HJ".into());
        b.my_report = 26;
        
        // B: listening
        let _ = b.on_intent(Intent::Listen);
        
        // B: receives "DK5HJ de GW4WND" (no report)
        let rx = RxEnvelope {
            payload: Payload::Call {
                from: "GW4WND".into(),
                to: "DK5HJ".into(),
            },
            format: Format::Fmt1,
            snr: None,
            utc_ms: 0,
            rx_slot: 0,
        };
        let _ = b.on_rx(rx);
        assert_eq!(b.state, QsoState::SendingReport);
        assert_eq!(b.their_call, Some("GW4WND".into()));
        
        // B: next_tx returns CallWithReport
        let tx = b.next_tx();
        assert!(matches!(tx, Some(Payload::CallWithReport { rpt: 26, .. })));
    }
    
    #[test]
    fn test_qso_record_generation() {
        let mut a = QsoEngine::new("GW4WND".into());
        a.set_band("2M".into());
        a.my_report = 27;
        
        // Start QSO
        let _ = a.on_intent(Intent::AnswerCq { their: "DK5HJ".into(), rpt: 27, grid: None });
        assert!(a.qso_start_utc_ms.is_some());
        
        // Simulate receiving R-report
        a.their_report = Some(26);
        
        // Generate record
        let record = a.make_qso_record();
        assert!(record.is_some());
        let rec = record.unwrap();
        assert_eq!(rec.call, "DK5HJ");
        assert_eq!(rec.operator, "GW4WND");
        assert_eq!(rec.band, "2M");
        assert_eq!(rec.rst_sent, "27");
        assert_eq!(rec.rst_rcvd, "26");
    }
}
