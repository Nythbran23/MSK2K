// src/fsk441rx/app.rs — FSK441 Transceiver GUI
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use std::path::PathBuf;
use std::process::Command;
use tokio::sync::mpsc;
use chrono::{DateTime, Utc};

mod params;
mod audio;
mod detector;
mod demod;
mod filter;
mod store;
mod geo;
mod tx;
mod qso;
mod spectrum;
mod accumulator;

use detector::run_detector;
use demod::longx;
use filter::ParsedMessage;
use store::Store;
use geo::{GeoValidator, PrefixDb, Qth};
use tx::{TxCommand, PeriodTimer, Period, SlotState, HamlibUpdate};
use spectrum::{compute_column, heat_color, FSK441_TONES_HZ, DISPLAY_BINS, hz_to_bin};
use accumulator::{FragmentAccumulator, Fragment, AccumulatedDecode};
use qso::{QsoState, on_decode, Transition};

// ─── Audio device enumeration (matches MSK2K) ─────────────────────────────────

fn enumerate_audio_devices() -> (Vec<String>, Vec<String>) {
    use cpal::traits::{DeviceTrait, HostTrait};
    use std::collections::HashMap;
    let host = cpal::default_host();

    // Use host.devices() — sees all devices including duplicate USB CODECs on macOS
    let device_list: Vec<cpal::Device> = {
        let from_all = host.devices().map(|d| d.collect::<Vec<_>>()).unwrap_or_default();
        if !from_all.is_empty() { from_all } else {
            let mut devs: Vec<cpal::Device> = Vec::new();
            if let Ok(d) = host.input_devices()  { devs.extend(d); }
            if let Ok(d) = host.output_devices() { devs.extend(d); }
            devs
        }
    };

    let mut all_devices: Vec<(String, bool, bool)> = Vec::new();
    for d in device_list {
        if let Ok(name) = d.name() {
            let has_in  = d.supported_input_configs().map(|mut c| c.next().is_some()).unwrap_or(false)
                       || d.default_input_config().is_ok();
            let has_out = d.supported_output_configs().map(|mut c| c.next().is_some()).unwrap_or(false)
                       || d.default_output_config().is_ok();
            all_devices.push((name, has_in, has_out));
        }
    }
    all_devices.sort_by(|a, b| a.0.cmp(&b.0));

    // Count duplicates — IC-9700 shows as two "USB Audio CODEC" entries, one RX one TX
    let mut name_counts: HashMap<String, usize> = HashMap::new();
    for (name, _, _) in &all_devices { *name_counts.entry(name.clone()).or_insert(0) += 1; }

    let mut group_caps: HashMap<String, Vec<(bool, bool)>> = HashMap::new();
    for (name, has_in, has_out) in &all_devices {
        if name_counts[name.as_str()] > 1 {
            group_caps.entry(name.clone()).or_default().push((*has_in, *has_out));
        }
    }

    let mut name_indices: HashMap<String, usize> = HashMap::new();
    let display_names: Vec<String> = all_devices.iter().map(|(name, has_in, has_out)| {
        if name_counts[name.as_str()] > 1 {
            let idx = name_indices.entry(name.clone()).or_insert(0);
            *idx += 1;
            let caps = &group_caps[name.as_str()];
            let has_rx_sib = caps.iter().any(|(i, _)| *i);
            let has_tx_sib = caps.iter().any(|(_, o)| *o);
            let label = match (*has_in, *has_out) {
                (true,  false) => "RX".to_string(),
                (false, true)  => "TX".to_string(),
                (true,  true)  => "RX/TX".to_string(),
                (false, false) => {
                    if has_rx_sib && !has_tx_sib { "TX".to_string() }
                    else if has_tx_sib && !has_rx_sib { "RX".to_string() }
                    else { format!("{}", idx) }
                }
            };
            format!("{} ({})", name, label)
        } else {
            name.clone()
        }
    }).collect();

    log::info!("[AUDIO] Devices: {:?}", display_names);
    // Same list for both input and output — user picks the (RX) instance for input
    // and the (TX) instance for output
    (display_names.clone(), display_names)
}

// ─── Rig list from rigctld -l (matches MSK2K) ────────────────────────────────

fn enumerate_rigs() -> Vec<(String, String)> {
    let mut list = Vec::new();
    // Try rigctld in PATH or bundled locations
    for cmd in &["rigctld", "/usr/local/bin/rigctld", "/opt/homebrew/bin/rigctld"] {
        if let Ok(out) = Command::new(cmd).arg("-l").output() {
            if let Ok(text) = String::from_utf8(out.stdout) {
                for line in text.lines() {
                    if line.trim().is_empty() || line.starts_with(" Rig") { continue; }
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 3 {
                        let id   = parts[0].to_string();
                        let name = format!("{} {}", parts[1], parts[2]);
                        list.push((id, name));
                    }
                }
                if !list.is_empty() { break; }
            }
        }
    }
    list
}

fn enumerate_serial_ports() -> Vec<String> {
    match serialport::available_ports() {
        Ok(ports) => ports.iter().map(|p| p.port_name.clone()).collect(),
        Err(_)    => vec![],
    }
}

// ─── Decode entry ─────────────────────────────────────────────────────────────

#[derive(Clone)]
pub(crate) struct DecodeEntry {
    timestamp:  DateTime<Utc>,
    df_hz:      f32,
    ccf_ratio:  f32,
    confidence: f32,
    score:      u8,
    raw:        String,
    callsigns:  Vec<String>,
    locator:    Option<String>,
    report:     Option<String>,
    is_cq:      bool,
    char_confs: Vec<f32>,
    is_accumulated: bool,
    #[allow(dead_code)]
    my_call:        String,
    df_bin_hz:      Option<i32>,
    passed_threshold: bool,  // false = shown only in RAW mode  // set for accumulated rows — used to update in place
}

enum EngineEvent { Decode(DecodeEntry), Accumulated(AccumulatedDecode), Hamlib(HamlibUpdate), Spectrum(Vec<f32>, f32) }

// ─── Settings ─────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Settings {
    #[allow(dead_code)]
    my_call:        String,
    my_loc:         String,
    sel_in:         Option<String>,
    sel_out:        Option<String>,
    hamlib_enabled: bool,
    their_call_hint: Option<String>,
    station_tracker_enabled: bool,
    station_min_pings:       usize,
    raw_mode:                bool,  // filter: 1=all, 2=confirmed, 5=strong
    rig_model:      String,
    rig_port:       String,
    rig_baud:       String,
    period:         Period,
    cty_path:       String,
    max_km:         f64,
    threshold:      f32,
    min_ccf:        f32,
    min_conf:       f32,
    tx_level:       f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            my_call:        "NOCALL".into(),
            my_loc:         "IO82KM".into(),
            sel_in:         Some("USB Audio CODEC".into()),
            sel_out:        Some("USB Audio CODEC".into()),
            hamlib_enabled: true,
            their_call_hint: None,
            station_tracker_enabled: true,
            station_min_pings: 1,
            raw_mode: false,
            rig_model:      String::new(),
            rig_port:       String::new(),
            rig_baud:       "19200".into(),
            period:         Period::TxSecond,
            cty_path:       "cty.dat".into(),
            max_km:         3000.0,
            threshold:      3.0,
            min_ccf:        300.0,
            min_conf:       0.70,
            tx_level:       0.8,
        }
    }
}

impl Settings {
    fn audio_in(&self) -> Option<String> { self.sel_in.clone() }
    fn audio_out(&self) -> Option<String> { self.sel_out.clone() }
    fn hamlib_addr(&self) -> Option<String> {
        if self.hamlib_enabled { Some("127.0.0.1:4532".into()) } else { None }
    }
}


// Ensures rigctld is killed cleanly when app exits — same as MSK2K
struct ProcessGuard(std::process::Child);

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        log::info!("[LAUNCHER] Shutting down rigctld (pid={})", self.0.id());
        // Release PTT cleanly before killing
        if let Ok(mut stream) = std::net::TcpStream::connect_timeout(
            &"127.0.0.1:4532".parse().unwrap(),
            std::time::Duration::from_millis(500),
        ) {
            use std::io::Write;
            let _ = stream.write_all(b"T 0\n");
            let _ = stream.flush();
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}


// ─── Station Tracker ──────────────────────────────────────────────────────────
// Persistently accumulates soft evidence across ALL pings at a given DF bin,
// surviving slot boundaries. Shows running reconstruction in the callsign panel.
#[derive(Default, Clone)]
pub struct TrackedStation {
    pub df_bin:     i32,        // DF rounded to nearest 43Hz bin
    pub ping_count: usize,
    pub last_seen:  Option<chrono::DateTime<chrono::Utc>>,
    pub first_seen: Option<chrono::DateTime<chrono::Utc>>,
    pub best_decode: String,    // highest-confidence single decode seen
    pub best_conf:  f32,
    pub callsigns:  Vec<(String, usize)>, // call → count
}

#[derive(Default)]
pub struct StationTracker {
    pub stations: Vec<TrackedStation>,
}

impl StationTracker {
    pub fn add_ping(&mut self, entry: &DecodeEntry) {
        let df_bin = (entry.df_hz / 43.0).round() as i32 * 43;
        let st = if let Some(s) = self.stations.iter_mut().find(|s| s.df_bin == df_bin) {
            s
        } else {
            self.stations.push(TrackedStation {
                df_bin, ..Default::default()
            });
            self.stations.last_mut().unwrap()
        };
        st.ping_count += 1;
        st.last_seen  = Some(entry.timestamp);
        if st.first_seen.is_none() { st.first_seen = Some(entry.timestamp); }
        // Keep best single-ping decode
        if entry.confidence > st.best_conf && !entry.raw.trim().is_empty() {
            st.best_conf   = entry.confidence;
            st.best_decode = entry.raw.trim().to_string();
        }
        // Accumulate callsigns — merge fragments into longest known form
        // e.g. I5Y + I5YD + I5YDI → all count under I5YDI
        for call in &entry.callsigns {
            // Check if this call is a prefix of an existing longer call
            let existing_longer = st.callsigns.iter_mut()
                .find(|(c,_)| c.starts_with(call.as_str()) && c.len() > call.len());
            if let Some(e) = existing_longer {
                e.1 += 1;
                continue;
            }
            // Check if this call extends a shorter existing call
            if let Some(e) = st.callsigns.iter_mut()
                .find(|(c,_)| call.starts_with(c.as_str()) && call.len() > c.len())
            {
                let count = e.1 + 1;
                let _ = std::mem::replace(e, (call.clone(), count));
                continue;
            }
            // New callsign
            if let Some(c) = st.callsigns.iter_mut().find(|(c,_)| c == call) {
                c.1 += 1;
            } else {
                st.callsigns.push((call.clone(), 1));
            }
        }
    }

    pub fn clear(&mut self) { self.stations.clear(); }

    // Stations active in last N seconds
    pub fn active(&self, secs: i64) -> Vec<&TrackedStation> {
        let cutoff = chrono::Utc::now() - chrono::Duration::seconds(secs);
        let mut active: Vec<&TrackedStation> = self.stations.iter()
            .filter(|s| s.last_seen.map(|t| t > cutoff).unwrap_or(false))
            .collect();
        active.sort_by(|a, b| b.ping_count.cmp(&a.ping_count));
        active
    }
}

/// One entry in the DF scatter strip — a confirmed ping dot.
#[derive(Clone)]
struct DfDot {
    /// X position: column index when the ping was detected
    col_idx:  usize,
    /// DF in Hz (signed, relative to nominal carrier)
    df_hz:    f32,
    /// Colour: confidence-coded (green high, yellow mid, grey low)
    r: u8, g: u8, b: u8,
}

// ─── App ──────────────────────────────────────────────────────────────────────

struct Fsk441App {
    settings:        Settings,
    settings_open:   bool,
    in_devs:         Vec<String>,
    out_devs:        Vec<String>,
    rig_list:        Vec<(String, String)>,
    rig_search:      String,
    serial_ports:    Vec<String>,
    decodes:         Vec<DecodeEntry>,
    selected:        Option<usize>,
    event_rx:        mpsc::UnboundedReceiver<EngineEvent>,
    tx_cmd_tx:       mpsc::UnboundedSender<TxCommand>,
    period_timer:    PeriodTimer,
    is_transmitting: bool,
    qso:             QsoState,
    their_call_edit: String,
    their_loc_edit:  String,
    report_sent:     String,
    report_rcvd:     String,
    // Editable TX message fields — shown in UI, user can override defaults
    tx_msgs:         [String; 6],  // [CQ, TX1, TX2, TX3, TX4, TX5]
    tx_active:       std::sync::Arc<std::sync::atomic::AtomicBool>,
    // Spectrogram: columns[i] = one FFT snapshot (vertical strip), X=time Y=freq
    show_accumulated: bool,
    active_tx_idx:   Option<usize>,  // which TX button is active
    wf_columns:      Vec<Vec<f32>>,
    df_dots:         Vec<DfDot>,   // DF scatter strip history
    wf_amplitude:    Vec<f32>,  // RMS amplitude per column (0..1)
    wf_period_idx:   i64,  // slot index when columns last cleared
    tx_msgs_key:     String,       // last key used to populate — repopulate only on change
    qso_log:         Vec<String>,
    seen_calls:      Vec<(String, chrono::DateTime<chrono::Utc>, usize)>,
    station_tracker: StationTracker,
    qso_summary:     Option<String>,
    rig_freq_hz:     Option<u64>,
    cat_connected:   bool,
    last_cat_rx:     Option<std::time::Instant>,
    _runtime:        tokio::runtime::Runtime,
    _rigctld:        Option<ProcessGuard>,
    settings_watch_tx: tokio::sync::watch::Sender<Settings>,
}

impl Fsk441App {
    fn new(_cc: &eframe::CreationContext) -> Self {
        let settings = load_config();
        let (in_devs, out_devs) = enumerate_audio_devices();
        let rig_list = enumerate_rigs();
        let serial_ports = enumerate_serial_ports();

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all().build().expect("Tokio runtime");

        let (event_tx, event_rx) = mpsc::unbounded_channel::<EngineEvent>();
        let (tx_cmd_tx, tx_cmd_rx) = mpsc::unbounded_channel::<TxCommand>();
        let (hamlib_tx, hamlib_rx)   = mpsc::unbounded_channel::<HamlibUpdate>();

        let s = settings.clone();
        let (settings_watch_tx, settings_watch_rx) = tokio::sync::watch::channel(s.clone());

        let tx_engine = tx::TxEngine::new(
            tx_cmd_rx, settings.period,
            settings.audio_out(),
            settings.hamlib_addr(),
            hamlib_tx,
        );
        rt.spawn(async move { tx_engine.run().await; });

        // Forward HamlibUpdates from TxEngine into main event loop
        let htx = event_tx.clone();
        rt.spawn(async move {
            let mut hx = hamlib_rx;
            while let Some(upd) = hx.recv().await {
                let _ = htx.send(EngineEvent::Hamlib(upd));
            }
        });

        // Spawn RX engine
        let tx_active_for_engine = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let tx_active_for_app    = tx_active_for_engine.clone();
        let et = event_tx.clone();
        rt.spawn(async move { run_engine(s, et, tx_active_for_engine, settings_watch_rx).await; });

        // Auto-launch rigctld if hamlib enabled and rig configured — same as MSK2K
        let mut rigctld_guard: Option<ProcessGuard> = None;
        if settings.hamlib_enabled
            && !settings.rig_model.is_empty()
            && !settings.rig_port.is_empty()
        {
            let baud: u32 = settings.rig_baud.parse().unwrap_or(19200);
            log::info!("[LAUNCHER] Auto-starting rigctld: model={} port={} baud={}",
                settings.rig_model, settings.rig_port, baud);
            // Kill any stale instance before launching
            let _ = Command::new("pkill").args(["-f", "rigctld"]).output();
            std::thread::sleep(std::time::Duration::from_millis(300));
            match Command::new("rigctld")
                .args(["-m", &settings.rig_model,
                       "-r", &settings.rig_port,
                       "-s", &baud.to_string(),
                       "-P", "RIG"])
                .spawn()
            {
                Ok(c)  => {
                    log::info!("[LAUNCHER] rigctld started (pid={})", c.id());
                    rigctld_guard = Some(ProcessGuard(c));
                }
                Err(e) => log::error!("[LAUNCHER] rigctld failed: {}", e),
            }
            // Give rigctld time to bind port 4532
            std::thread::sleep(std::time::Duration::from_millis(800));
        }

        let period = settings.period;
        Self {
            settings_open: false, in_devs, out_devs, rig_list,
            rig_search: String::new(), serial_ports,
            decodes: Vec::new(), selected: None, event_rx, tx_cmd_tx,
            period_timer: PeriodTimer::new(period), is_transmitting: false,
            qso: QsoState::Idle,
            their_call_edit: String::new(), their_loc_edit: String::new(),
            report_sent: "26".into(), report_rcvd: String::new(),
            tx_active: tx_active_for_app,
        show_accumulated: true,
        active_tx_idx: None,
        wf_columns: Vec::new(),
        df_dots: Vec::new(),
        wf_amplitude: Vec::new(),
        wf_period_idx: -1,
        tx_msgs: Default::default(), tx_msgs_key: String::new(), qso_log: Vec::new(), seen_calls: Vec::new(), station_tracker: StationTracker::default(), qso_summary: None, settings_watch_tx, rig_freq_hz: None, cat_connected: false, last_cat_rx: None, _rigctld: rigctld_guard, settings, _runtime: rt,
        }
    }

    fn refresh_audio_devices(&mut self) {
        let (i, o) = enumerate_audio_devices();
        if let Some(ref sel) = self.settings.sel_in.clone() {
            if !i.contains(sel) { self.settings.sel_in = None; }
        }
        if let Some(ref sel) = self.settings.sel_out.clone() {
            if !o.contains(sel) { self.settings.sel_out = None; }
        }
        self.in_devs = i; self.out_devs = o;
    }

    fn poll_events(&mut self) {
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                EngineEvent::Accumulated(acc) => {
                    // Accumulated decode: prepend with ★ marker and different colour
                    let entry = DecodeEntry {
                        timestamp:  chrono::Utc::now(),
                        df_hz:      0.0,
                        ccf_ratio:  acc.n_fragments as f32 * 100.0,
                        confidence: acc.mean_conf,
                        score:      70u8,  // always show accumulated — confidence shown in text
                        raw:        format!("★ {}", acc.text),
                        callsigns:  vec![],
                        locator:    None,
                        report:     None,
                        is_cq:      acc.text.contains(" CQ ") || acc.text.starts_with("CQ "),
                        char_confs: acc.char_conf,
                        is_accumulated: true,
                        my_call: self.settings.my_call.clone(),
                        df_bin_hz: Some(acc.df_bin_hz),
                        passed_threshold: true,
                    };
                    // Update seen callsigns — exclude MYCALL
                    let my = self.settings.my_call.to_uppercase();
                    for call in &entry.callsigns {
                        if call.eq_ignore_ascii_case(&my) { continue; }
                        if let Some(e) = self.seen_calls.iter_mut().find(|(c,_,_)| c == call) {
                            e.2 += 1;
                        } else {
                            self.seen_calls.push((call.clone(), entry.timestamp, 1));
                        }
                    }
                    // Update station tracker
                    if self.settings.station_tracker_enabled {
                        self.station_tracker.add_ping(&entry);
                    }
                    // Update existing ★ row for this DF bin in place, or append
                    let df_key = entry.df_bin_hz;
                    if let Some(pos) = self.decodes.iter().position(|e|
                        e.is_accumulated && e.df_bin_hz.is_some() && e.df_bin_hz == df_key
                    ) {
                        self.decodes[pos] = entry.clone();
                    } else {
                        self.decodes.push(entry.clone());
                        if self.decodes.len() > 500 { self.decodes.remove(0); }
                    }
                    // DF scatter dot for accumulated entries
                    let (dr,dg,db) = if entry.confidence >= 0.75 { (80,220,80) }
                        else if entry.confidence >= 0.50 { (220,200,80) }
                        else { (120,120,120) };
                    self.df_dots.push(DfDot { col_idx: self.wf_columns.len(), df_hz: entry.df_hz, r:dr,g:dg,b:db });
                    if self.df_dots.len() > 2000 { self.df_dots.remove(0); }
                    continue;
                }
                EngineEvent::Spectrum(bins, rms) => {
                    let now_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64;
                    let slot = now_ms / 30_000;
                    if slot != self.wf_period_idx {
                        self.wf_columns.clear();
                        self.wf_amplitude.clear();
                        self.wf_period_idx = slot;
                    }
                    self.wf_columns.push(bins);
                    self.wf_amplitude.push(rms);
                    continue;
                }
                EngineEvent::Hamlib(upd) => {
                    if let Some(f) = upd.freq {
                        self.rig_freq_hz  = Some(f);
                        self.cat_connected = true;
                        self.last_cat_rx  = Some(std::time::Instant::now());
                    }
                    if let Some(false) = upd.connected {
                        self.cat_connected = false;
                        self.rig_freq_hz   = None;
                    }
                    if upd.transmitting != self.is_transmitting {
                        self.is_transmitting = upd.transmitting;
                        self.tx_active.store(upd.transmitting, std::sync::atomic::Ordering::Relaxed);
                    }
                    continue;
                }
                EngineEvent::Decode(entry) => {
                    // Discard loopback during TX
                    if self.is_transmitting { continue; }
            if self.qso.is_active() {
                let parsed = ParsedMessage {
                    raw: entry.raw.clone(), callsigns: entry.callsigns.clone(),
                    locator: entry.locator.clone(), report: entry.report.clone(),
                    is_cq: entry.is_cq, message_type: filter::MessageType::Garbage,
                    validity_score: entry.score, valid_callsigns: entry.callsigns.clone(),
                };
                let mc = self.settings.my_call.clone();
                let rs = self.report_sent.clone();
                log::info!("[APP] on_decode: qso={:?} raw={}", std::mem::discriminant(&self.qso), parsed.raw);
                let t  = on_decode(&mut self.qso, &parsed, &mc, move || rs.clone());
                log::info!("[APP] on_decode result: {:?}", t);
                match t {
                    Transition::Auto(msg) => {
                        self.qso_log.push(format!("AUTO: {}", msg));
                        if let Some(tx_msg) = self.qso.tx_message() {
                            let dev = self.settings.audio_out();
                            let _ = self.tx_cmd_tx.send(TxCommand::Transmit { message: tx_msg, output_device: dev });
                        }
                    }
                    Transition::Complete => {
                        log::info!("[APP] HALT from Transition::Complete");
                        self.qso_log.push("QSO COMPLETE — click LOG QSO".to_string());
                        let _ = self.tx_cmd_tx.send(TxCommand::Halt);
                        self.is_transmitting = false;
                    }
                    Transition::Count73(0) => {
                        log::info!("[APP] HALT from Count73(0)");
                        let _ = self.tx_cmd_tx.send(TxCommand::Halt);
                        self.is_transmitting = false;
                    }
                    _ => {}
                }
            }
            // Update seen callsigns
            for call in &entry.callsigns {
                // Merge fragments: I5Y / I5YD / I5YDI → longest form
                let longer = self.seen_calls.iter_mut()
                    .find(|(c,_,_)| c.starts_with(call.as_str()) && c.len() > call.len());
                if let Some(e) = longer { e.2 += 1; continue; }
                if let Some(e) = self.seen_calls.iter_mut()
                    .find(|(c,_,_)| call.starts_with(c.as_str()) && call.len() > c.len())
                {
                    let (_, ts, n) = e.clone();
                    *e = (call.clone(), ts, n + 1);
                    continue;
                }
                if let Some(e) = self.seen_calls.iter_mut().find(|(c,_,_)| c == call) {
                    e.2 += 1;
                } else {
                    self.seen_calls.push((call.clone(), entry.timestamp, 1));
                }
            }
            // Update station tracker
            if self.settings.station_tracker_enabled {
                self.station_tracker.add_ping(&entry);
            }
            self.decodes.push(entry.clone());
            if self.decodes.len() > 500 { self.decodes.remove(0); }
            // DF scatter dot
            let (dr,dg,db) = if entry.confidence >= 0.75 { (80,220,80) }
                else if entry.confidence >= 0.50 { (220,200,80) }
                else { (120,120,120) };
            self.df_dots.push(DfDot { col_idx: self.wf_columns.len(), df_hz: entry.df_hz, r:dr,g:dg,b:db });
            if self.df_dots.len() > 2000 { self.df_dots.remove(0); }
            } // end Decode arm
            } // end match
        }
        // Filter — don't process decodes while transmitting (half duplex)
        if self.is_transmitting { self.decodes.retain(|_| true); }
    }

    fn populate_tx_msgs(&mut self) {
        let mc   = self.settings.my_call.clone();
        let ml   = self.settings.my_loc.clone();
        let tc   = self.their_call_edit.trim().to_uppercase();
        let rs   = self.report_sent.clone();
        let loc4 = ml[..ml.len().min(4)].to_string();

        // Only repopulate when callsign/locator/report changes — preserve user edits
        let key = format!("{},{},{},{},{}", mc, loc4, tc, rs, self.settings.my_loc);
        if key == self.tx_msgs_key { return; }
        self.tx_msgs_key = key;

        // CQ format: "CQ MY_CALL MY_GRID4" (CQ first — standard FSK441)
        self.tx_msgs[0] = format!("CQ {} {}", mc, loc4);
        // TX1: THEIR_CALL MY_CALL (no locator)
        self.tx_msgs[1] = if tc.is_empty() { format!("<CALL> {}", mc) }
                          else { format!("{} {}", tc, mc) };
        // TX2: THEIR_CALL MY_CALL REPORT REPORT
        self.tx_msgs[2] = if tc.is_empty() { format!("<CALL> {} {} {}", mc, rs, rs) }
                          else { format!("{} {} {} {}", tc, mc, rs, rs) };
        // TX3: THEIR_CALL MY_CALL R MY_REPORT
        self.tx_msgs[3] = if tc.is_empty() { format!("<CALL> {} R{} R{}", mc, rs, rs) }
                          else { format!("{} {} R{} R{}", tc, mc, rs, rs) };
        // TX4: THEIR_CALL MY_CALL RRR
        self.tx_msgs[4] = if tc.is_empty() { format!("<CALL> {} RRR RRR", mc) }
                          else { format!("{} {} RRR RRR", tc, mc) };
        // TX5: THEIR_CALL MY_CALL 73
        self.tx_msgs[5] = if tc.is_empty() { format!("<CALL> {} 73 73 73", mc) }
                          else { format!("{} {} 73 73 73", tc, mc) };
    }

    fn send_tx(&mut self, msg: &str) {
        let dev = self.settings.audio_out();
        let _ = self.tx_cmd_tx.send(TxCommand::Transmit {
            message: msg.to_string(),
            output_device: dev,
        });
        // Set tx_active immediately so detector/spectrum gate fires without UI lag
        self.is_transmitting = true;
        self.tx_active.store(true, std::sync::atomic::Ordering::Relaxed);
    }


    fn clear_decodes(&mut self) {
        self.decodes.clear();
        self.seen_calls.clear();
        self.station_tracker.clear();
        self.df_dots.clear();
        self.qso_summary = None;
        self.selected = None;
    }

    fn generate_qso_summary(&mut self) {
        let tc = self.their_call_edit.trim().to_uppercase();
        let tl = self.their_loc_edit.trim().to_uppercase();
        let mc = self.settings.my_call.trim().to_uppercase();
        let ml = self.settings.my_loc.trim().to_uppercase();
        let rs = self.report_sent.trim().to_string();
        let rr = self.report_rcvd.trim().to_string();
        let their_pings: Vec<&DecodeEntry> = self.decodes.iter()
            .filter(|e| e.callsigns.iter().any(|c| c.contains(&tc) || tc.contains(c.as_str())))
            .collect();
        let first_ping = their_pings.first()
            .map(|e| e.timestamp.format("%H:%M:%S").to_string())
            .unwrap_or_else(|| "?".to_string());
        let last_ping = their_pings.last()
            .map(|e| e.timestamp.format("%H:%M:%S").to_string())
            .unwrap_or_else(|| "?".to_string());
        let n_pings  = their_pings.len();
        let peak_ccf = their_pings.iter().map(|e| e.ccf_ratio as u64).max().unwrap_or(0);
        let now = chrono::Utc::now();
        let summary = format!(
            "QSO {}\n{} ↔ {}\nMy: {} Their: {}\nRpt: {} / {}\n{} pings {}-{}\nPeak CCF: {}",
            now.format("%Y-%m-%d %H:%M UTC"),
            mc, tc,
            ml, if tl.is_empty() { "?".to_string() } else { tl },
            if rs.is_empty() { "?".to_string() } else { rs },
            if rr.is_empty() { "?".to_string() } else { rr },
            n_pings, first_ping, last_ping,
            peak_ccf,
        );
        log::info!("[QSO] Summary: {}", summary);
        self.qso_summary = Some(summary);
    }

    fn halt_tx(&mut self) {
        log::info!("[APP] HALT from halt_tx()");
        let _ = self.tx_cmd_tx.send(TxCommand::Halt);
        self.is_transmitting = false;
        // Ungate the audio fanout so spectrum and detector resume immediately
        self.tx_active.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

impl eframe::App for Fsk441App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_events();
        // Click on empty space in central panel deselects
        if ctx.input(|i| i.pointer.any_click()) {
            if !ctx.is_pointer_over_area() {
                // pointer not over any egui widget — deselect
            }
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(200));

        // ── Top bar ───────────────────────────────────────────────────────
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                // FSK441 + callsign
                ui.label(egui::RichText::new("FSK441")
                    .strong().color(egui::Color32::from_rgb(0, 150, 255)));
                ui.separator();
                ui.label(egui::RichText::new(
                    format!("{} {}", self.settings.my_call, self.settings.my_loc))
                    .strong());
                ui.separator();

                // TX/RX slot — always show when a TX is queued or active
                if self.active_tx_idx.is_some() || self.is_transmitting {
                    let (slot, remaining) = self.period_timer.current_slot();
                    ui.label(match slot {
                        SlotState::Tx => egui::RichText::new(format!("■ TX {:02}s", remaining))
                            .color(egui::Color32::RED).strong(),
                        SlotState::Rx => egui::RichText::new(format!("○ RX {:02}s", remaining))
                            .color(egui::Color32::from_rgb(0, 200, 100)),
                    });
                    ui.separator();
                }

                // Frequency display — MSK2K style
                // Connected + freq → green "144.370.00 MHz"
                // Not connected    → grey "No CAT"
                if self.cat_connected {
                    if let Some(freq) = self.rig_freq_hz {
                        let mhz  = freq / 1_000_000;
                        let khz  = (freq % 1_000_000) / 1_000;
                        let hz   = (freq % 1_000) / 10;
                        let col  = egui::Color32::from_rgb(100, 200, 130);
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 0.0;
                            ui.label(egui::RichText::new(format!("{}.{:03}.", mhz, khz))
                                .monospace().size(14.0).color(col));
                            ui.vertical(|ui| {
                                ui.add_space(4.0);
                                ui.label(egui::RichText::new(format!("{:02}", hz))
                                    .monospace().size(10.0).color(col));
                            });
                            ui.label(egui::RichText::new(" MHz")
                                .monospace().size(14.0).color(col));
                        });
                    }
                } else {
                    ui.label(egui::RichText::new("No CAT")
                        .monospace().color(egui::Color32::GRAY));
                }
                ui.separator();

                // Period selector inline
                egui::ComboBox::from_id_salt("period_top")
                    .width(115.0)
                    .selected_text(match self.settings.period {
                        Period::TxFirst    => "TX 1st 30s",
                        Period::TxSecond   => "TX 2nd 30s",
                        Period::TxFirst15  => "TX 1st 15s",
                        Period::TxSecond15 => "TX 2nd 15s",
                    })
                    .show_ui(ui, |ui| {
                        for (p, label) in [
                            (Period::TxFirst,    "TX 1st 30s"),
                            (Period::TxSecond,   "TX 2nd 30s"),
                            (Period::TxFirst15,  "TX 1st 15s"),
                            (Period::TxSecond15, "TX 2nd 15s"),
                        ] {
                            if ui.selectable_value(&mut self.settings.period, p, label).clicked() {
                                self.period_timer = PeriodTimer::new(p);
                                let _ = self.tx_cmd_tx.send(TxCommand::SetPeriod(p));
                                save_config(&self.settings);
                            }
                        }
                    });
                ui.separator();

                // UTC clock + slot countdown
                let now_utc   = chrono::Utc::now();
                ui.label(egui::RichText::new(now_utc.format("%H:%M:%S").to_string())
                    .monospace().color(egui::Color32::from_gray(200)));
                // Slot countdown — red=TX slot, green=RX slot
                {
                    let ts        = now_utc.timestamp();
                    let slot_secs = match self.settings.period {
                        Period::TxFirst | Period::TxSecond     => 30i64,
                        Period::TxFirst15 | Period::TxSecond15 => 15i64,
                    };
                    let tx_first = matches!(self.settings.period,
                        Period::TxFirst | Period::TxFirst15);
                    let cycle_pos = ts % (slot_secs * 2);
                    let in_tx = if tx_first { cycle_pos < slot_secs }
                                else        { cycle_pos >= slot_secs };
                    let remain = slot_secs - (cycle_pos % slot_secs);
                    let col = if in_tx { egui::Color32::from_rgb(220, 80, 80) }
                              else     { egui::Color32::from_rgb(80, 180, 80) };
                    ui.label(egui::RichText::new(format!("{}s", remain))
                        .monospace().size(13.0).color(col));
                }

                // Settings far right
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("⚙ Settings").clicked() {
                        self.settings_open = !self.settings_open;
                        if self.settings_open { self.refresh_audio_devices(); }
                    }
                });
            });
        });

        // ── QSO panel (bottom) ────────────────────────────────────────────
        egui::TopBottomPanel::bottom("qso").min_height(190.0).show(ctx, |ui| {
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(egui::RichText::new("Rpt Sent").small());
                    egui::ComboBox::from_id_salt("rpt_s")
                        .width(55.0).selected_text(&self.report_sent)
                        .show_ui(ui, |ui| {
                            for r in &["26","27","28","29","36","37","47","57","59"] {
                                if ui.selectable_value(&mut self.report_sent, r.to_string(), *r).clicked() {
                                    self.tx_msgs_key.clear();
                                }
                            }
                        });
                });
                ui.separator();
                ui.vertical(|ui| {
                    ui.label(egui::RichText::new("Their Call").small());
                    // Lock fields when Their Call is set (QSO in progress)
                    let in_qso = self.settings.their_call_hint.is_some();
                    if ui.add(egui::TextEdit::singleline(&mut self.their_call_edit)
                        .desired_width(90.0).hint_text("eg I5YDI")
                        .interactive(!in_qso)).changed()
                    {
                        self.tx_msgs_key.clear();
                        self.settings.their_call_hint = if self.their_call_edit.is_empty() {
                            None
                        } else {
                            Some(self.their_call_edit.trim().to_uppercase())
                        };
                        let _ = self.settings_watch_tx.send(self.settings.clone());
                    }
                    if in_qso {
                        ui.label(egui::RichText::new("🔒").small().color(egui::Color32::YELLOW));
                    }
                });
                ui.vertical(|ui| {
                    ui.label(egui::RichText::new("Their Loc").small());
                    ui.add(egui::TextEdit::singleline(&mut self.their_loc_edit)
                        .desired_width(55.0).hint_text("JN53"));
                });
                ui.vertical(|ui| {
                    ui.label(egui::RichText::new("Rpt Rcvd").small());
                    ui.add(egui::TextEdit::singleline(&mut self.report_rcvd)
                        .desired_width(40.0).hint_text("26"));
                });
            });

            ui.add_space(5.0);
            ui.separator();
            ui.add_space(3.0);

            // ── Editable TX message fields — WSJT style ───────────────────
            // Auto-populate defaults from callsign/locator/report fields.
            // User can edit any field freely before clicking its button.
            self.populate_tx_msgs();

            let labels = ["CQ", "TX1", "TX2", "TX3", "TX4", "TX5"];
            let _active_idx: Option<usize> = match &self.qso {
                QsoState::CallingCq { .. }     => Some(0),
                QsoState::CallingStation { .. } => Some(1),
                QsoState::SendingReport { .. }  => Some(2),
                QsoState::SendingRReport { .. } => Some(3),
                QsoState::SendingRR { .. }      => Some(4),
                QsoState::Sending73 { .. }      => Some(5),
                _                               => None,
            };

            for i in 0..6usize {
                ui.horizontal(|ui| {
                    let is_queued = self.active_tx_idx == Some(i);
                    let is_txing  = self.is_transmitting && is_queued;
                    let btn_color = if is_txing {
                        egui::Color32::from_rgb(255, 80, 80)     // red = actively transmitting
                    } else if is_queued {
                        egui::Color32::from_rgb(255, 200, 0)     // amber = queued for next slot
                    } else {
                        egui::Color32::from_rgb(160, 160, 160)   // grey = idle
                    };
                    let btn = egui::Button::new(
                        egui::RichText::new(labels[i]).strong().color(btn_color)
                    ).min_size(egui::vec2(35.0, 0.0));

                    if ui.add(btn).clicked() {
                        let msg = self.tx_msgs[i].clone();
                        // Don't transmit if message still has placeholder
                        if msg.contains("<CALL>") {
                            self.qso_log.push("⚠ Enter Their Call first".to_string());
                        } else {
                            log::info!("[APP] {} clicked: {}", labels[i], msg);
                            self.qso_log.push(format!("{}: {}", labels[i], msg));
                            self.active_tx_idx = Some(i);
                            self.send_tx(&msg);
                            // TX5 = 73 sent — generate QSO summary
                            if i == 5 {
                                self.generate_qso_summary();
                            }
                        }
                    }

                    ui.add(egui::TextEdit::singleline(&mut self.tx_msgs[i])
                        .desired_width(360.0)
                        .font(egui::TextStyle::Monospace));

                    // HALT and LOG on same row as CQ
                    if i == 0 {
                        ui.separator();
                        if ui.button(
                            egui::RichText::new("■ HALT").color(egui::Color32::RED).strong()
                        ).clicked() {
                            log::info!("[APP] HALT button clicked");
                            self.halt_tx();
                            self.active_tx_idx = None;
                            self.qso = QsoState::Idle;
                            self.qso_log.push("HALTED".to_string());
                        }
                        if ui.button(egui::RichText::new("⟳ Clear")
                            .color(egui::Color32::from_gray(180))).clicked()
                        {
                            self.their_call_edit.clear();
                            self.their_loc_edit.clear();
                            self.report_rcvd = "26".to_string();
                            self.active_tx_idx = None;
                            self.tx_msgs_key.clear();
                            self.halt_tx();
                            self.qso = QsoState::Idle;
                            self.qso_log.push("--- Cleared ---".to_string());
                        }

                        if matches!(&self.qso, QsoState::Complete { .. }) {
                            if ui.button(
                                egui::RichText::new("📋 LOG").color(egui::Color32::GREEN).strong()
                            ).clicked() {
                                if let QsoState::Complete { their_call, their_loc,
                                                             report_sent, report_rcvd } = &self.qso
                                {
                                    self.qso_log.push(format!(
                                        "LOGGED: {} {} S:{} R:{}",
                                        their_call, their_loc, report_sent, report_rcvd
                                    ));
                                }
                                self.qso = QsoState::Idle;
                                self.their_call_edit.clear();
                                self.their_loc_edit.clear();
                            }
                        }
                    }
                });
            }

            ui.add_space(4.0);

            if !self.qso_log.is_empty() {
                ui.separator();
                let start = self.qso_log.len().saturating_sub(4);
                for e in &self.qso_log[start..] {
                    ui.label(egui::RichText::new(e).small()
                        .color(egui::Color32::from_rgb(160, 160, 160)));
                }
            }
        });

        // ── Decode list ───────────────────────────────────────────────────
        // ─── Callsign / Station Tracker panel (right side) ──────────────────────
        egui::SidePanel::right("callsign_panel")
            .min_width(180.0)
            .max_width(240.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Stations Heard").strong().small());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let tracker_label = if self.settings.station_tracker_enabled {
                            egui::RichText::new("Tracker ON").small()
                                .color(egui::Color32::from_rgb(100, 220, 100))
                        } else {
                            egui::RichText::new("Tracker OFF").small()
                                .color(egui::Color32::from_gray(140))
                        };
                        if ui.small_button(tracker_label).clicked() {
                            self.settings.station_tracker_enabled =
                                !self.settings.station_tracker_enabled;
                        }
                    });
                });
                // Filter buttons: All / ≥2 / ≥5
                ui.horizontal(|ui| {
                    for (label, val) in [("All",1usize),("≥2",2),("≥5",5)] {
                        let selected = self.settings.station_min_pings == val;
                        let col = if selected {
                            egui::Color32::from_rgb(100, 180, 255)
                        } else {
                            egui::Color32::from_gray(160)
                        };
                        if ui.add(egui::Button::new(
                            egui::RichText::new(label).small().color(col)
                        ).small().frame(selected)).clicked() {
                            self.settings.station_min_pings = val;
                        }
                    }
                    ui.label(egui::RichText::new("pings").small()
                        .color(egui::Color32::from_gray(120)));
                });
                ui.separator();

                // QSO summary if available
                if let Some(ref summary) = self.qso_summary {
                    egui::ScrollArea::vertical()
                        .id_salt("qso_summary_scroll")
                        .max_height(100.0)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new(summary)
                                .small().color(egui::Color32::from_rgb(100, 220, 130)));
                        });
                    ui.separator();
                }

                egui::ScrollArea::vertical()
                    .id_salt("callsign_scroll")
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {

                        // Station tracker — active in last 5 minutes
                        if self.settings.station_tracker_enabled {
                            let min_p = self.settings.station_min_pings;
                        let active = self.station_tracker.active(300);
                        let active: Vec<_> = active.into_iter().filter(|s| s.ping_count >= min_p).collect();
                            if !active.is_empty() {
                                ui.label(egui::RichText::new("── Active ──")
                                    .small().color(egui::Color32::from_gray(160)));
                                for st in &active {
                                    // Confidence colour: green = many pings, amber = few
                                    let col = if st.ping_count >= 5 {
                                        egui::Color32::from_rgb(80, 220, 100)
                                    } else if st.ping_count >= 2 {
                                        egui::Color32::from_rgb(220, 180, 60)
                                    } else {
                                        egui::Color32::from_gray(160)
                                    };
                                    // Top callsigns at this DF
                                    let mut top_calls = st.callsigns.clone();
                                    top_calls.sort_by(|a,b| b.1.cmp(&a.1));
                                    let call_str = top_calls.iter().take(2)
                                        .map(|(c,n)| format!("{} ×{}", c, n))
                                        .collect::<Vec<_>>().join("  ");
                                    ui.horizontal(|ui| {
                                        ui.label(egui::RichText::new(
                                            format!("{:+.0}Hz", st.df_bin))
                                            .monospace().small()
                                            .color(egui::Color32::from_gray(140)));
                                        ui.label(egui::RichText::new(
                                            format!("×{}", st.ping_count))
                                            .small().color(col));
                                    });
                                    if !call_str.is_empty() {
                                        ui.label(egui::RichText::new(&call_str)
                                            .monospace().small().color(col));
                                    }
                                    // Best decode fragment
                                    if !st.best_decode.is_empty() {
                                        let preview = &st.best_decode[..st.best_decode.len().min(30)];
                                        ui.label(egui::RichText::new(preview)
                                            .monospace().small()
                                            .color(egui::Color32::from_gray(180)));
                                    }
                                    if let (Some(first), Some(last)) = (st.first_seen, st.last_seen) {
                                        ui.label(egui::RichText::new(
                                            format!("{}-{}", first.format("%H:%M"), last.format("%H:%M")))
                                            .small().color(egui::Color32::from_gray(110)));
                                    }
                                    ui.add_space(2.0);
                                }
                                ui.separator();
                            }
                        }

                        // Callsign list (all session)
                        ui.label(egui::RichText::new("── Session ──")
                            .small().color(egui::Color32::from_gray(160)));
                        let min_p = self.settings.station_min_pings;
                        let mut calls = self.seen_calls.clone();
                        calls.sort_by(|a, b| b.2.cmp(&a.2));
                        for (call, first_seen, count) in calls.iter().filter(|(_,_,n)| *n >= min_p) {
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new(call)
                                    .monospace().small()
                                    .color(egui::Color32::from_rgb(100, 200, 255)));
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    ui.label(egui::RichText::new(format!("×{}", count))
                                        .small().color(egui::Color32::from_gray(150)));
                                    ui.label(egui::RichText::new(
                                        first_seen.format("%H:%M").to_string())
                                        .small().color(egui::Color32::from_gray(120)));
                                });
                            });
                        }
                    });
            });

                egui::CentralPanel::default().show(ctx, |ui| {
            // ── Spectrogram: X=time (0..30s), Y=frequency (0..3kHz) ──────────
            {
                let wf_height = 150.0f32;
                let avail_w   = ui.available_width();
                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(avail_w, wf_height), egui::Sense::hover()
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(0, 0, 20));

                let n_cols = self.wf_columns.len();
                if n_cols > 0 {
                    // Columns per period: period_secs * 11025 / 1024
                    let period_secs = match self.settings.period {
                        Period::TxFirst15 | Period::TxSecond15 => 15u32,
                        _ => 30u32,
                    };
                    let max_cols = (period_secs * 11025 / 1024).max(1) as usize;
                    // col_w fills exactly avail_w over the full period
                    let col_w = avail_w / max_cols as f32;
                    let bin_h = wf_height / DISPLAY_BINS as f32;

                    for (ci, col) in self.wf_columns.iter().enumerate() {
                        let x = rect.left() + ci as f32 * col_w;
                        for (bi, &v) in col.iter().enumerate() {
                            // bi=0 is DC (bottom), bi=DISPLAY_BINS-1 is 3kHz (top)
                            // Invert: low freq at bottom, high freq at top
                            let y = rect.bottom() - (bi as f32 + 1.0) * bin_h;
                            if v > 0.05 {
                                painter.rect_filled(
                                    egui::Rect::from_min_size(
                                        egui::pos2(x, y),
                                        egui::vec2(col_w.max(1.5), bin_h.max(1.0)),
                                    ),
                                    0.0,
                                    heat_color(v),
                                );
                            }
                        }
                    }

                    // FSK441 tone markers — subtle dashed lines, low alpha
                    // Drawn as short segments with gaps to avoid visual clutter
                    let dash_len = 8.0f32;
                    let gap_len  = 12.0f32;
                    let step     = dash_len + gap_len;
                    let tone_col = egui::Color32::from_rgba_unmultiplied(180, 160, 60, 55);
                    for &hz in &FSK441_TONES_HZ {
                        let bin = hz_to_bin(hz);
                        let y = rect.bottom() - (bin as f32 + 0.5) * bin_h;
                        let mut x = rect.left();
                        while x < rect.right() {
                            let x_end = (x + dash_len).min(rect.right());
                            painter.line_segment(
                                [egui::pos2(x, y), egui::pos2(x_end, y)],
                                egui::Stroke::new(0.8, tone_col),
                            );
                            x += step;
                        }
                    }

                    // Frequency axis labels (right edge, Y axis)
                    for khz in [0u32, 500, 1000, 1500, 2000, 2500, 3000] {
                        let bin = hz_to_bin(khz as f32);
                        if bin >= DISPLAY_BINS { continue; }
                        let y = rect.bottom() - bin as f32 * bin_h;
                        painter.text(
                            egui::pos2(rect.right() - 2.0, y),
                            egui::Align2::RIGHT_CENTER,
                            format!("{}k", khz / 1000),
                            egui::FontId::monospace(8.0),
                            egui::Color32::from_rgba_unmultiplied(180, 180, 180, 80),
                        );
                    }

                    // Amplitude envelope trace — green line, dB scale
                    // 40 dB dynamic range: noise floor at bottom, signals above
                    if self.wf_amplitude.len() > 1 {
                        let trace_h  = wf_height * 0.28;
                        let baseline = rect.bottom();
                        let db_range = 40.0f32; // dB shown (noise floor to top)

                        // Estimate noise floor as 20th percentile of period so far
                        let mut sorted = self.wf_amplitude.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let noise = sorted[sorted.len() / 5].max(1e-7);

                        let mut pts: Vec<egui::Pos2> = Vec::with_capacity(n_cols);
                        for (ci, &amp) in self.wf_amplitude.iter().enumerate() {
                            let x  = rect.left() + ci as f32 * col_w + col_w * 0.5;
                            let db = 20.0 * (amp / noise).log10(); // dB above noise floor
                            let norm = (db / db_range).clamp(0.0, 1.0);
                            let y  = baseline - norm * trace_h;
                            pts.push(egui::pos2(x, y));
                        }

                        for i in 1..pts.len() {
                            painter.line_segment(
                                [pts[i-1], pts[i]],
                                egui::Stroke::new(1.5,
                                    egui::Color32::from_rgba_unmultiplied(0, 220, 80, 210)),
                            );
                        }

                        // dB axis labels on right edge (inside spectrogram)
                        for db in [0i32, 10, 20, 30, 40] {
                            let norm = db as f32 / db_range;
                            let y = baseline - norm * trace_h;
                            painter.text(
                                egui::pos2(rect.right() - 28.0, y),
                                egui::Align2::LEFT_CENTER,
                                format!("{}dB", db),
                                egui::FontId::monospace(8.0),
                                egui::Color32::from_rgba_unmultiplied(0, 160, 40, 100),
                            );
                        }
                    }

                    // Time cursor — white vertical line at current position
                    let x_now = rect.left() + n_cols as f32 * col_w;
                    painter.line_segment(
                        [egui::pos2(x_now, rect.top()), egui::pos2(x_now, rect.bottom())],
                        egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(255,255,255,60)),
                    );
                }

                painter.rect_stroke(rect, 0.0,
                    egui::Stroke::new(1.0, egui::Color32::from_gray(60)));
            }

            // ── DF Scatter Strip: X=time (synced to waterfall), Y=DF ±500Hz ──
            {
                let strip_h  = 40.0f32;
                let df_range = 500.0f32; // ±500 Hz maps to top/bottom
                let avail_w  = ui.available_width();
                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(avail_w, strip_h), egui::Sense::hover()
                );
                let painter = ui.painter_at(rect);
                // Background
                painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(0, 0, 15));

                let period_secs = match self.settings.period {
                    Period::TxFirst15 | Period::TxSecond15 => 15u32,
                    _ => 30u32,
                };
                let max_cols = (period_secs * 11025 / 1024).max(1) as usize;
                let col_w    = avail_w / max_cols as f32;
                let _n_cols  = self.wf_columns.len();

                // Zero line (df=0) — dim white
                let y_zero = rect.top() + strip_h * 0.5;
                painter.line_segment(
                    [egui::pos2(rect.left(), y_zero), egui::pos2(rect.right(), y_zero)],
                    egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(255,255,255,25)),
                );
                // ±200 Hz guide lines — very dim
                for df_guide in [-200.0f32, 200.0] {
                    let y = rect.top() + strip_h * (0.5 - df_guide / (df_range * 2.0));
                    painter.line_segment(
                        [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                        egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(100,100,100,30)),
                    );
                }

                // Draw dots
                for dot in &self.df_dots {
                    // X: align to waterfall column position
                    let x = rect.left() + dot.col_idx as f32 * col_w;
                    if x < rect.left() || x > rect.right() { continue; }
                    // Y: df_hz mapped to ±df_range, clamped
                    let df_clamped = dot.df_hz.clamp(-df_range, df_range);
                    let y = rect.top() + strip_h * (0.5 - df_clamped / (df_range * 2.0));
                    painter.circle_filled(
                        egui::pos2(x, y),
                        2.5,
                        egui::Color32::from_rgba_unmultiplied(dot.r, dot.g, dot.b, 200),
                    );
                }

                // Y-axis labels
                let label_col = egui::Color32::from_gray(80);
                painter.text(egui::pos2(rect.right() - 38.0, rect.top() + 2.0),
                    egui::Align2::LEFT_TOP, "+500Hz", egui::FontId::proportional(9.0), label_col);
                painter.text(egui::pos2(rect.right() - 38.0, rect.bottom() - 11.0),
                    egui::Align2::LEFT_TOP, "-500Hz", egui::FontId::proportional(9.0), label_col);
                painter.text(egui::pos2(rect.right() - 22.0, y_zero - 5.0),
                    egui::Align2::LEFT_TOP, "0", egui::FontId::proportional(9.0), label_col);

                painter.rect_stroke(rect, 0.0,
                    egui::Stroke::new(1.0, egui::Color32::from_gray(40)));
            }

            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Decoded Pings").strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // RAW mode toggle
                    let raw_label = if self.settings.raw_mode {
                        egui::RichText::new("≈MSHV").small()
                            .color(egui::Color32::from_rgb(100, 200, 255))
                            .strong()
                    } else {
                        egui::RichText::new("≈MSHV").small()
                            .color(egui::Color32::from_gray(130))
                    };
                    if ui.add(egui::Button::new(raw_label).small()).clicked() {
                        self.settings.raw_mode = !self.settings.raw_mode;
                        // Send updated settings to engine
                        let _ = self.settings_watch_tx.send(self.settings.clone());
                    }
                    ui.separator();
                    if ui.small_button("Clear").clicked() {
                        self.clear_decodes();
                    }
                });
            });
            ui.separator();
            ui.horizontal(|ui| {
                ui.monospace(format!("{:<10} {:>7} {:>8} {:>5} {:>3}  Decode",
                    "Time", "DF(Hz)", "CCF", "Conf", "Scr"));
            });
            ui.separator();

            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for i in 0..self.decodes.len() {
                        let e   = &self.decodes[i];
                        let sel = self.selected == Some(i);
                        let e_passed = e.passed_threshold;
                        let e_accum  = e.is_accumulated;
                        let settings_raw = self.settings.raw_mode;

                        // Header: time df ccf conf score — always dim grey
                        let header = format!("{:<10} {:>+7.0} {:>8.1} {:>5.2} {:>3}  ",
                            e.timestamp.format("%H:%M:%S"), e.df_hz, e.ccf_ratio,
                            e.confidence, e.score);

                        // Render row with click detection via allocate_rect
                        let row_start = ui.cursor().min;
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 0.0;

                            // Selection highlight
                            if sel {
                                let r = ui.max_rect();
                                ui.painter().rect_filled(r, 0.0,
                                    egui::Color32::from_rgba_unmultiplied(255,255,255,18));
                            }

                            // Header in dim grey monospace
                            ui.label(egui::RichText::new(&header)
                                .monospace()
                                .color(egui::Color32::from_gray(110)));

                            // Render each character coloured by its confidence
                            let chars: Vec<char> = e.raw.trim().chars().collect();
                            let confs = &e.char_confs;
                            let conf_thresh_bold = 0.5f32;
                            let conf_thresh_show = 0.15f32;

                            // RAW(≈MSHV) mode: rows passing MSHV-filter but not our pipeline
                            // shown in plain yellow — "WSJT shows this, we add colour coding"
                            // Rows passing both = normal colour coding applies
                            if settings_raw && !e_passed && !e_accum {
                                // MSHV shows this — plain white/yellow like WSJT output
                                for ch in chars.iter() {
                                    ui.label(egui::RichText::new(ch.to_string())
                                        .monospace()
                                        .color(egui::Color32::from_rgb(200, 200, 160)));
                                }
                            } else {

                            // Check if any callsign in this row is tracker-confirmed
                            // (seen ≥ min_pings times at same DF in the station tracker)
                            let df_bin = (e.df_hz / 43.0).round() as i32 * 43;
                            let tracker_confirmed = self.settings.station_tracker_enabled && {
                                let min_p = self.settings.station_min_pings;
                                self.station_tracker.stations.iter().any(|st| {
                                    st.df_bin == df_bin
                                    && st.ping_count >= min_p.max(2) // at least 2 to confirm
                                    && e.callsigns.iter().any(|c|
                                        st.callsigns.iter().any(|(sc,_)| sc == c))
                                })
                            };

                            for (ci, ch) in chars.iter().enumerate() {
                                let conf = confs.get(ci).copied().unwrap_or(0.0);
                                // When tracker-confirmed, entire row is cyan at varying brightness
                                // High conf = bright cyan, low conf = dim cyan, very low = darkest cyan
                                // This makes the whole row readable as a unit
                                let (r, g, b, bold) = if e.is_accumulated {
                                    if conf >= conf_thresh_bold { (255u8, 200u8,  50u8, true) }
                                    else if conf >= conf_thresh_show { (180u8, 140u8, 40u8, false) }
                                    else { (100u8, 80u8, 30u8, false) }
                                } else if tracker_confirmed {
                                    // Full row in cyan — brightness tracks confidence
                                    if conf >= conf_thresh_bold      { (80u8,  210u8, 255u8, true)  }
                                    else if conf >= conf_thresh_show  { (50u8,  150u8, 200u8, false) }
                                    else                              { (30u8,  100u8, 140u8, false) }
                                } else if conf >= conf_thresh_bold {
                                    if e.score >= 80 { (80u8,  255u8,  80u8, true) }
                                    else             { (200u8, 200u8,  60u8, true) }
                                } else if conf >= conf_thresh_show {
                                    (140u8, 140u8, 140u8, false)
                                } else {
                                    (70u8, 70u8, 70u8, false)
                                };
                                let rt = egui::RichText::new(ch.to_string())
                                    .monospace()
                                    .color(egui::Color32::from_rgb(r, g, b));
                                let rt = if bold { rt.strong() } else { rt };
                                ui.label(rt);
                            }
                            } // end RAW else
                        });

                        let in_qso = self.settings.their_call_hint.is_some();
                        // Invisible clickable overlay across the full row
                        let row_rect = egui::Rect::from_min_max(
                            row_start,
                            egui::pos2(ui.max_rect().right(), ui.cursor().min.y),
                        );
                        let resp = ui.allocate_rect(row_rect, egui::Sense::click());

                        // Handle row click — second click deselects, click elsewhere deselects
                        if (resp.clicked() || resp.secondary_clicked()) && !in_qso {
                            if self.selected == Some(i) {
                                self.selected = None;
                                continue;
                            }
                            self.selected = Some(i);
                            let my = self.settings.my_call.to_uppercase();
                            let foreign: Vec<&String> = e.callsigns.iter()
                                .filter(|c| !c.eq_ignore_ascii_case(&my))
                                .collect();
                            // Only auto-populate if exactly one foreign call — otherwise
                            // user must click the individual callsign button below
                            if foreign.len() == 1 {
                                self.their_call_edit = foreign[0].clone();
                                self.tx_msgs_key.clear();
                            } else if foreign.is_empty() {
                                if let Some(c) = e.callsigns.first() {
                                    self.their_call_edit = c.clone();
                                    self.tx_msgs_key.clear();
                                }
                            }
                            if let Some(loc) = &e.locator { self.their_loc_edit = loc.clone(); }
                            if let Some(rpt) = &e.report  { self.report_rcvd = rpt.clone(); }
                        }

                        // If multiple callsigns on this row, show small clickable call buttons
                        // These appear when the row is selected, overlaid in a popup-style strip
                        if sel && !in_qso {
                            let my = self.settings.my_call.to_uppercase();
                            let foreign: Vec<String> = e.callsigns.iter()
                                .filter(|c| !c.eq_ignore_ascii_case(&my))
                                .cloned()
                                .collect();
                            if foreign.len() > 1 {
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("  → ").small()
                                        .color(egui::Color32::YELLOW));
                                    for call in &foreign {
                                        if ui.add(egui::Button::new(
                                            egui::RichText::new(call).small().monospace()
                                                .color(egui::Color32::from_rgb(100, 200, 255))
                                        ).small().frame(true)).clicked() {
                                            self.their_call_edit = call.clone();
                                            self.tx_msgs_key.clear();
                                            if let Some(loc) = &e.locator {
                                                self.their_loc_edit = loc.clone();
                                            }
                                        }
                                    }
                                });
                            }
                        }

                        if resp.double_clicked() && e.is_cq && !self.their_call_edit.is_empty() {
                            let mc = self.settings.my_call.clone();
                            let ml = self.settings.my_loc.clone();
                            let tc = self.their_call_edit.trim().to_uppercase();
                            let tl = self.their_loc_edit.trim().to_uppercase();
                            let rs = self.report_sent.clone();
                            self.qso = QsoState::answer_cq(mc, ml, tc, tl, rs);
                            let msg = self.qso.tx_message().unwrap_or_default();
                            self.qso_log.push(format!("Answer CQ: {}", msg));
                            self.send_tx(&msg);
                        }

                        // Highlight selected row
                        if sel {
                            ui.painter().rect_filled(
                                ui.min_rect(),
                                0.0,
                                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 15),
                            );
                        }
                    }
                });
        });

        // ── Settings window (matches MSK2K layout) ────────────────────────
        if self.settings_open {
            egui::Window::new("⚙ Settings")
                .collapsible(false).resizable(true).default_width(450.0)
                .show(ctx, |ui| {

                    // ── Station Setup ─────────────────────────────────────
                    ui.heading("Station Setup");
                    ui.horizontal(|ui| {
                        ui.label("My Callsign:");
                        if ui.text_edit_singleline(&mut self.settings.my_call).changed() {
                            self.settings.my_call = self.settings.my_call.to_uppercase();
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("My Locator: ");
                        if ui.text_edit_singleline(&mut self.settings.my_loc).changed() {
                            self.settings.my_loc = self.settings.my_loc.to_uppercase();
                        }
                    });

                    // ── Rig Control ───────────────────────────────────────
                    ui.separator();
                    ui.heading("Rig Control (Hamlib)");
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.settings.hamlib_enabled, "Enable CAT Control");
                    });

                    if self.settings.hamlib_enabled {
                        ui.indent("cat", |ui| {
                            egui::Grid::new("rig_grid")
                                .num_columns(2).spacing([10.0, 8.0])
                                .show(ui, |ui| {

                                ui.label("Rig Selection:");
                                ui.vertical(|ui| {
                                    ui.text_edit_singleline(&mut self.rig_search)
                                        .on_hover_text("Type model number to filter (e.g. 9700)");
                                    let cur = if self.settings.rig_model.is_empty() {
                                        "Select Rig...".to_string()
                                    } else {
                                        self.rig_list.iter()
                                            .find(|(id, _)| id == &self.settings.rig_model)
                                            .map(|(_, n)| n.clone())
                                            .unwrap_or_else(|| format!("ID: {}", self.settings.rig_model))
                                    };
                                    egui::ComboBox::from_id_salt("rig_sel")
                                        .selected_text(cur).width(250.0)
                                        .show_ui(ui, |ui| {
                                            let srch = self.rig_search.to_uppercase();
                                            for (id, name) in &self.rig_list.clone() {
                                                if srch.is_empty()
                                                    || name.to_uppercase().contains(&srch)
                                                    || id.contains(&srch)
                                                {
                                                    ui.selectable_value(
                                                        &mut self.settings.rig_model,
                                                        id.clone(), name);
                                                }
                                            }
                                        });
                                });
                                ui.end_row();

                                ui.label("Serial Port:");
                                ui.horizontal(|ui| {
                                    let port_disp = if self.settings.rig_port.is_empty() {
                                        "Select Port...".to_string()
                                    } else {
                                        self.settings.rig_port.clone()
                                    };
                                    egui::ComboBox::from_id_salt("ser_port")
                                        .selected_text(&port_disp).width(280.0)
                                        .show_ui(ui, |ui| {
                                            for p in &self.serial_ports.clone() {
                                                ui.selectable_value(
                                                    &mut self.settings.rig_port, p.clone(), p);
                                            }
                                        });
                                    if ui.button("🔄").clicked() {
                                        self.serial_ports = enumerate_serial_ports();
                                    }
                                });
                                ui.end_row();

                                ui.label("Baud Rate:");
                                ui.text_edit_singleline(&mut self.settings.rig_baud);
                                ui.end_row();
                            });
                        });
                        ui.label(egui::RichText::new(
                            "Start rigctld before launching FSK441, e.g:\n  rigctld -m 3081 -r /dev/cu.usbserial-XXX -s 19200")
                            .small().color(egui::Color32::GRAY));
                    }

                    // ── Audio Hardware ────────────────────────────────────
                    ui.separator();
                    ui.heading("Audio Hardware");
                    ui.horizontal(|ui| {
                        if ui.button("🔄 Refresh Devices")
                            .on_hover_text("Re-scan audio hardware. Use after swapping USB radios.")
                            .clicked()
                        {
                            self.refresh_audio_devices();
                        }
                        if self.settings.sel_in.is_none() || self.settings.sel_out.is_none() {
                            ui.label(egui::RichText::new("⚠ No device selected")
                                .small().color(egui::Color32::from_rgb(255, 180, 0)));
                        }
                    });

                    ui.add_space(4.0);
                    ui.label("Input Device:");
                    egui::ComboBox::from_id_salt("in_dev")
                        .selected_text(self.settings.sel_in.clone()
                            .unwrap_or_else(|| "Default".into()))
                        .width(300.0)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.settings.sel_in, None, "Default");
                            for d in &self.in_devs.clone() {
                                ui.selectable_value(
                                    &mut self.settings.sel_in, Some(d.clone()), d);
                            }
                        });

                    ui.add_space(8.0);
                    ui.label("Output Device:");
                    egui::ComboBox::from_id_salt("out_dev")
                        .selected_text(self.settings.sel_out.clone()
                            .unwrap_or_else(|| "Default".into()))
                        .width(300.0)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.settings.sel_out, None, "Default");
                            for d in &self.out_devs.clone() {
                                ui.selectable_value(
                                    &mut self.settings.sel_out, Some(d.clone()), d);
                            }
                        });

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.label("TX Level:");
                        ui.add(egui::Slider::new(&mut self.settings.tx_level, 0.0..=1.0)
                            .custom_formatter(|v, _| format!("{}%", (v * 100.0).round() as i32)));
                    });

                    // ── TX/RX Period ──────────────────────────────────────
                    ui.separator();
                    ui.heading("TX/RX Period");
                    ui.horizontal(|ui| {
                        ui.label("Period:");
                        egui::ComboBox::from_id_salt("period")
                            .width(180.0)
                            .selected_text(match self.settings.period {
                                Period::TxFirst    => "TX 1st 30s",
                                Period::TxSecond   => "TX 2nd 30s",
                                Period::TxFirst15  => "TX 1st 15s",
                                Period::TxSecond15 => "TX 2nd 15s",
                            })
                            .show_ui(ui, |ui| {
                                for (p, label) in [
                                    (Period::TxFirst,    "TX 1st 30s"),
                                    (Period::TxSecond,   "TX 2nd 30s"),
                                    (Period::TxFirst15,  "TX 1st 15s"),
                                    (Period::TxSecond15, "TX 2nd 15s"),
                                ] {
                                    ui.selectable_value(&mut self.settings.period, p, label);
                                }
                            });
                    });

                    // ── Filtering ─────────────────────────────────────────
                    ui.separator();
                    ui.heading("Filtering");
                    egui::Grid::new("filt").num_columns(2).spacing([10.0, 6.0]).show(ui, |ui| {
                        ui.label("cty.dat path:"); ui.text_edit_singleline(&mut self.settings.cty_path); ui.end_row();
                        ui.label("Max range (km):"); ui.add(egui::Slider::new(&mut self.settings.max_km, 500.0..=5000.0).integer()); ui.end_row();
                        ui.label("Min CCF ratio:"); ui.add(egui::Slider::new(&mut self.settings.min_ccf, 50.0..=2000.0).integer()); ui.end_row();
                        ui.label("Min confidence:"); ui.add(egui::Slider::new(&mut self.settings.min_conf, 0.30..=0.99)); ui.end_row();
                    });

                    ui.separator();
                    if ui.button("Save & Close").clicked() {
                        self.settings_open = false;
                        save_config(&self.settings);
                        // Apply changes to running TxEngine immediately
                        let _ = self.tx_cmd_tx.send(TxCommand::SetPeriod(self.settings.period));
                        self.period_timer = PeriodTimer::new(self.settings.period);
                        let _ = self.tx_cmd_tx.send(TxCommand::SetOutputDevice(self.settings.audio_out()));
                        let _ = self.tx_cmd_tx.send(TxCommand::SetHamlib(self.settings.hamlib_addr()));
                        // Kill existing rigctld if hamlib disabled
                        if !self.settings.hamlib_enabled {
                            std::process::Command::new("pkill").args(["-f", "rigctld"]).spawn().ok();
                            log::info!("[LAUNCHER] Hamlib disabled — killed rigctld");
                        }
                    }
                });
        }
    }
}

// ─── Engine ───────────────────────────────────────────────────────────────────

async fn run_engine(settings: Settings, event_tx: mpsc::UnboundedSender<EngineEvent>, tx_active: std::sync::Arc<std::sync::atomic::AtomicBool>, mut settings_rx: tokio::sync::watch::Receiver<Settings>) {
    let mut settings = settings; // mutable local copy

    let qth = Qth::from_maidenhead(&settings.my_loc).unwrap_or_default();
    let db  = PrefixDb::load(&PathBuf::from(&settings.cty_path));
    let geo = GeoValidator::new(db, qth, settings.max_km);
    // Use ~/.fsk441/fsk441.db — don't exit on failure, just log and continue
    let db_path = {
        let mut p = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        p.push(".fsk441");
        std::fs::create_dir_all(&p).ok();
        p.push("fsk441.db");
        p
    };
    let store_opt = match Store::open(&db_path) {
        Ok(s) => { log::info!("[DB] Opened {:?}", db_path); Some(s) }
        Err(e) => { log::warn!("[DB] Cannot open {:?}: {} — continuing without DB", db_path, e); None }
    };
    let store = store_opt;
    let session_id = store.as_ref().and_then(|s|
        s.new_session(Some(settings.sel_in.as_deref().unwrap_or("default")), None).ok()
    ).unwrap_or(0);

    // Two channels from audio: one for detector, one for spectrum
    let (audio_tx,      audio_rx)      = mpsc::unbounded_channel::<Vec<f32>>();
    let (audio_spec_tx, mut audio_spec_rx) = mpsc::unbounded_channel::<Vec<f32>>();
    let (ping_tx, mut ping_rx) = mpsc::unbounded_channel::<detector::DetectedPing>();

    if let Err(e) = audio::start_live(settings.audio_in(), audio_tx).await {
        log::error!("[ENGINE] Audio: {}", e); return;
    }

    // Fan out raw audio to detector AND spectrum computer
    let spec_tx2 = event_tx.clone();
    #[allow(unused_variables)]
    let tx_active_spec = tx_active.clone();
    tokio::spawn(async move {
        let mut planner = rustfft::FftPlanner::<f32>::new();
        let mut _chunk_count = 0u64;
        while let Some(chunk) = audio_spec_rx.recv().await {
            // Every audio chunk = one column — 1024/11025 = ~93ms per column
            let rms = {
                let sum: f32 = chunk.iter().map(|&s| s * s).sum();
                (sum / chunk.len() as f32).sqrt()
            };
            let bins = compute_column(&chunk, &mut planner);
            let _ = spec_tx2.send(EngineEvent::Spectrum(bins, rms));
        }
    });

    // Fan audio to detector and spectrum — gate both on tx_active
    let (fanout_ping_tx, fanout_ping_rx) = mpsc::unbounded_channel::<Vec<f32>>();
    let tx_active_fanout = tx_active.clone();
    tokio::spawn(async move {
        let mut rx = audio_rx;
        while let Some(chunk) = rx.recv().await {
            if !tx_active_fanout.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = audio_spec_tx.send(chunk.clone());
                let _ = fanout_ping_tx.send(chunk);
            }
        }
    });

    tokio::spawn(run_detector(fanout_ping_rx, ping_tx, settings.threshold, params::DEFAULT_DFTOL));

    let mut frag_acc = FragmentAccumulator::new();
    let mut current_slot_idx: i64 = -1;
    // Brief startup delay — allow audio stream to stabilise before accepting pings
    let engine_start = std::time::Instant::now();
    const STARTUP_MS: u128 = 3000;
    let mut tx_cooldown_until: Option<std::time::Instant> = None;

    while let Some(ping) = ping_rx.recv().await {
        let is_tx = tx_active.load(std::sync::atomic::Ordering::Relaxed);
        // Pull latest settings if changed (their_call_hint, period, etc.)
        if settings_rx.has_changed().unwrap_or(false) {
            let old_hint = settings.their_call_hint.clone();
            settings = settings_rx.borrow_and_update().clone();

            // Sync QSO context constraint with their_call_hint
            match (&old_hint, &settings.their_call_hint) {
                (None, Some(their)) => {
                    // QSO just started — apply constraint
                    frag_acc.set_constraint(&settings.my_call, their);
                }
                (Some(_), None) => {
                    // QSO ended — clear constraint
                    frag_acc.clear_constraint();
                }
                (Some(old), Some(new)) if old != new => {
                    // Their call changed mid-QSO (unlikely but handle it)
                    frag_acc.set_constraint(&settings.my_call, new);
                }
                _ => {}
            }
        }

        // Drain any pings that queued during TX — they are our own audio
        if is_tx {
            frag_acc.prune_older_than(300); // housekeeping during TX
            // Set cooldown so we reject ring-buffer bleed for 2.5s after TX ends
            tx_cooldown_until = Some(std::time::Instant::now()
                + std::time::Duration::from_millis(3500));
            frag_acc.clear();
            while ping_rx.try_recv().is_ok() {}
            continue;
        }

        // Reject pings during cooldown after TX (loopback tail)
        if let Some(until) = tx_cooldown_until {
            if std::time::Instant::now() < until {
                continue;
            } else {
                tx_cooldown_until = None;
            }
        }

        // Detect slot boundary → process accumulated fragments (DON'T clear — persist cross-slot)
        let now_ms    = chrono::Utc::now().timestamp_millis();
        let slot_idx  = now_ms / 30_000;
        if slot_idx != current_slot_idx && current_slot_idx >= 0 {
            let n = frag_acc.fragment_count();
            if n > 0 {
                log::info!("[ACCUM] Slot boundary — processing {} cross-slot fragments", n);
                for acc in frag_acc.process() {
                    let _ = event_tx.send(EngineEvent::Accumulated(acc));
                }
                // Don't clear — keep accumulating cross-slot
                // Prune fragments older than 5 minutes
                frag_acc.prune_older_than(300);
            }
        }
        current_slot_idx = slot_idx;

        // Offload CPU-bound demodulation to blocking thread pool
        let audio   = ping.audio.clone();
        let df_hz   = ping.df_hz;
        let result  = tokio::task::spawn_blocking(move || {
            longx(&audio, df_hz, params::DEFAULT_DFTOL)
        }).await.unwrap();
        let parsed = ParsedMessage::parse_geo(&result.raw_decode, Some(&geo));
        if let Some(ref s) = store { let _ = s.insert_ping(session_id, &ping, &result, &parsed); }

        if result.raw_decode.trim().len() < 2 { continue; }

        // Startup guard — ignore pings for first 3s while audio settles
        if engine_start.elapsed().as_millis() < STARTUP_MS { continue; }

        // Hard guard: own TX leakage has CCF > 50000 — reject regardless of timing
        if ping.ccf_ratio > 50_000.0 {
            log::debug!("[ENGINE] Rejected own-TX leakage ping ccf={:.0}", ping.ccf_ratio);
            continue;
        }

        // ── Threshold computation ────────────────────────────────────────────────
        let in_qso = settings.their_call_hint.is_some();
        let has_callsign = !parsed.callsigns.is_empty();

        // MSHV-equivalent filter: what WSJT/MSHV would show on screen
        // conf >= 0.45, garbage ratio < 40%, at least one word >= 3 printable chars
        let mshv_pass = {
            let conf_ok = result.mean_confidence >= 0.45;
            let raw = result.raw_decode.trim();
            let total_chars = raw.chars().filter(|c| !c.is_whitespace()).count().max(1);
            let garbage_chars = raw.chars().filter(|c| !c.is_alphanumeric() && !c.is_whitespace()).count();
            let garbage_ok = (garbage_chars as f32 / total_chars as f32) < 0.40;
            let has_word = raw.split_whitespace().any(|w| {
                w.len() >= 3 && w.chars().all(|c| c.is_alphanumeric())
            });
            conf_ok && garbage_ok && has_word
        };

        // Our scoring pipeline
        let threshold_pass = if in_qso {
            ping.ccf_ratio >= settings.min_ccf * 0.1 || ping.ccf_ratio >= 10.0
        } else {
            let (eff_min_conf, eff_min_score) = if ping.ccf_ratio > 1000.0 {
                (0.30f32, if result.mean_confidence > 0.85 { 30u8 } else { 42u8 })
            } else if ping.ccf_ratio > 500.0 {
                (settings.min_conf * 0.75, 52u8)
            } else if ping.ccf_ratio > 200.0 {
                if has_callsign && result.mean_confidence > 0.75 { (settings.min_conf * 0.80, 50u8) }
                else if has_callsign { (settings.min_conf * 0.85, 55u8) }
                else { (settings.min_conf * 0.90, 62u8) }
            } else if has_callsign && ping.ccf_ratio > 100.0 {
                (0.55f32, 52u8)
            } else {
                (settings.min_conf * 1.1, 65u8)
            };
            (result.mean_confidence >= eff_min_conf || ping.ccf_ratio >= settings.min_ccf)
                && parsed.validity_score >= eff_min_score
        };

        // RAW mode = MSHV-equivalent: show what WSJT would show
        // Normal mode = our pipeline
        // In both cases threshold_pass drives colour coding
        let show = if settings.raw_mode { mshv_pass } else { threshold_pass || in_qso };
        if !show { continue; }

        // Per-character confidence: max of each char's 48-element prob array
        let char_confs: Vec<f32> = result.char_probs.iter()
            .map(|probs| probs.iter().cloned().fold(0.0f32, f32::max))
            .collect();

        // Add to accumulator (before emitting single-ping decode)
        if let Some(frag) = Fragment::from_result(&result, ping.ccf_ratio) {
            frag_acc.add(frag);
        }

        let _ = event_tx.send(EngineEvent::Decode(DecodeEntry {
            timestamp:  ping.timestamp, df_hz: ping.df_hz, ccf_ratio: ping.ccf_ratio,
            confidence: result.mean_confidence, score: parsed.validity_score,
            raw: result.raw_decode, callsigns: parsed.valid_callsigns,
            locator: parsed.locator, report: parsed.report, is_cq: parsed.is_cq,
            char_confs, is_accumulated: false,
            my_call: settings.my_call.clone(),
            df_bin_hz: None,
            passed_threshold: threshold_pass,
        }));
    }
}


// ─── Config persistence (same pattern as MSK2K) ───────────────────────────────

fn config_path() -> PathBuf {
    let mut p = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    p.push(".fsk441");
    std::fs::create_dir_all(&p).ok();
    p.push("fsk441.cfg");
    p
}

fn save_config(s: &Settings) {
    let period = match s.period {
        Period::TxFirst    => "first",
        Period::TxSecond   => "second",
        Period::TxFirst15  => "first15",
        Period::TxSecond15 => "second15",
    };
    let data = format!(
        "my_call={}\nmy_loc={}\ninput={}\noutput={}\nrig_model={}\nrig_port={}\nrig_baud={}\nhamlib_enabled={}\nperiod={}\ncty_path={}\nmax_km={}\nmin_ccf={}\nmin_conf={}\ntx_level={}\n",
        s.my_call,
        s.my_loc,
        s.sel_in.as_deref().unwrap_or(""),
        s.sel_out.as_deref().unwrap_or(""),
        s.rig_model,
        s.rig_port,
        s.rig_baud,
        s.hamlib_enabled,
        period,
        s.cty_path,
        s.max_km,
        s.min_ccf,
        s.min_conf,
        s.tx_level,
    );
    if let Err(e) = std::fs::write(config_path(), data) {
        log::warn!("Failed to save config: {}", e);
    } else {
        log::info!("Config saved to {:?}", config_path());
    }
}

fn load_config() -> Settings {
    let mut s = Settings::default();
    let path  = config_path();
    let Ok(text) = std::fs::read_to_string(&path) else { return s; };
    for line in text.lines() {
        let Some((k, v)) = line.split_once('=') else { continue; };
        let v = v.trim();
        match k.trim() {
            "my_call"        => s.my_call   = v.to_string(),
            "my_loc"         => s.my_loc    = v.to_string(),
            "input"          => s.sel_in    = if v.is_empty() { None } else { Some(v.to_string()) },
            "output"         => s.sel_out   = if v.is_empty() { None } else { Some(v.to_string()) },
            "rig_model"      => s.rig_model = v.to_string(),
            "rig_port"       => s.rig_port  = v.to_string(),
            "rig_baud"       => s.rig_baud  = v.to_string(),
            "hamlib_enabled" => s.hamlib_enabled = v == "true",
            "period"         => s.period    = match v {
                "first"    => Period::TxFirst,
                "first15"  => Period::TxFirst15,
                "second15" => Period::TxSecond15,
                _          => Period::TxSecond,
            },
            "cty_path"       => s.cty_path  = v.to_string(),
            "max_km"         => s.max_km    = v.parse().unwrap_or(3000.0),
            "min_ccf"        => s.min_ccf   = v.parse().unwrap_or(300.0),
            "min_conf"       => s.min_conf  = v.parse().unwrap_or(0.70),
            "tx_level"       => s.tx_level  = v.parse().unwrap_or(0.8),
            _ => {}
        }
    }
    log::info!("Config loaded: call={} loc={} in={:?} out={:?}", s.my_call, s.my_loc, s.sel_in, s.sel_out);
    s
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();
    eframe::run_native("FSK441 Transceiver",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1100.0, 700.0])
                .with_min_inner_size([800.0, 520.0]),
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(Fsk441App::new(cc)))))
}


impl Drop for Fsk441App {
    fn drop(&mut self) {
        if let Some(ref mut child) = self._rigctld {
            log::info!("[LAUNCHER] Killing rigctld (pid={})", child.0.id());
            let _ = child.0.kill();
        }
    }
}
