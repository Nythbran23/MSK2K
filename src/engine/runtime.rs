// src/engine/runtime.rs

use anyhow::Result;
use std::fs;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio::time::sleep;
use crate::engine::hamlib::{HamlibClient, HamlibUpdate};
use std::process::{Command, Child};
use std::path::PathBuf;

use msk2k_dsp::callsign::CallsignCodec;
// 🟢 Import Rendered enum
use crate::proto::{Payload, Rendered};

use crate::engine::bus::{EngineHandle, SlotParity, SlotPeriod, UiCmd, UiEvent};
use crate::modem::{run_transmitter_task, RxAudioCfg, RxConfigUpdate, RxDecoded, TxRequest};
use crate::proto::{self, render_payload, Format, RxEnvelope};
use crate::qso::{Action, EngineEvent, Intent, QsoEngine, QsoState};

pub fn start() -> EngineHandle {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let (cmds, cmd_rx) = mpsc::unbounded_channel::<UiCmd>();
    let (evt_tx, events) = mpsc::unbounded_channel::<UiEvent>();

    rt.spawn(async move {
        if let Err(e) = run_runtime(cmd_rx, evt_tx).await {
            log::error!("engine runtime crashed: {e}");
        }
    });

    EngineHandle {
        cmds,
        events,
        _rt: rt,
    }
}

fn utc_ms_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

fn slot_len_ms(period: SlotPeriod) -> i64 {
    match period {
        SlotPeriod::S15 => 15_000,
        SlotPeriod::S30 => 30_000,
    }
}

fn slot_index(utc_ms: i64, slot_len_ms: i64) -> i64 {
    if slot_len_ms <= 0 {
        return 0;
    }
    utc_ms / slot_len_ms
}

fn slot_parity(utc_ms: i64, slot_len_ms: i64) -> u8 {
    (slot_index(utc_ms, slot_len_ms) % 2) as u8
}

// ─── CONFIG PERSISTENCE ───
#[derive(Default, Clone)]
struct AppConfig {
    my_call: String,
    input_device: Option<String>,
    output_device: Option<String>,
    rig_model: String,
    rig_port: String,
    rig_baud: String,
}
fn config_file_path() -> std::path::PathBuf {
    let mut path = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    path.push(".msk2k");
    std::fs::create_dir_all(&path).ok();
    path.push("msk2k.cfg");
    path
}

fn load_config() -> AppConfig {
    let mut cfg = AppConfig::default();
    let cfg_path = config_file_path();
    if let Ok(contents) = fs::read_to_string(&cfg_path) {
        for line in contents.lines() {
            if let Some((k, v)) = line.split_once('=') {
                match k.trim() {
                    "my_call" => cfg.my_call = v.trim().to_string(),
                    "input" => {
                        let val = v.trim();
                        if !val.is_empty() {
                            cfg.input_device = Some(val.to_string());
                        }
                    }
                    "output" => {
                        let val = v.trim();
                        if !val.is_empty() {
                            cfg.output_device = Some(val.to_string());
                        }
                    }
                    "rig_model" => cfg.rig_model = v.trim().to_string(),
                    "rig_port" => cfg.rig_port = v.trim().to_string(),
                    "rig_baud" => cfg.rig_baud = v.trim().to_string(),
                    _ => {}
                }
            }
        }
        log::info!(
            "📂 Loaded config: Call={}, In={:?}, Out={:?}, Rig={} Port={} Baud={}",
            cfg.my_call,
            cfg.input_device,
            cfg.output_device,
            cfg.rig_model,
            cfg.rig_port,
            cfg.rig_baud
        );
    }
    cfg
}

fn save_config(cfg: &AppConfig) {
    let data = format!(
        "my_call={}\ninput={}\noutput={}\nrig_model={}\nrig_port={}\nrig_baud={}\n",
        cfg.my_call,
        cfg.input_device.as_deref().unwrap_or(""),
        cfg.output_device.as_deref().unwrap_or(""),
        cfg.rig_model,
        cfg.rig_port,
        cfg.rig_baud
    );
    if let Err(e) = fs::write(config_file_path(), data) {
        log::warn!("Failed to save config: {}", e);
    }
}
fn get_rigctld_path() -> String {
    let binary_name = if cfg!(target_os = "windows") { "rigctld.exe" } else { "rigctld" };
    if let Ok(mut path) = std::env::current_exe() {
        path.pop();
        let local_path = path.join(binary_name); 
        if local_path.exists() {
            log::info!("[LAUNCHER] Found bundled rigctld at: {:?}", local_path);
            return local_path.to_string_lossy().to_string();
        }
        let tools_path = path.join("tools").join(binary_name);
        if tools_path.exists() {
            log::info!("[LAUNCHER] Found bundled rigctld at: {:?}", tools_path);
            return tools_path.to_string_lossy().to_string();
        }
    }
    log::info!("[LAUNCHER] Using system rigctld (not bundled)");
    binary_name.to_string()
}
async fn run_runtime(
    mut cmd_rx: mpsc::UnboundedReceiver<UiCmd>,
    evt_tx: mpsc::UnboundedSender<UiEvent>,
) -> Result<()> {
    let (tx_req_tx, tx_req_rx) = mpsc::unbounded_channel::<TxRequest>();

    tokio::spawn(async move {
        if let Err(e) = run_transmitter_task(tx_req_rx).await {
            log::error!("TX worker crashed: {e}");
        }
    });

    let (rx_decoded_tx, mut rx_decoded_rx) = mpsc::unbounded_channel::<RxDecoded>();
    let mut rx_stop_tx: Option<mpsc::UnboundedSender<()>> = None;
    let mut rx_config_tx: Option<mpsc::UnboundedSender<RxConfigUpdate>> = None;

    // LOAD CONFIG
    let saved_cfg = load_config();

    // 🟢 CRITICAL: Tell the UI what we loaded so it fills the boxes
    let _ = evt_tx.send(UiEvent::ConfigLoaded {
        my_call: saved_cfg.my_call.clone(),
        input_device: saved_cfg.input_device.clone(),
        output_device: saved_cfg.output_device.clone(),
        rig_model: saved_cfg.rig_model.clone(),
        rig_port: saved_cfg.rig_port.clone(),
        rig_baud: saved_cfg.rig_baud.clone(),
    });

    let mut input_device: Option<String> = saved_cfg.input_device.clone();
    let mut output_device: Option<String> = saved_cfg.output_device.clone();
    let mut current_rig_model: String = saved_cfg.rig_model.clone();
    let mut current_rig_port: String = saved_cfg.rig_port.clone();
    let mut current_rig_baud: String = saved_cfg.rig_baud.clone();
    let sample_rate: u32 = 48_000;
    let buffer_size: usize = 1024;
    let mut output_level: f32 = 0.4;
    let decode_window_secs: f32 = 4.0;

    let mut qso_engine = QsoEngine::new(saved_cfg.my_call.clone());
    let mut auto_qso: bool = true;

    let mut is_grid_mode = false;
    let mut current_grid_indices: Option<[usize; 4]> = None;

    let mut slot_period = SlotPeriod::S15;
    let mut slot_parity_cfg = SlotParity::Odd;
    let mut running = false;
    let mut tx_active = false;
    let mut last_slot_index: Option<i64> = None;

    let mut observed_remote_slot: Option<u8> = None;
    let mut was_calling_cq: bool = false;
    let mut rx_needs_restart: bool = true;
    let mut diag_tick: u64 = 0; // 🔍 DIAGNOSTIC: heartbeat counter

    let (ham_update_tx, mut ham_update_rx) = mpsc::unbounded_channel();
    let mut hamlib: Option<HamlibClient> = None;

    // 🟢 2. Initialize Launcher Process Holder
    let mut rigctld_process: Option<Child> = None;
    let mut launched_rig_model: String = String::new();
    let mut launched_serial_port: String = String::new();
    let mut launched_baud_rate: u32 = 0;


    // 🟢 AUTO-START: If valid config exists, start LISTENING immediately.
    if !qso_engine.my_call.is_empty() && input_device.is_some() {
        log::info!("⚡ Auto-starting RX with saved configuration...");

        let _ = tx_req_tx.send(TxRequest::ApplyAudio {
            output_device: output_device.clone(),
            output_level,
            sample_rate,
            buffer_size,
            my_call: qso_engine.my_call.clone(),
            their_call: qso_engine.their_call.clone().unwrap_or_default(),
        });

        let (_, events) = qso_engine.on_intent(Intent::Listen);
        process_qso_events(&events, &evt_tx, &rx_config_tx);

        running = true;
        rx_needs_restart = true; // Forces restart_rx in the first loop tick

        let _ = evt_tx.send(UiEvent::Info(format!("Auto-started: {}", qso_engine.my_call)));
    } else {
        log::info!("🚀 Engine started (Waiting for setup)");
        let _ = evt_tx.send(UiEvent::Info(
            "Enter callsign and press LISTEN".into(),
        ));
    }

    // 🟢 AUTO-START HAMLIB: If saved config has rig model + port, launch rigctld once at startup
    if !current_rig_model.is_empty() && !current_rig_port.is_empty() {
        let baud: u32 = current_rig_baud.parse().unwrap_or(19200);
        log::info!("[LAUNCHER] Auto-starting rigctld from saved config: model={} port={} baud={}", 
            current_rig_model, current_rig_port, baud);
        
        // Linux: suppress DTR/RTS before opening serial port
        #[cfg(target_os = "linux")]
        {
            let _ = Command::new("stty")
                .args(&["-F", &current_rig_port, "-hupcl", "-crtscts", &baud.to_string()])
                .output();
        }
        
        let rig_cmd = get_rigctld_path();
        let mut cmd = Command::new(rig_cmd);
        cmd.args(&["-m", &current_rig_model, "-r", &current_rig_port, "-s", &baud.to_string()]);
        
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000);
        }
        
        match cmd.spawn() {
            Ok(c) => {
                log::info!("[LAUNCHER] rigctld auto-started (pid={})", c.id());
                launched_rig_model = current_rig_model.clone();
                launched_serial_port = current_rig_port.clone();
                launched_baud_rate = baud;
                rigctld_process = Some(c);
                // Give rigctld time to bind port 4532
                std::thread::sleep(std::time::Duration::from_millis(800));
                // Now connect Hamlib TCP client
                hamlib = Some(HamlibClient::new("127.0.0.1:4532".to_string(), ham_update_tx.clone()));
                log::info!("[LAUNCHER] Hamlib TCP client connected");
            }
            Err(e) => log::error!("[LAUNCHER] Auto-start failed: {}", e),
        }
    }
    
    loop {
        tokio::select! {
            _ = sleep(Duration::from_millis(50)) => {
                if !running { continue; }

                diag_tick += 1;
                // 🔍 DIAGNOSTIC: Log heartbeat every ~10 seconds
                if diag_tick % 200 == 0 {
                    log::info!("💓 HEARTBEAT: state={}, running={}, rx_config_tx={}, rx_stop_tx={}, their_call={:?}, observed_slot={:?}",
                        qso_engine.state, running,
                        rx_config_tx.is_some(), rx_stop_tx.is_some(),
                        qso_engine.their_call, observed_remote_slot);
                }

                // 🟢 Poll rig frequency every ~5 seconds
                if diag_tick % 100 == 0 {
                    if let Some(h) = &hamlib { h.refresh(); }
                }

                let now_ms = utc_ms_now();
                let slen = slot_len_ms(slot_period);
                let sidx = slot_index(now_ms, slen);
                let slot = slot_parity(now_ms, slen);

                let base_tx_slot: u8 = match slot_parity_cfg {
                    SlotParity::Odd => 1,
                    SlotParity::Even => 0,
                };

                let my_tx_slot: u8 = if let Some(remote) = observed_remote_slot { 1 - remote } else { base_tx_slot };

                let is_tx_state = matches!(
                    qso_engine.state,
                    QsoState::CallingCq | QsoState::CallingStn | QsoState::SendingReport |
                    QsoState::SendingRReport | QsoState::SendingRr | QsoState::Sending73
                );

                let should_tx = is_tx_state && (slot == my_tx_slot);

                if rx_needs_restart && !qso_engine.my_call.is_empty() {
                    restart_rx(&rx_decoded_tx, &mut rx_stop_tx, &mut rx_config_tx, RxAudioCfg {
                        input_device: input_device.clone(),
                        sample_rate,
                        buffer_size,
                        slot_len_ms: slen as u32,
                        my_call: qso_engine.my_call.clone(),
                        their_call: qso_engine.their_call.clone(),
                        decode_window_secs,
                        my_tx_slot,
                        rx_slot_override: observed_remote_slot,
                        listen_all_slots: true,
                    });
                    rx_needs_restart = false;
                }

                if last_slot_index.is_none() || sidx != last_slot_index.unwrap() {
                    last_slot_index = Some(sidx);

                    if let Some(tx) = &rx_config_tx {
                        let _ = tx.send(RxConfigUpdate::EndOfPeriod);
                    }

                    if should_tx {
                        if let Some(h) = &hamlib { h.set_ptt(true); }
                        if !tx_active { tx_active = true; let _ = evt_tx.send(UiEvent::TxActive(true)); }

                        if let Some(payload) = qso_engine.next_tx() {
                            let mut final_payload = payload;

                            if was_calling_cq && is_grid_mode {
                                if let Some(indices) = current_grid_indices {
                                    let codec = CallsignCodec::new();
                                    if let Ok(bits) = codec.encode_cq_with_grid(&qso_engine.my_call, &indices) {
                                        log::info!("[TX] 📡 PACKING GRID CQ (Type 11) - Bits 54,55: [{}, {}]", bits[54], bits[55]);
                                        final_payload = Payload::CqWithGrid { 
                                            from: qso_engine.my_call.clone(), 
                                            grid_bits: bits
                                        };
                                    }
                                }
                            }

                            // 🟢 THE ARCHITECTURAL SHIFT
                            let rendered = render_payload(&final_payload);
                            match rendered {
                                Rendered::Bits(bits) => {
                                    // Bypasses string-to-base37 encoder
                                    let _ = tx_req_tx.send(TxRequest::RawBits {
                                        bits,
                                        slot_len_ms: slen as u32,
                                        my_call: qso_engine.my_call.clone(),
                                        their_call: qso_engine.their_call.clone().unwrap_or_default(),
                                    });
                                    let _ = evt_tx.send(UiEvent::TxText { text: format!("CQ de {} [GRID]", qso_engine.my_call) });
                                }
                                Rendered::Text(raw) => {
                                    // Standard Base-37 path
                                    let _ = tx_req_tx.send(TxRequest::Text {
                                        rendered: raw.clone(),
                                        slot_len_ms: slen as u32,
                                        my_call: qso_engine.my_call.clone(),
                                        their_call: qso_engine.their_call.clone().unwrap_or_default(),
                                    });
                                    let _ = evt_tx.send(UiEvent::TxText { text: raw });
                                }
                            }

                            // 🟢 Check if QSO complete after transmission
                            // Note: next_tx() already increments tx_repeat_count, don't do it again!
                            if qso_engine.state == QsoState::Sending73 {
                                let max_73_repeats = 5;

                                if qso_engine.tx_repeat_count >= max_73_repeats {
                                    let their = qso_engine.their_call.clone().unwrap_or_default();
                                    if let Some(rec) = qso_engine.make_qso_record() {
                                        let _ = evt_tx.send(UiEvent::QsoLogged { record: rec });
                                    }
                                    let _ = evt_tx.send(UiEvent::Info(format!("✓ QSO with {} complete", their)));
                                    let _ = evt_tx.send(UiEvent::TheirCallChanged { 
                                        callsign: String::new(),
                                        grid: None, // 🟢 NEW
                                    });

                                    if was_calling_cq {
                                        let (_, ev2) = qso_engine.on_intent(Intent::Cq);
                                        process_qso_events(&ev2, &evt_tx, &rx_config_tx);
                                        
                                        // 🟢 CRITICAL FIX: Must keep running=true!
                                        // running is already true, just reset slot tracking
                                        last_slot_index = None;
                                        observed_remote_slot = None;
                                    } else {
                                        let (_, ev2) = qso_engine.on_intent(Intent::Listen);
                                        process_qso_events(&ev2, &evt_tx, &rx_config_tx);
                                        
                                        // Keep running for listen mode too
                                        observed_remote_slot = None;
                                    }
                                    update_rx_config(&rx_config_tx, RxConfigUpdate::TheirCall(None));
                                }
                            }
                        }
                    }
                        else {
                        // Only release PTT once when transitioning from TX to RX
                        if tx_active {
                            if let Some(h) = &hamlib { h.set_ptt(false); }
                            tx_active = false;
                            let _ = evt_tx.send(UiEvent::TxActive(false));
                        }
                    }    
                }
            }

            Some(cmd) = cmd_rx.recv() => {
                match cmd {
                    UiCmd::SetInputDevice(dev) => { input_device = dev; }
                    UiCmd::SetOutputDevice(dev) => { output_device = dev; }
                    UiCmd::SetSlotPeriod(p) => { slot_period = p; last_slot_index = None; }
                    UiCmd::SetSlotParity(p) => { slot_parity_cfg = p; }
                    UiCmd::SetAutoQso(on) => { auto_qso = on; }
                    UiCmd::SetTxLevel(level) => { 
                        output_level = level.clamp(0.0, 1.0);
                        log::info!("[ENGINE] TX level set to {:.0}%", output_level * 100.0);
                    }
                    
                    UiCmd::ConfigureHamlib { enabled, address } => {
                        if enabled {
                            hamlib = Some(HamlibClient::new(address, ham_update_tx.clone()));
                        } else {
                            hamlib = None;
                        }
                    }

                    // 2. Configure Launcher
                    UiCmd::ConfigureLauncher { enable_launcher, rig_model, serial_port, baud_rate } => {
                        // Track rig settings for config save
                        current_rig_model = rig_model.clone();
                        current_rig_port = serial_port.clone();
                        current_rig_baud = baud_rate.to_string();

                        // Check if same config is already running (skip re-launch)
                        let mut skip_launch = false;
                        if enable_launcher && !serial_port.is_empty()
                            && launched_rig_model == rig_model
                            && launched_serial_port == serial_port
                            && launched_baud_rate == baud_rate
                        {
                            if let Some(ref mut child) = rigctld_process {
                                match child.try_wait() {
                                    Ok(Some(_status)) => {
                                        log::warn!("[LAUNCHER] rigctld had exited — will re-launch");
                                        rigctld_process = None;
                                    }
                                    Ok(None) => {
                                        log::info!("[LAUNCHER] rigctld already running with same config — skipping");
                                        skip_launch = true;
                                    }
                                    Err(e) => {
                                        log::warn!("[LAUNCHER] Failed to check rigctld status: {} — will re-launch", e);
                                        rigctld_process = None;
                                    }
                                }
                            }
                        }

                        if !skip_launch {
                            // Kill old process first
                            if let Some(mut child) = rigctld_process.take() {
                                log::info!("[LAUNCHER] Killing old rigctld (pid={})", child.id());
                                let _ = child.kill();
                                let _ = child.wait();
                                std::thread::sleep(std::time::Duration::from_millis(500));
                            }
                            launched_rig_model.clear();
                            launched_serial_port.clear();
                            launched_baud_rate = 0;

                            if enable_launcher && !serial_port.is_empty() {
                                // 🟢 LINUX FIX: Suppress DTR/RTS before rigctld opens the serial port
                                #[cfg(target_os = "linux")]
                                {
                                    log::info!("[LAUNCHER] Pre-configuring {} to suppress DTR/RTS", serial_port);
                                    let stty_result = Command::new("stty")
                                        .args(&["-F", &serial_port, "-hupcl", "-crtscts", &baud_rate.to_string()])
                                        .output();
                                    match stty_result {
                                        Ok(out) => {
                                            if !out.status.success() {
                                                log::warn!("[LAUNCHER] stty failed: {}", String::from_utf8_lossy(&out.stderr));
                                            }
                                        }
                                        Err(e) => log::warn!("[LAUNCHER] Could not run stty: {}", e),
                                    }
                                }

                                log::info!("[LAUNCHER] Starting rigctld: Model={} Port={} Baud={}", rig_model, serial_port, baud_rate);
                                
                                let rig_cmd = get_rigctld_path();
                                let mut cmd = Command::new(rig_cmd);
                                cmd.args(&["-m", &rig_model, "-r", &serial_port, "-s", &baud_rate.to_string()]);
                                
                                #[cfg(target_os = "windows")]
                                {
                                    use std::os::windows::process::CommandExt;
                                    cmd.creation_flags(0x08000000);
                                }
                                
                                match cmd.spawn() {
                                    Ok(c) => {
                                        log::info!("[LAUNCHER] rigctld started (pid={})", c.id());
                                        rigctld_process = Some(c);
                                        launched_rig_model = rig_model;
                                        launched_serial_port = serial_port;
                                        launched_baud_rate = baud_rate;
                                        // Give rigctld time to open port and bind TCP 4532
                                        std::thread::sleep(std::time::Duration::from_millis(800));
                                    }
                                    Err(e) => log::error!("[LAUNCHER] Failed to start: {}", e),
                                }
                            }
                        }
                    }
                    UiCmd::ApplyAudio => {
                        let _ = tx_req_tx.send(TxRequest::ApplyAudio {
                            output_device: output_device.clone(),
                            output_level,
                            sample_rate,
                            buffer_size,
                            my_call: qso_engine.my_call.clone(),
                            their_call: qso_engine.their_call.clone().unwrap_or_default(),
                        });
                        rx_needs_restart = true;
                        
                        save_config(&AppConfig {
                            my_call: qso_engine.my_call.clone(),
                            input_device: input_device.clone(),
                            output_device: output_device.clone(),
                            rig_model: current_rig_model.clone(),
                            rig_port: current_rig_port.clone(),
                            rig_baud: current_rig_baud.clone(),
                        });
                        log::info!("[ENGINE] Audio settings applied & saved");
                    }

                    UiCmd::Listen { my_call: mc, their_call: tc, auto_slots: _ } => {
                        if !mc.is_empty() { 
                            qso_engine.set_my_call(mc);
                            save_config(&AppConfig {
                                my_call: qso_engine.my_call.clone(),
                                input_device: input_device.clone(),
                                output_device: output_device.clone(),
                                rig_model: current_rig_model.clone(),
                                rig_port: current_rig_port.clone(),
                                rig_baud: current_rig_baud.clone(),
                            });
                        }
                        
                        qso_engine.set_their_call(if tc.is_empty() { None } else { Some(tc) });

                        let (_, events) = qso_engine.on_intent(Intent::Listen);
                        process_qso_events(&events, &evt_tx, &rx_config_tx);

                        running = true;
                        observed_remote_slot = None;
                        was_calling_cq = false;
                        rx_needs_restart = true;
                        
                        // 🟢 CRITICAL FIX: Initialize audio output
                        let _ = tx_req_tx.send(TxRequest::ApplyAudio {
                            output_device: output_device.clone(),
                            output_level,
                            sample_rate,
                            buffer_size,
                            my_call: qso_engine.my_call.clone(),
                            their_call: qso_engine.their_call.clone().unwrap_or_default(),
                        });
                        
                        let _ = evt_tx.send(UiEvent::Info("LISTEN mode".into()));
                    }

                    UiCmd::StartCqWithGrid { my_call: mc, grid_indices: gi, auto_slots: _ } => {
                        if !mc.is_empty() { qso_engine.set_my_call(mc.clone()); }
                        
                        let (_, events) = qso_engine.on_intent(Intent::Cq);
                        process_qso_events(&events, &evt_tx, &rx_config_tx);

                        is_grid_mode = true;
                        current_grid_indices = Some(gi);

                        running = true;
                        last_slot_index = None;
                        observed_remote_slot = None;
                        was_calling_cq = true;
                        rx_needs_restart = true;
                        
                        // 🟢 CRITICAL FIX: Initialize TX audio output!
                        let _ = tx_req_tx.send(TxRequest::ApplyAudio {
                            output_device: output_device.clone(),
                            output_level,
                            sample_rate,
                            buffer_size,
                            my_call: qso_engine.my_call.clone(),
                            their_call: qso_engine.their_call.clone().unwrap_or_default(),
                        });
                        
                        let _ = evt_tx.send(UiEvent::Info("CQ mode (Maidenhead Active)".into()));
                    }

                    UiCmd::StartCq { my_call: mc, auto_slots: _ } => {
                        if !mc.is_empty() { qso_engine.set_my_call(mc); }
                        let (_, events) = qso_engine.on_intent(Intent::Cq);
                        process_qso_events(&events, &evt_tx, &rx_config_tx);

                        is_grid_mode = false;

                        running = true;
                        last_slot_index = None;
                        observed_remote_slot = None;
                        was_calling_cq = true;
                        rx_needs_restart = true;
                        
                        // 🟢 CRITICAL FIX: Initialize TX audio output!
                        let _ = tx_req_tx.send(TxRequest::ApplyAudio {
                            output_device: output_device.clone(),
                            output_level,
                            sample_rate,
                            buffer_size,
                            my_call: qso_engine.my_call.clone(),
                            their_call: qso_engine.their_call.clone().unwrap_or_default(),
                        });
                        
                        let _ = evt_tx.send(UiEvent::Info("CQ mode".into()));
                    }

                    UiCmd::CallStation { my_call: mc, their_call: tc } | UiCmd::ColdCall { my_call: mc, their_call: tc } => {
                        if !mc.is_empty() { qso_engine.set_my_call(mc); }
                        let (_, events) = qso_engine.on_intent(Intent::Call { their: tc.clone() });
                        process_qso_events(&events, &evt_tx, &rx_config_tx);
                        let _ = evt_tx.send(UiEvent::TheirCallChanged { 
                            callsign: tc.clone(),
                            grid: None, // 🟢 NEW: No grid when manually calling
                        });

                        running = true;
                        last_slot_index = None;
                        observed_remote_slot = None;
                        was_calling_cq = false;
                        let _ = tx_req_tx.send(TxRequest::ApplyAudio {
                            output_device: output_device.clone(),
                            output_level,
                            sample_rate,
                            buffer_size,
                            my_call: qso_engine.my_call.clone(),
                            their_call: tc,
                        });
                        rx_needs_restart = true;
                    }

                    UiCmd::AnswerCq { my_call: mc, their_call: tc, rpt, rx_slot, grid } => {
                        if !mc.is_empty() { qso_engine.set_my_call(mc); }
                        qso_engine.set_my_report(rpt);

                        if let Some(their_slot) = rx_slot {
                            observed_remote_slot = Some(their_slot);
                            update_rx_config(&rx_config_tx, RxConfigUpdate::SlotTiming {
                                my_tx_slot: 1 - their_slot,
                                rx_slot_override: Some(their_slot),
                                listen_all_slots: false,
                                slot_len_ms: slot_len_ms(slot_period) as u32,
                            });
                        }

                        let (_, events) = qso_engine.on_intent(Intent::AnswerCq { 
                            their: tc.clone(), 
                            rpt,
                            grid: grid.clone(), // 🟢 NEW: Pass grid to QSO engine
                        });
                        process_qso_events(&events, &evt_tx, &rx_config_tx);

                        running = true;
                        last_slot_index = None;
                        was_calling_cq = false;
                        let _ = tx_req_tx.send(TxRequest::ApplyAudio {
                            output_device: output_device.clone(),
                            output_level,
                            sample_rate,
                            buffer_size,
                            my_call: qso_engine.my_call.clone(),
                            their_call: qso_engine.their_call.clone().unwrap_or_default(),
                        });
                        rx_needs_restart = true;
                    }

                    UiCmd::Stop => {
                        running = false;
                        if let Some(h) = &hamlib { h.set_ptt(false); }
                        if tx_active { tx_active = false; let _ = evt_tx.send(UiEvent::TxActive(false)); }
                        let (_, events) = qso_engine.on_intent(Intent::Abort);
                        process_qso_events(&events, &evt_tx, &rx_config_tx);
                        observed_remote_slot = None;
                        was_calling_cq = false;
                        if let Some(st) = rx_stop_tx.take() { let _ = st.send(()); }
                        let _ = tx_req_tx.send(TxRequest::Stop);
                        let _ = evt_tx.send(UiEvent::Info("STOPPED".into()));
                    }
                    _ => {}
                }
            }

            Some(decoded) = rx_decoded_rx.recv() => {
                log::info!("📥 RX DECODE: text='{}', snr={:?}, slot={}, accumulated={}, from_call='{}', format={}",
                    decoded.msg.text, decoded.snr, decoded.rx_slot, decoded.is_accumulated,
                    decoded.msg.from_call, decoded.msg.format);
                let mut text = decoded.msg.text.clone();
                if decoded.is_accumulated { text.push_str(" [A]"); }
                let from_call = &decoded.msg.from_call;

                let _ = evt_tx.send(UiEvent::RxText {
                    text: text.clone(),
                    snr: decoded.snr,
                    utc_ms: decoded.utc_ms,
                    rx_slot: decoded.rx_slot,
                });

                if !qso_engine.my_call.is_empty() && from_call.to_uppercase() == qso_engine.my_call.to_uppercase() {
                    continue;
                }

                let payload = match proto::message_to_payload(&decoded.msg) {
                    Some(p) => p,
                    None => continue,
                };

                if observed_remote_slot.is_none()
                    || matches!(qso_engine.state, QsoState::Listening | QsoState::CallingCq | QsoState::CallingStn)
                {
                    observed_remote_slot = Some(decoded.rx_slot);
                    let my_new_tx_slot = 1 - decoded.rx_slot;
                    let slot_par = if my_new_tx_slot == 0 { SlotParity::Even } else { SlotParity::Odd };
                    let _ = evt_tx.send(UiEvent::TxSlotChanged { slot: slot_par });

                    update_rx_config(&rx_config_tx, RxConfigUpdate::SlotTiming {
                        my_tx_slot: my_new_tx_slot,
                        rx_slot_override: Some(decoded.rx_slot),
                        listen_all_slots: false,
                        slot_len_ms: slot_len_ms(slot_period) as u32,
                    });
                }
                
                if auto_qso {
                    let (action, events) = qso_engine.on_rx(RxEnvelope {
                        payload,
                        format: if decoded.msg.format == 1 { Format::Fmt1 } else { Format::Fmt2 },
                        snr: decoded.snr,
                        utc_ms: decoded.utc_ms,
                        rx_slot: decoded.rx_slot,
                    });

                    let mut qso_completed = false;
                    for event in &events {
                        match event {
                            EngineEvent::StateChanged(state) => {
                                if !qso_completed { let _ = evt_tx.send(UiEvent::State(state.to_string())); }
                            }
                            EngineEvent::Info(msg) => { let _ = evt_tx.send(UiEvent::Info(msg.clone())); }
                            EngineEvent::TheirCallChanged { callsign, grid } => {
                                let _ = evt_tx.send(UiEvent::TheirCallChanged { 
                                    callsign: callsign.clone(),
                                    grid: grid.clone(), // 🟢 NEW: Forward grid to UI
                                });
                                let tc = if callsign.is_empty() { None } else { Some(callsign.clone()) };
                                update_rx_config(&rx_config_tx, RxConfigUpdate::TheirCall(tc));
                            }
                            EngineEvent::QsoComplete { their, record } => {
                                qso_completed = true;
                                let _ = evt_tx.send(UiEvent::Info(format!("✓ QSO with {} complete", their)));
                                if let Some(rec) = record { let _ = evt_tx.send(UiEvent::QsoLogged { record: rec.clone() }); }
                                let _ = evt_tx.send(UiEvent::TheirCallChanged { 
                                    callsign: String::new(),
                                    grid: None, // 🟢 NEW: Clear grid when QSO completes
                                });

                                if was_calling_cq {
                                    let (_, ev2) = qso_engine.on_intent(Intent::Cq);
                                    process_qso_events(&ev2, &evt_tx, &rx_config_tx);
                                    
                                    // 🟢 CRITICAL FIX: Set running=true to actually transmit!
                                    running = true;
                                    last_slot_index = None;
                                    observed_remote_slot = None;
                                    
                                    let _ = evt_tx.send(UiEvent::State("CQ mode".into()));
                                } else {
                                    let (_, ev2) = qso_engine.on_intent(Intent::Listen);
                                    process_qso_events(&ev2, &evt_tx, &rx_config_tx);
                                    
                                    // 🟢 Also set running=true for Listen mode
                                    running = true;
                                    observed_remote_slot = None;
                                    
                                    let _ = evt_tx.send(UiEvent::State("Listening".into()));
                                }
                                update_rx_config(&rx_config_tx, RxConfigUpdate::TheirCall(None));
                            }
                            _ => {}
                        }
                    }

                    if let Action::Transmit(tx_env) = action {
                        let _ = evt_tx.send(UiEvent::TxText { text: tx_env.raw.clone() });
                        let _ = tx_req_tx.send(TxRequest::Text {
                            rendered: tx_env.raw,
                            slot_len_ms: slot_len_ms(slot_period) as u32,
                            my_call: qso_engine.my_call.clone(),
                            their_call: qso_engine.their_call.clone().unwrap_or_default(),
                        });
                    }
                }
            }
            Some(update) = ham_update_rx.recv() => {
                if let Some(freq) = update.freq {
                    log::debug!("[Hamlib] Freq: {} Hz", freq);
                    let _ = evt_tx.send(UiEvent::RigFreqChanged { freq_hz: freq });
                }
            }
            else => break,
        }
    }
    
    // Cleanup: kill rigctld on exit
    if let Some(mut child) = rigctld_process.take() {
        log::info!("[LAUNCHER] Shutting down rigctld (pid={})", child.id());
        let _ = child.kill();
        let _ = child.wait();
    }
    
    Ok(())
}

fn restart_rx(
    rx_decoded_tx: &mpsc::UnboundedSender<RxDecoded>,
    rx_stop_tx: &mut Option<mpsc::UnboundedSender<()>>,
    rx_config_tx: &mut Option<mpsc::UnboundedSender<RxConfigUpdate>>,
    cfg: RxAudioCfg,
) {
    // 🟢 Stop old RX first
    if let Some(st) = rx_stop_tx.take() {
        let _ = st.send(());
        log::info!("🛑 Stopping old RX, waiting for cleanup...");
        // Give old audio stream time to close (prevents conflicts)
        std::thread::sleep(Duration::from_millis(100));
    }
    
    let (stop_tx, stop_rx) = mpsc::unbounded_channel();
    *rx_stop_tx = Some(stop_tx);
    let (config_tx, config_rx) = mpsc::unbounded_channel();
    *rx_config_tx = Some(config_tx);

    log::info!("🎧 RX start - device={:?}, sr={}, buf={}, slot_len={}ms, my_call={}, their_call={:?}, decode_window={}s, my_tx_slot={}, rx_slot_override={:?}, listen_all={}",
        cfg.input_device, cfg.sample_rate, cfg.buffer_size, cfg.slot_len_ms,
        cfg.my_call, cfg.their_call, cfg.decode_window_secs,
        cfg.my_tx_slot, cfg.rx_slot_override, cfg.listen_all_slots);
    tokio::spawn(crate::modem::run_receiver(
        cfg,
        rx_decoded_tx.clone(),
        stop_rx,
        config_rx,
    ));
}

fn update_rx_config(
    rx_config_tx: &Option<mpsc::UnboundedSender<RxConfigUpdate>>,
    update: RxConfigUpdate,
) {
    if let Some(tx) = rx_config_tx {
        let _ = tx.send(update);
    }
}

fn process_qso_events(
    events: &[EngineEvent],
    evt_tx: &mpsc::UnboundedSender<UiEvent>,
    rx_config_tx: &Option<mpsc::UnboundedSender<RxConfigUpdate>>,
) {
    for event in events {
        match event {
            EngineEvent::StateChanged(state) => {
                let _ = evt_tx.send(UiEvent::State(state.to_string()));
            }
            EngineEvent::Info(msg) => {
                let _ = evt_tx.send(UiEvent::Info(msg.clone()));
            }
            EngineEvent::Tx(tx_env) => {
                let _ = evt_tx.send(UiEvent::TxText {
                    text: tx_env.raw.clone(),
                });
            }
            _ => {}
        }
    }
}