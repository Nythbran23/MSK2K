// src/gui/app.rs
use cpal::traits::{DeviceTrait, HostTrait};
use eframe::egui;
use std::collections::HashMap;
use std::process::Command;

use crate::engine::{EngineHandle, SlotParity, SlotPeriod, UiCmd, UiEvent};
use crate::engine::report_calc::report_from_correlation;
use crate::qso::adif::{AdifLogger, QsoRecord};

pub fn run_gui() -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 600.0])
            .with_min_inner_size([500.0, 250.0]),
        ..Default::default()
    };
    let engine = crate::engine::start_engine();

    if let Err(e) = eframe::run_native(
        "MSK2K",
        options,
        Box::new(|_cc| Ok(Box::new(Msk2kEguiApp::new(engine)))),
    ) {
        eprintln!("eframe error: {e}");
    }
    Ok(())
}

#[derive(Clone)]
struct LogEntry {
    text: String,
    colored: bool,
    timestamp: String,
}

struct Msk2kEguiApp {
    engine: EngineHandle,
    my_call: String,
    their_call: String,
    band: String,
    custom_band_input: String,
    rig_freq_hz: Option<u64>,
    tx_level: f32,
    slot_parity: SlotParity,
    slot_period: SlotPeriod,
    saved_slot_parity: Option<SlotParity>,
    in_devs: Vec<String>,
    out_devs: Vec<String>,
    sel_in: Option<String>,
    sel_out: Option<String>,
    settings_open: bool,
    is_listening: bool,
    is_calling_cq: bool,
    in_active_qso: bool,
    is_transmitting: bool,
    rx_log: Vec<LogEntry>,
    tx_log: Vec<LogEntry>,
    cq_log: Vec<LogEntry>,
    last_corr: f32,
    last_rx_slot: Option<u8>,
    decode_counts: HashMap<String, u32>,
    decode_log_index: HashMap<String, usize>,
    cq_log_index: HashMap<String, usize>,
    qso_log: Vec<QsoRecord>,
    qso_log_expanded: bool,
    adif_logger: AdifLogger,
    current_state: String,
    config: crate::settings::Config,
    
    // Rig Control State
    hamlib_enabled: bool,
    hamlib_address: String,
    launcher_enabled: bool,
    rig_model: String,
    rig_port: String,
    rig_baud: String,
    available_ports: Vec<String>,
    rig_list: Vec<(String, String)>, // Stores (ID, Name) e.g. ("3081", "Icom IC-9700")
    rig_search: String,
}

impl Msk2kEguiApp {
    fn new(engine: EngineHandle) -> Self {
        let (in_devs, out_devs) = enumerate_audio_devices();
        let adif_path = AdifLogger::default_path();
        let adif_logger = AdifLogger::new(&adif_path);
        let qso_log = adif_logger.read_all().unwrap_or_default();
        let config_path = crate::settings::default_config_path();
        let config = crate::settings::Config::load(&config_path).unwrap_or_default();
        let rig_cmd = get_bundled_rigctl_path();
        
        // We use "rigctld -l" because we bundled rigctld, not rigctl.
        // Luckily, rigctld -l produces the exact same list!
        // We also hide the console window on Windows to keep it clean.
        #[cfg(target_os = "windows")]
        // 🟢 CROSS-PLATFORM LIST LOADING
        let rig_cmd = get_bundled_rigctl_path();
        let mut rig_list = Vec::new();

        #[cfg(target_os = "windows")]
        let output = {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            Command::new(&rig_cmd)
                .arg("-l")
                .creation_flags(CREATE_NO_WINDOW)
                .output()
        };

        #[cfg(not(target_os = "windows"))]
        let output = Command::new(&rig_cmd).arg("-l").output();

        if let Ok(out) = output {
            if let Ok(text) = String::from_utf8(out.stdout) {
                for line in text.lines() {
                    // Skip headers or garbage lines
                    if line.trim().is_empty() || line.starts_with(" Rig") || line.len() < 10 { continue; }
                    
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 3 {
                        let id = parts[0].to_string();
                        // Combine Manufacturer + Model
                        // e.g. "Icom" + "IC-9700" = "Icom IC-9700"
                        let name = format!("{} {}", parts[1], parts[2]); 
                        rig_list.push((id, name));
                    }
                }
            }
        }
        Self {
            engine,
            my_call: "NOCALL".to_string(),
            their_call: String::new(),
            band: "2M".to_string(),
            custom_band_input: String::new(),
            rig_freq_hz: None,
            slot_parity: SlotParity::Odd,
            slot_period: SlotPeriod::S15,
            saved_slot_parity: None,
            in_devs,
            out_devs,
            sel_in: None,
            sel_out: None,
            settings_open: false,
            is_listening: false,
            is_calling_cq: false,
            in_active_qso: false,
            is_transmitting: false,
            rx_log: Vec::new(),
            tx_log: Vec::new(),
            cq_log: Vec::new(),
            last_corr: 0.0,
            last_rx_slot: None,
            decode_counts: HashMap::new(),
            decode_log_index: HashMap::new(),
            cq_log_index: HashMap::new(),
            qso_log,
            qso_log_expanded: false, 
            adif_logger,
            current_state: "Idle".to_string(),
            
            // Rig Control - load from saved config (read before config moves)
            hamlib_enabled: config.station.hamlib_enabled,
            hamlib_address: "127.0.0.1:4532".to_string(),
            launcher_enabled: false,
            rig_model: config.station.rig_model.clone(),
            rig_port: config.station.rig_port.clone(),
            rig_baud: config.station.rig_baud.clone(),
            tx_level: config.station.tx_level,

            config,
            available_ports: Vec::new(),
            rig_list,
            rig_search: String::new(),
        }
    }

    fn refresh_serial_ports(&mut self) {
        if let Ok(ports) = serialport::available_ports() {
            self.available_ports = ports.into_iter().map(|p| p.port_name).collect();
            self.available_ports.sort();
        }
    }

    fn calc_report(&self) -> i16 { report_from_correlation(self.last_corr).parse().unwrap_or(27) }

    fn color_history_for_call(&mut self, call: &str) {
        let t = call.to_uppercase();
        for entry in self.rx_log.iter_mut() { if entry.text.to_uppercase().contains(&t) { entry.colored = true; } }
        for entry in self.cq_log.iter_mut() { if entry.text.to_uppercase().contains(&t) { entry.colored = true; } }
    }

    fn reset_dedupe(&mut self) { self.decode_counts.clear(); self.decode_log_index.clear(); self.cq_log_index.clear(); }

    fn drain_events(&mut self) {
        while let Ok(ev) = self.engine.events.try_recv() {
            match ev {
                UiEvent::ConfigLoaded { my_call, input_device, output_device } => {
                    log::info!("[UI] ConfigLoaded event received");
                    
                    if !my_call.is_empty() { self.my_call = my_call.clone(); }
                    
                    if let Some(in_d) = input_device {
                        if self.in_devs.contains(&in_d) { self.sel_in = Some(in_d); }
                    }
                    if let Some(out_d) = output_device {
                        if self.out_devs.contains(&out_d) { self.sel_out = Some(out_d); }
                    }

                    if !self.my_call.is_empty() && self.my_call != "NOCALL" && self.sel_in.is_some() {
                        self.is_listening = true;
                        self.current_state = "Listening".to_string();
                        log::info!("[UI] Auto-start detected, UI is_listening = true");
                    }

                    // Auto-launch Hamlib if enabled in saved config
                    if self.hamlib_enabled && !self.rig_model.is_empty() && !self.rig_port.is_empty() {
                        log::info!("[UI] Auto-starting Hamlib: model={} port={}", self.rig_model, self.rig_port);
                        let _ = self.engine.cmds.send(UiCmd::ConfigureHamlib { 
                            enabled: true, 
                            address: "127.0.0.1:4532".to_string(),
                        });
                        let baud = self.rig_baud.parse().unwrap_or(19200);
                        let _ = self.engine.cmds.send(UiCmd::ConfigureLauncher { 
                            enable_launcher: true,
                            rig_model: self.rig_model.clone(),
                            serial_port: self.rig_port.clone(),
                            baud_rate: baud,
                        });
                    }
                },
                UiEvent::RxText { text, snr, utc_ms, rx_slot } => {
                    self.last_corr = snr.unwrap_or(self.last_corr);
                    self.last_rx_slot = Some(rx_slot);
                    
                    let ts = {
                        let secs = (utc_ms / 1000) as i64;
                        chrono::DateTime::from_timestamp(secs, 0)
                            .map(|dt| dt.format("%H:%M").to_string())
                            .unwrap_or_default()
                    };
                    
                    let key = text.to_uppercase().trim().to_string();
                    let count = self.decode_counts.entry(key.clone()).or_insert(0);
                    *count += 1;
                    
                    let display = format!("{} ({})", text, count);
                    let stamp = self.in_active_qso || !self.their_call.is_empty();
                    let is_cq = key.contains("CQ");

                    if is_cq {
                        if *count == 1 {
                            let _ = push_cap_entry(&mut self.cq_log, LogEntry { text: display.clone(), colored: stamp, timestamp: ts.clone() });
                            self.cq_log_index.insert(key.clone(), self.cq_log.len().saturating_sub(1));
                        } else if let Some(&idx) = self.cq_log_index.get(&key) {
                            if idx < self.cq_log.len() { 
                                self.cq_log[idx].text = display; 
                                self.cq_log[idx].timestamp = ts;
                                if stamp { self.cq_log[idx].colored = true; } 
                            }
                        }
                    } else {
                        if *count == 1 {
                            let _ = push_cap_entry(&mut self.rx_log, LogEntry { text: display.clone(), colored: stamp, timestamp: ts.clone() });
                            self.decode_log_index.insert(key.clone(), self.rx_log.len().saturating_sub(1));
                        } else if let Some(&idx) = self.decode_log_index.get(&key) {
                            if idx < self.rx_log.len() { 
                                self.rx_log[idx].text = display; 
                                self.rx_log[idx].timestamp = ts;
                                if stamp { self.rx_log[idx].colored = true; } 
                            }
                        }
                    }
                }
                UiEvent::TxText { text } => { push_cap_entry(&mut self.tx_log, LogEntry { text, colored: self.in_active_qso || !self.their_call.is_empty(), timestamp: String::new() }); }
                UiEvent::State(s) => {
                    self.current_state = s.clone();
                    if s.contains("Listening") { self.is_listening = true; }
                    else if s.contains("CallingCq") { self.is_listening = false; self.is_calling_cq = true; }
                    else if s.contains("Sending") || s.contains("CallingStn") { self.is_listening = false; self.is_calling_cq = false; self.in_active_qso = true; }
                }
                UiEvent::TheirCallChanged { callsign, grid } => { 
                    self.their_call = callsign.clone(); 
                    if !callsign.is_empty() { 
                        self.color_history_for_call(&callsign); 
                    }
                    let _ = grid; 
                }
                UiEvent::TxSlotChanged { slot } => {
                    if self.saved_slot_parity.is_none() {
                        self.saved_slot_parity = Some(self.slot_parity);
                    }
                    self.slot_parity = slot;
                }
                UiEvent::QsoLogged { record } => {
                    let mut updated_record = record;
                    updated_record.band = self.band.clone();
                    if let Some(freq) = self.rig_freq_hz {
                        let khz = ((freq + 500) / 1000) * 1000;
                        updated_record.freq = Some(khz as f64 / 1_000_000.0);
                    }
                    
                    let _ = self.adif_logger.log_qso(&updated_record); 
                    self.qso_log.insert(0, updated_record);
                    self.their_call = String::new(); 
                    self.in_active_qso = false;
                    self.reset_dedupe();
                    
                    if let Some(saved) = self.saved_slot_parity.take() {
                        self.slot_parity = saved;
                    }
                }
                UiEvent::RigFreqChanged { freq_hz } => {
                    self.rig_freq_hz = Some(freq_hz);
                    self.band = freq_to_band(freq_hz);
                }
                UiEvent::TxActive(active) => {
                    self.is_transmitting = active;
                }
                _ => {}
            }
        }
    }

    fn send_apply_audio(&mut self) {
        let _ = self.engine.cmds.send(UiCmd::SetInputDevice(self.sel_in.clone()));
        let _ = self.engine.cmds.send(UiCmd::SetOutputDevice(self.sel_out.clone()));
        let _ = self.engine.cmds.send(UiCmd::SetTxLevel(self.tx_level));
        let _ = self.engine.cmds.send(UiCmd::SetSlotParity(self.slot_parity));
        let _ = self.engine.cmds.send(UiCmd::SetSlotPeriod(self.slot_period));
        let _ = self.engine.cmds.send(UiCmd::ApplyAudio);
    }
}

impl eframe::App for Msk2kEguiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(egui::Visuals::dark());
        self.drain_events();

        if self.settings_open {
            let mut close = false;

            egui::Window::new("⚙ Settings").collapsible(false).resizable(true).default_width(450.0).show(ctx, |ui| {
                ui.heading("Station Setup");
                ui.horizontal(|ui| { 
                    ui.label("My Callsign:"); 
                    
                    let max_len = if self.config.station.use_grid_in_cq { 7 } else { 10 };
                    
                    if ui.text_edit_singleline(&mut self.my_call).changed() { 
                        self.my_call = self.my_call.to_uppercase();
                        if self.my_call.len() > max_len {
                            self.my_call.truncate(max_len);
                        }
                    }
                    
                    let count_color = if self.my_call.len() == max_len { 
                        egui::Color32::from_rgb(255, 180, 0) 
                    } else { 
                        egui::Color32::GRAY 
                    };
                    ui.label(
                        egui::RichText::new(format!("{}/{}", self.my_call.len(), max_len))
                            .small()
                            .color(count_color)
                    );
                });

                ui.separator();
                
                ui.heading("Maidenhead Locator");
                ui.horizontal(|ui| {
                    let alphabet: Vec<char> = "ABCDEFGHIJKLMNOPQR".chars().collect();
                    egui::ComboBox::from_id_salt("grid_f1")
                        .selected_text(alphabet[self.config.station.grid_indices[0]].to_string())
                        .width(40.0)
                        .show_ui(ui, |ui| {
                            for (i, c) in alphabet.iter().enumerate() {
                                ui.selectable_value(&mut self.config.station.grid_indices[0], i, c.to_string());
                            }
                        });
                    egui::ComboBox::from_id_salt("grid_f2")
                        .selected_text(alphabet[self.config.station.grid_indices[1]].to_string())
                        .width(40.0)
                        .show_ui(ui, |ui| {
                            for (i, c) in alphabet.iter().enumerate() {
                                ui.selectable_value(&mut self.config.station.grid_indices[1], i, c.to_string());
                            }
                        });
                    egui::ComboBox::from_id_salt("grid_s1")
                        .selected_text(self.config.station.grid_indices[2].to_string())
                        .width(40.0)
                        .show_ui(ui, |ui| {
                            for n in 0..10 {
                                ui.selectable_value(&mut self.config.station.grid_indices[2], n, n.to_string());
                            }
                        });
                    egui::ComboBox::from_id_salt("grid_s2")
                        .selected_text(self.config.station.grid_indices[3].to_string())
                        .width(40.0)
                        .show_ui(ui, |ui| {
                            for n in 0..10 {
                                ui.selectable_value(&mut self.config.station.grid_indices[3], n, n.to_string());
                            }
                        });
                });

                ui.checkbox(&mut self.config.station.use_grid_in_cq, "Encode Grid in CQ (MSK2K Mode)");
                
                let max_call_len = if self.config.station.use_grid_in_cq { 7 } else { 10 };
                if self.my_call.len() > max_call_len {
                    self.my_call.truncate(max_call_len);
                }
                
                ui.add_space(5.0);
                ui.label(egui::RichText::new("Note: Grid mode limits callsign to 7 characters.").small().color(egui::Color32::GRAY));
                
                // 🟢 RIG CONTROL SECTION
                ui.separator();
                ui.heading("Rig Control (Hamlib)");
                
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.hamlib_enabled, "Enable CAT Control");
                });

                if self.hamlib_enabled {
                    ui.indent("cat_indent", |ui| {
                        egui::Grid::new("launcher_grid").num_columns(2).spacing([10.0, 8.0]).show(ui, |ui| {
                            ui.label("Rig Selection:");
                            ui.vertical(|ui| {
                                ui.text_edit_singleline(&mut self.rig_search)
                                    .on_hover_text("Type your radio name (e.g. '7300') to filter");
                                
                                let current_name = if self.rig_model.is_empty() {
                                    "Select Rig...".to_string()
                                } else {
                                    self.rig_list.iter()
                                        .find(|(id, _)| id == &self.rig_model)
                                        .map(|(_, name)| name.clone())
                                        .unwrap_or_else(|| format!("ID: {}", self.rig_model))
                                };

                                egui::ComboBox::from_id_salt("rig_select")
                                    .selected_text(current_name)
                                    .width(250.0)
                                    .show_ui(ui, |ui| {
                                        let search_upper = self.rig_search.to_uppercase();
                                        for (id, name) in &self.rig_list {
                                            if self.rig_search.is_empty() || name.to_uppercase().contains(&search_upper) || id.contains(&search_upper) {
                                                if ui.selectable_value(&mut self.rig_model, id.clone(), name).clicked() {
                                                }
                                            }
                                        }
                                    });
                            });
                            ui.end_row();

                            ui.label("Serial Port:");
                            ui.horizontal(|ui| {
                                let port_display = if self.rig_port.is_empty() { "Select Port...".to_string() } else { self.rig_port.clone() };
                                egui::ComboBox::from_id_salt("ser_port")
                                    .selected_text(&port_display)
                                    .width(350.0)
                                    .show_ui(ui, |ui| {
                                        for p in &self.available_ports {
                                            ui.selectable_value(&mut self.rig_port, p.clone(), p);
                                        }
                                    });
                                if ui.button("🔄").clicked() { self.refresh_serial_ports(); }
                            });
                            ui.end_row();

                            ui.label("Baud Rate:");
                            ui.text_edit_singleline(&mut self.rig_baud);
                            ui.end_row();
                        });
                    });
                }
                
                ui.separator();
                ui.heading("Audio Hardware");
                ui.label("Input Device:");
                egui::ComboBox::from_id_salt("in_dev").selected_text(self.sel_in.clone().unwrap_or_else(|| "Default".into())).show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.sel_in, None, "Default");
                    for d in &self.in_devs { ui.selectable_value(&mut self.sel_in, Some(d.clone()), d.clone()); }
                });
                
                ui.add_space(10.0);
                ui.label("Output Device:");
                egui::ComboBox::from_id_salt("out_dev").selected_text(self.sel_out.clone().unwrap_or_else(|| "Default".into())).show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.sel_out, None, "Default");
                    for d in &self.out_devs { ui.selectable_value(&mut self.sel_out, Some(d.clone()), d.clone()); }
                });
                
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.label("TX Level:");
                    let slider = egui::Slider::new(&mut self.tx_level, 0.0..=1.0)
                        .custom_formatter(|v, _| format!("{}%", (v * 100.0).round() as i32))
                        .custom_parser(|s| s.trim_end_matches('%').parse::<f64>().ok().map(|v| v / 100.0));
                    if ui.add(slider).changed() {
                        let _ = self.engine.cmds.send(UiCmd::SetTxLevel(self.tx_level));
                    }
                });
                
                ui.add_space(15.0);
                if ui.button("Save & Close").clicked() { close = true; }
            });
            
            if close { 
                self.send_apply_audio();
                
                // 1. Configure Hamlib
                let _ = self.engine.cmds.send(UiCmd::ConfigureHamlib { 
                    enabled: self.hamlib_enabled, 
                    address: "127.0.0.1:4532".to_string(),
                });

                // 2. Configure Launcher (auto-start rigctld when CAT enabled)
                if self.hamlib_enabled {
                    let baud = self.rig_baud.parse().unwrap_or(19200);
                    let _ = self.engine.cmds.send(UiCmd::ConfigureLauncher { 
                        enable_launcher: true,
                        rig_model: self.rig_model.clone(),
                        serial_port: self.rig_port.clone(),
                        baud_rate: baud,
                    });
                } else {
                     let _ = self.engine.cmds.send(UiCmd::ConfigureLauncher { 
                        enable_launcher: false, rig_model: String::new(), serial_port: String::new(), baud_rate: 0 
                    });
                }

                // 🟢 3. RESTORE LISTENING MODE (This is the code you wanted!)
                let _ = self.engine.cmds.send(UiCmd::Listen {
                    my_call: self.my_call.clone(),
                    their_call: String::new(),
                    auto_slots: true,
                });
                self.is_listening = true;
                self.is_calling_cq = false;
                self.current_state = "Listening".to_string();

                // 4. Save Config (including rig settings)
                self.config.station.hamlib_enabled = self.hamlib_enabled;
                self.config.station.rig_model = self.rig_model.clone();
                self.config.station.rig_port = self.rig_port.clone();
                self.config.station.rig_baud = self.rig_baud.clone();
                self.config.station.tx_level = self.tx_level;
                let _ = self.config.save(&crate::settings::default_config_path());
                self.settings_open = false; 
            }
        }

        // ... (Rest of GUI: Top Bar, Actions, Logbook) ...
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.add_space(5.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(&self.my_call).strong().size(22.0).color(egui::Color32::GRAY));
                ui.add_space(20.0);
                
                // Frequency from rig (or "No CAT" if not connected)
                let freq_display = if let Some(freq) = self.rig_freq_hz {
                    let khz = ((freq + 500) / 1000) * 1000;
                    format!("{:.3} MHz", khz as f64 / 1_000_000.0)
                } else {
                    "No CAT".to_string()
                };
                ui.label(egui::RichText::new(&freq_display).monospace().size(16.0).color(egui::Color32::from_rgb(100, 200, 130)));
                
                let saved_selection = ui.visuals().selection.bg_fill;
                let saved_inactive_bg = ui.visuals().widgets.inactive.weak_bg_fill;
                let slate = egui::Color32::from_rgb(70, 90, 110);
                ui.visuals_mut().selection.bg_fill = slate;
                ui.visuals_mut().widgets.inactive.weak_bg_fill = slate;
                
                ui.add_space(15.0); ui.label("PERIOD:");
                if ui.selectable_value(&mut self.slot_period, SlotPeriod::S15, "15s").changed() || ui.selectable_value(&mut self.slot_period, SlotPeriod::S30, "30s").changed() { let _ = self.engine.cmds.send(UiCmd::SetSlotPeriod(self.slot_period)); }
                ui.add_space(15.0); ui.label("TX SLOT:");
                if ui.selectable_value(&mut self.slot_parity, SlotParity::Even, "Even").changed() || ui.selectable_value(&mut self.slot_parity, SlotParity::Odd, "Odd").changed() { let _ = self.engine.cmds.send(UiCmd::SetSlotParity(self.slot_parity)); }
                ui.visuals_mut().selection.bg_fill = saved_selection;
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("⚙").clicked() { 
                        self.settings_open = true; 
                        self.refresh_serial_ports();
                    }
                    ui.label(egui::RichText::new(chrono::Utc::now().format("%H:%M:%S Z").to_string()).monospace());
                });
            });
        });

        egui::TopBottomPanel::top("actions").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let blue = egui::Color32::from_rgb(56, 120, 70);
                if ui.add_sized([90.0, 30.0], egui::Button::new("📻 LISTEN").fill(if self.is_listening { blue } else { egui::Color32::from_rgb(45, 45, 45) })).clicked() {
                    if self.is_listening {
                        log::info!("[UI] LISTEN button clicked - STOPPING");
                        self.is_listening = false;
                        self.is_calling_cq = false;
                        let _ = self.engine.cmds.send(UiCmd::Stop);
                    } else {
                        log::info!("[UI] LISTEN button clicked - STARTING");
                        log::info!("   my_call: {}", self.my_call);
                        self.is_listening = true;
                        self.is_calling_cq = false;
                        self.their_call = String::new();
                        let _ = self.engine.cmds.send(UiCmd::Listen { 
                            my_call: self.my_call.clone(), 
                            their_call: String::new(), 
                            auto_slots: true 
                        });
                        log::info!("[UI] Listen command sent");
                    }
                }
                
                if ui.add_sized([90.0, 30.0], egui::Button::new("📢 CALL CQ").fill(if self.is_calling_cq { blue } else { egui::Color32::from_rgb(45, 45, 45) })).clicked() {
                    if self.is_calling_cq { 
                        self.is_calling_cq = false; 
                        let _ = self.engine.cmds.send(UiCmd::Stop); 
                    } else { 
                        self.is_listening = false; 
                        self.is_calling_cq = true; 
                        self.their_call = String::new(); 
                        
                        if self.config.station.use_grid_in_cq {
                            let _ = self.engine.cmds.send(UiCmd::StartCqWithGrid { 
                                my_call: self.my_call.clone(), 
                                grid_indices: self.config.station.grid_indices,
                                auto_slots: true 
                            });
                        } else {
                            let _ = self.engine.cmds.send(UiCmd::StartCq { 
                                my_call: self.my_call.clone(), 
                                auto_slots: true 
                            });
                        }
                    }
                }
                
                ui.add_space(20.0); 
                ui.label("TARGET:");
                
                ui.horizontal(|ui| {
                    if ui.text_edit_singleline(&mut self.their_call).changed() {
                        self.their_call = self.their_call.to_uppercase();
                        if self.their_call.len() > 10 {
                            self.their_call.truncate(10);
                        }
                    }
                    
                    if !self.their_call.is_empty() {
                        let count_color = if self.their_call.len() == 10 { 
                            egui::Color32::from_rgb(255, 180, 0) 
                        } else { 
                            egui::Color32::GRAY 
                        };
                        ui.label(
                            egui::RichText::new(format!("{}/10", self.their_call.len()))
                                .small()
                                .color(count_color)
                        );
                    }
                });
                if ui.button("CALL").clicked() {
                    let t = self.their_call.clone(); self.color_history_for_call(&t);
                    let _ = self.engine.cmds.send(UiCmd::CallStation { my_call: self.my_call.clone(), their_call: t });
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("⏹ STOP").clicked() { self.their_call = String::new(); self.is_calling_cq = false; self.is_listening = false; let _ = self.engine.cmds.send(UiCmd::Stop); }
                    if self.in_active_qso || !self.their_call.is_empty() {
                        ui.add_space(10.0);
                        let green = egui::Color32::from_rgb(56, 120, 70);
                        ui.add(egui::Button::new(egui::RichText::new("IN QSO").strong().color(egui::Color32::WHITE)).fill(green).sense(egui::Sense::hover()));
                    }
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let mut clicked_call: Option<String> = None;
            let mut clicked_grid: Option<String> = None; 
            ui.columns(3, |cols| {
                for i in 0..3 {
                    cols[i].vertical(|ui| {
                        ui.horizontal(|ui| {
                            let label = match i { 0 => "📥 RX", 1 => "📤 TX", _ => "🎯 SPOTS" };
                            if i == 1 && self.is_transmitting {
                                ui.heading(egui::RichText::new(label).color(egui::Color32::from_rgb(255, 50, 50)));
                            } else {
                                ui.heading(label);
                            }
                            
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.small_button("🗑").clicked() { match i { 0 => self.rx_log.clear(), 1 => self.tx_log.clear(), _ => self.cq_log.clear() }; }
                            });
                        });
                        
                        let id = match i { 0 => "rx_sc", 1 => "tx_sc", _ => "cq_sc" };
                        let available_h = ui.available_height();
                        egui::ScrollArea::vertical().id_salt(id).max_height(available_h).stick_to_bottom(false).show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            let log = match i { 0 => &self.rx_log, 1 => &self.tx_log, _ => &self.cq_log };
                            for entry in log.iter().rev() {
                                if i == 1 {
                                    ui.allocate_ui(egui::vec2(ui.available_width(), 18.0), |ui| {
                                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                            if entry.colored {
                                                egui::Frame::none()
                                                    .fill(egui::Color32::from_rgba_unmultiplied(31, 111, 235, 6))
                                                    .inner_margin(egui::Margin::symmetric(6.0, 2.0))
                                                    .show(ui, |ui| {
                                                        ui.monospace(&entry.text);
                                                    });
                                                let rect = ui.max_rect();
                                                let bar_rect = egui::Rect::from_min_max(
                                                    egui::pos2(rect.max.x - 3.0, rect.min.y),
                                                    egui::pos2(rect.max.x, rect.max.y)
                                                );
                                                ui.painter().rect_filled(bar_rect, 0.0, egui::Color32::from_rgb(31, 111, 235));
                                            } else {
                                                ui.monospace(&entry.text);
                                            }
                                        });
                                    });
                                } else {
                                    ui.allocate_ui(egui::vec2(ui.available_width(), 18.0), |ui| {
                                        let rect = ui.available_rect_before_wrap();
                                        let color = if entry.colored { 
                                            egui::Color32::from_rgba_unmultiplied(31, 143, 58, 6)
                                        } else { 
                                            egui::Color32::TRANSPARENT 
                                        };
                                        
                                        egui::Frame::none().fill(color).inner_margin(egui::Margin::symmetric(10.0, 2.0)).show(ui, |ui| {
                                            if entry.colored {
                                                let bar_rect = egui::Rect::from_min_max(
                                                    egui::pos2(rect.min.x, rect.min.y), 
                                                    egui::pos2(rect.min.x + 3.0, rect.max.y)
                                                );
                                                ui.painter().rect_filled(bar_rect, 0.0, egui::Color32::from_rgb(31, 143, 58));
                                            }
                                            
                                            ui.horizontal(|ui| {
                                                if ui.selectable_label(false, &entry.text).clicked() { 
                                                    if let Some((call, grid)) = extract_callsign_and_grid(&entry.text) { 
                                                        clicked_call = Some(call);
                                                        clicked_grid = grid;
                                                    } 
                                                }
                                                if !entry.timestamp.is_empty() {
                                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                        ui.label(egui::RichText::new(&entry.timestamp).small().color(egui::Color32::GRAY));
                                                    });
                                                }
                                            });
                                        });
                                    });
                                }
                            }
                        });
                    });
                }
            });

            if let Some(target) = clicked_call { 
                self.their_call = target.clone(); self.color_history_for_call(&target);
                let _ = self.engine.cmds.send(UiCmd::AnswerCq { 
                    my_call: self.my_call.clone(), 
                    their_call: target, 
                    rpt: self.calc_report(), 
                    rx_slot: self.last_rx_slot,
                    grid: clicked_grid, 
                }); 
            }
        });

        egui::TopBottomPanel::bottom("log_footer").resizable(true).min_height(30.0).show(ctx, |ui| {
            let arrow = if self.qso_log_expanded { "▼" } else { "▶" };
            if ui.add(egui::Button::new(egui::RichText::new(format!("Logbook {}", arrow)).color(egui::Color32::WHITE)).frame(false)).clicked() {
                self.qso_log_expanded = !self.qso_log_expanded;
            }

            if self.qso_log_expanded {
                ui.add_space(5.0);
                ui.separator();
                
                egui::Grid::new("log_header_grid")
                    .num_columns(8)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.add_sized([80.0, 20.0], egui::Label::new(egui::RichText::new("DATE").strong().color(egui::Color32::GRAY)));
                        ui.add_sized([65.0, 20.0], egui::Label::new(egui::RichText::new("START (Z)").strong().color(egui::Color32::GRAY)));
                        ui.add_sized([65.0, 20.0], egui::Label::new(egui::RichText::new("END (Z)").strong().color(egui::Color32::GRAY)));
                        ui.add_sized([75.0, 20.0], egui::Label::new(egui::RichText::new("FREQ").strong().color(egui::Color32::GRAY)));
                        ui.add_sized([90.0, 20.0], egui::Label::new(egui::RichText::new("STATION").strong().color(egui::Color32::GRAY)));
                        ui.add_sized([50.0, 20.0], egui::Label::new(egui::RichText::new("GRID").strong().color(egui::Color32::GRAY)));
                        ui.add_sized([40.0, 20.0], egui::Label::new(egui::RichText::new("SENT").strong().color(egui::Color32::GRAY)));
                        ui.add_sized([40.0, 20.0], egui::Label::new(egui::RichText::new("RCVD").strong().color(egui::Color32::GRAY)));
                        ui.end_row();
                    });

                ui.add_space(2.0);
                ui.separator();

                egui::ScrollArea::vertical().id_salt("log_sc_data").max_height(200.0).show(ui, |ui| {
                    egui::Grid::new("log_data_grid")
                        .striped(true)
                        .num_columns(8)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            for q in &self.qso_log {
                                ui.add_sized([80.0, 18.0], egui::Label::new(&q.display_date()));
                                ui.add_sized([65.0, 18.0], egui::Label::new(&q.display_time_on()));
                                ui.add_sized([65.0, 18.0], egui::Label::new(&q.display_time_off()));
                                let freq_str = q.freq.map(|f| format!("{:.3}", f)).unwrap_or_else(|| "—".to_string());
                                ui.add_sized([75.0, 18.0], egui::Label::new(&freq_str));
                                ui.add_sized([90.0, 18.0], egui::Label::new(egui::RichText::new(&q.call).strong().color(egui::Color32::LIGHT_BLUE)));
                                ui.add_sized([50.0, 18.0], egui::Label::new(q.gridsquare.as_deref().unwrap_or("—")));
                                ui.add_sized([40.0, 18.0], egui::Label::new(&q.rst_sent));
                                ui.add_sized([40.0, 18.0], egui::Label::new(&q.rst_rcvd));
                                ui.end_row();
                            }
                        });
                });
            }
        });

        egui::TopBottomPanel::bottom("status_strip").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.small(format!("STATE: {}", self.current_state));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.last_corr > 0.0 { ui.small(format!("Quality: {}%", (self.last_corr * 100.0) as i32)); }
                });
            });
        });
        
        ctx.request_repaint();
    }
}

fn extract_callsign_and_grid(text: &str) -> Option<(String, Option<String>)> {
    let t = text.to_uppercase();
    
    let clean_text = if let Some(pos) = t.find('(') {
        t[..pos].trim()
    } else {
        t.trim()
    };
    
    let parts: Vec<&str> = clean_text.split_whitespace().collect();
    
    if parts.len() >= 3 && parts[0] == "CQ" && parts[1] == "DE" {
        let call = parts[2];
        if parts.len() >= 4 {
            let potential_grid = parts[3];
            if potential_grid.len() == 4 
                && potential_grid.chars().take(2).all(|c| c.is_alphabetic())
                && potential_grid.chars().skip(2).all(|c| c.is_numeric()) 
            {
                return Some((call.to_string(), Some(potential_grid.to_string())));
            }
        }
        return Some((call.to_string(), None));
    }
    
    if parts.len() >= 3 && parts[0] == "CQ" {
        let call = parts[1];
        let potential_grid = parts[2];
        
        if potential_grid.len() == 4 
            && potential_grid.chars().take(2).all(|c| c.is_alphabetic())
            && potential_grid.chars().skip(2).all(|c| c.is_numeric()) 
        {
            return Some((call.to_string(), Some(potential_grid.to_string())));
        } else {
            return Some((call.to_string(), None));
        }
    }
    
    if parts.len() == 2 && parts[0] == "CQ" {
        return Some((parts[1].to_string(), None));
    }
    
    if clean_text.contains(" DE ") { 
        let call = clean_text.split(" DE ").nth(1)?.split_whitespace().next().map(|s| s.to_string())?;
        return Some((call, None));
    }
    
    if parts.len() >= 2 {
        let potential_call = parts[0];
        let potential_grid = parts[1];
        
        if potential_grid.len() == 4
            && potential_grid.chars().take(2).all(|c| c.is_alphabetic())
            && potential_grid.chars().skip(2).all(|c| c.is_numeric())
        {
            return Some((potential_call.to_string(), Some(potential_grid.to_string())));
        }
    }
    
    None
}

fn extract_callsign(text: &str) -> Option<String> {
    extract_callsign_and_grid(text).map(|(call, _)| call)
}

fn push_cap_entry(v: &mut Vec<LogEntry>, s: LogEntry) { if v.len() >= 100 { v.remove(0); } v.push(s); }

fn enumerate_audio_devices() -> (Vec<String>, Vec<String>) {
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = cpal::default_host();
    
    // Collect ALL devices with their capabilities
    let mut all_devices: Vec<(String, bool, bool)> = Vec::new(); // (name, has_input, has_output)
    
    if let Ok(devs) = host.devices() {
        for d in devs {
            if let Ok(name) = d.name() {
                let has_in = d.supported_input_configs().map(|mut c| c.next().is_some()).unwrap_or(false)
                    || d.default_input_config().is_ok();
                let has_out = d.supported_output_configs().map(|mut c| c.next().is_some()).unwrap_or(false)
                    || d.default_output_config().is_ok();
                all_devices.push((name, has_in, has_out));
            }
        }
    }
    
    all_devices.sort_by(|a, b| a.0.cmp(&b.0));
    
    // For duplicate names, label them with capabilities to help the user
    let mut name_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (name, _, _) in &all_devices {
        *name_counts.entry(name.clone()).or_insert(0) += 1;
    }
    
    // First pass: collect capabilities per duplicate group
    let mut group_caps: std::collections::HashMap<String, Vec<(bool, bool)>> = std::collections::HashMap::new();
    for (name, has_in, has_out) in &all_devices {
        if name_counts[name.as_str()] > 1 {
            group_caps.entry(name.clone()).or_default().push((*has_in, *has_out));
        }
    }
    
    let mut name_indices: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let display_names: Vec<String> = all_devices.iter().map(|(name, has_in, has_out)| {
        if name_counts[name.as_str()] > 1 {
            let idx = name_indices.entry(name.clone()).or_insert(0);
            *idx += 1;
            let caps = &group_caps[name.as_str()];
            let has_rx_sibling = caps.iter().any(|(i, _)| *i);
            let has_tx_sibling = caps.iter().any(|(_, o)| *o);
            
            let label = match (*has_in, *has_out) {
                (true, false) => "RX".to_string(),
                (false, true) => "TX".to_string(),
                (true, true)  => "RX/TX".to_string(),
                (false, false) => {
                    // Can't detect capability — infer from sibling
                    if has_rx_sibling && !has_tx_sibling {
                        "TX".to_string()  // Sibling is RX, so this must be TX
                    } else if has_tx_sibling && !has_rx_sibling {
                        "RX".to_string()  // Sibling is TX, so this must be RX
                    } else {
                        format!("{}", idx)  // Can't infer, use number
                    }
                }
            };
            format!("{} ({})", name, label)
        } else {
            name.clone()
        }
    }).collect();
    
    log::info!("[AUDIO] All devices: {:?}", display_names);
    
    (display_names.clone(), display_names)
}

// 🟢 PASTE AT BOTTOM OF src/gui/app.rs

fn get_bundled_rigctl_path() -> String {
    // 1. Determine binary name based on OS
    let binary_name = if cfg!(target_os = "windows") { "rigctld.exe" } else { "rigctld" };
    
    // 2. Find the current executable path
    if let Ok(mut path) = std::env::current_exe() {
        path.pop(); // Remove "msk2k" filename
        
        // 3. Check "tools" folder (Release / Bundled)
        let tools_path = path.join("tools").join(binary_name);
        if tools_path.exists() {
            return tools_path.to_string_lossy().to_string();
        }

        // 4. Check same folder (Development / Flat structure)
        let local_path = path.join(binary_name);
        if local_path.exists() {
            return local_path.to_string_lossy().to_string();
        }
    }
    
    // 5. Fallback: Assume installed globally (Homebrew/Linux package)
    // On Mac/Linux, this is often where it lives if not bundled correctly.
    "rigctld".to_string()
}

/// Map a frequency in Hz to amateur band name
fn freq_to_band(freq_hz: u64) -> String {
    let khz = ((freq_hz + 500) / 1000) * 1000;
    let mhz = khz as f64 / 1_000_000.0;
    match mhz {
        f if (1.8..=2.0).contains(&f) => "160M".to_string(),
        f if (3.5..=4.0).contains(&f) => "80M".to_string(),
        f if (5.0..=5.5).contains(&f) => "60M".to_string(),
        f if (7.0..=7.3).contains(&f) => "40M".to_string(),
        f if (10.0..=10.2).contains(&f) => "30M".to_string(),
        f if (14.0..=14.35).contains(&f) => "20M".to_string(),
        f if (18.0..=18.2).contains(&f) => "17M".to_string(),
        f if (21.0..=21.45).contains(&f) => "15M".to_string(),
        f if (24.8..=25.0).contains(&f) => "12M".to_string(),
        f if (28.0..=29.7).contains(&f) => "10M".to_string(),
        f if (50.0..=54.0).contains(&f) => "6M".to_string(),
        f if (70.0..=70.5).contains(&f) => "4M".to_string(),
        f if (144.0..=148.0).contains(&f) => "2M".to_string(),
        f if (220.0..=225.0).contains(&f) => "1.25M".to_string(),
        f if (420.0..=450.0).contains(&f) => "70CM".to_string(),
        f if (902.0..=928.0).contains(&f) => "33CM".to_string(),
        f if (1240.0..=1300.0).contains(&f) => "23CM".to_string(),
        _ => format!("{:.3}", mhz),
    }
}