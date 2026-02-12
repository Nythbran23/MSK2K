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
        let mut rig_list = Vec::new();
        // Try running "rigctl -l"
        if let Ok(output) = Command::new("rigctl").arg("-l").output() {
            if let Ok(text) = String::from_utf8(output.stdout) {
                for line in text.lines() {
                    // Skip header or short lines
                    if line.starts_with(" Rig") || line.len() < 10 { continue; }
                    
                    // Parse: "3081  Icom  IC-9700 ..."
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 3 {
                        let id = parts[0].to_string();
                        // Combine Mfg + Model for display (e.g. "Icom IC-9700")
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
            config,
            
            // Rig Control Defaults
            hamlib_enabled: true, 
            hamlib_address: "127.0.0.1:4532".to_string(),
            launcher_enabled: false,
            rig_model: "3081".to_string(), // Default to IC-9700
            rig_port: String::new(),
            rig_baud: "19200".to_string(),
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
                    
                    let _ = self.adif_logger.log_qso(&updated_record); 
                    self.qso_log.insert(0, updated_record);
                    self.their_call = String::new(); 
                    self.in_active_qso = false;
                    self.reset_dedupe();
                    
                    if let Some(saved) = self.saved_slot_parity.take() {
                        self.slot_parity = saved;
                    }
                }
                _ => {}
            }
        }
    }

    fn send_apply_audio(&mut self) {
        let _ = self.engine.cmds.send(UiCmd::SetInputDevice(self.sel_in.clone()));
        let _ = self.engine.cmds.send(UiCmd::SetOutputDevice(self.sel_out.clone()));
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
            // 🟢 WIDER SETTINGS WINDOW
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
                        ui.checkbox(&mut self.launcher_enabled, "Auto-Start Rigctld (Embedded)");
                        
                        if self.launcher_enabled {
                            egui::Grid::new("launcher_grid").num_columns(2).spacing([10.0, 8.0]).show(ui, |ui| {
                                // 🟢 NEW RIG SELECTOR
                                ui.label("Rig Selection:");
                                ui.vertical(|ui| {
                                    // 1. Search Box
                                    ui.text_edit_singleline(&mut self.rig_search)
                                        .on_hover_text("Type your radio name (e.g. '7300') to filter");
                                    
                                    // 2. Dropdown
                                    let current_name = self.rig_list.iter()
                                        .find(|(id, _)| id == &self.rig_model)
                                        .map(|(_, name)| name.clone())
                                        .unwrap_or_else(|| format!("ID: {}", self.rig_model));

                                    egui::ComboBox::from_id_salt("rig_select")
                                        .selected_text(current_name)
                                        .width(250.0)
                                        .show_ui(ui, |ui| {
                                            // Filter the massive list based on user typing
                                            let search_upper = self.rig_search.to_uppercase();
                                            for (id, name) in &self.rig_list {
                                                if self.rig_search.is_empty() || name.to_uppercase().contains(&search_upper) || id.contains(&search_upper) {
                                                    if ui.selectable_value(&mut self.rig_model, id.clone(), name).clicked() {
                                                        // Auto-clear search on select if you want
                                                    }
                                                }
                                            }
                                        });
                                });
                                ui.end_row();

                                ui.label("Serial Port:");
                                ui.horizontal(|ui| {
                                    egui::ComboBox::from_id_salt("ser_port")
                                        .selected_text(&self.rig_port)
                                        .width(200.0)
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
                        } else {
                            ui.horizontal(|ui| {
                                ui.label("Network Address:");
                                ui.text_edit_singleline(&mut self.hamlib_address);
                            });
                            ui.label(egui::RichText::new("Ensure 'rigctld' is running externally.").small().italics().color(egui::Color32::GRAY));
                        }
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
                
                ui.add_space(15.0);
                if ui.button("Save & Close").clicked() { close = true; }
            });
            
            if close { 
                self.send_apply_audio();
                
                // 1. Configure Hamlib
                let address = if self.launcher_enabled { "127.0.0.1:4532".to_string() } else { self.hamlib_address.clone() };
                let _ = self.engine.cmds.send(UiCmd::ConfigureHamlib { 
                    enabled: self.hamlib_enabled, 
                    address 
                });

                // 2. Configure Launcher
                if self.launcher_enabled {
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

                // 4. Save Config
                let _ = self.config.save(&crate::settings::default_config_path());
                self.settings_open = false; 
            }
        }

        // ... (Rest of GUI: Top Bar, Actions, Logbook) ...
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(&self.my_call).strong().size(22.0).color(egui::Color32::GRAY));
                ui.add_space(20.0); ui.label("BAND:");
                
                let display_band = self.band.clone(); 
                let saved_selection = ui.visuals().selection.bg_fill;
                let saved_inactive_bg = ui.visuals().widgets.inactive.weak_bg_fill;
                let slate = egui::Color32::from_rgb(70, 90, 110);
                ui.visuals_mut().selection.bg_fill = slate;
                ui.visuals_mut().widgets.inactive.weak_bg_fill = slate;
                egui::ComboBox::from_id_salt("band").selected_text(&display_band).width(100.0).show_ui(ui, |ui| {
                    for b in &["6M", "4M", "2M", "70CM"] { 
                        ui.selectable_value(&mut self.band, b.to_string(), *b);
                    }
                    ui.separator();
                    
                    ui.label("Custom:");
                    let text_edit = egui::TextEdit::singleline(&mut self.custom_band_input)
                        .hint_text("e.g. 144.350")
                        .desired_width(80.0);
                    
                    let response = ui.add(text_edit);
                    response.request_focus();
                    
                    if response.changed() {
                        if self.custom_band_input.len() > 7 {
                            self.custom_band_input.truncate(7);
                        }
                        self.custom_band_input = self.custom_band_input.to_uppercase();
                    }
                    
                    if ui.input(|i| i.key_pressed(egui::Key::Enter)) && !self.custom_band_input.is_empty() {
                        self.band = self.custom_band_input.clone();
                        self.custom_band_input.clear();
                    }
                    
                    ui.label(egui::RichText::new("Type and press Enter").small().italics().color(egui::Color32::GRAY));
                });
                ui.visuals_mut().widgets.inactive.weak_bg_fill = saved_inactive_bg;
                
                ui.add_space(15.0); ui.label("PERIOD:");
                if ui.selectable_value(&mut self.slot_period, SlotPeriod::S15, "15s").changed() || ui.selectable_value(&mut self.slot_period, SlotPeriod::S30, "30s").changed() { let _ = self.engine.cmds.send(UiCmd::SetSlotPeriod(self.slot_period)); }
                ui.add_space(15.0); ui.label("TX SLOT:");
                if ui.selectable_value(&mut self.slot_parity, SlotParity::Even, "Even").changed() || ui.selectable_value(&mut self.slot_parity, SlotParity::Odd, "Odd").changed() { let _ = self.engine.cmds.send(UiCmd::SetSlotParity(self.slot_parity)); }
                ui.visuals_mut().selection.bg_fill = saved_selection;
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("⚙").clicked() { self.settings_open = true; }
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
                            ui.heading(label);
                            
                            if i == 0 {
                                ui.label(egui::RichText::new("Auto-Level").size(10.0).color(egui::Color32::GRAY));
                            } else if i == 1 {
                                ui.label(egui::RichText::new("Fixed Output").size(10.0).color(egui::Color32::GRAY));
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
                        ui.add_sized([60.0, 20.0], egui::Label::new(egui::RichText::new("BAND").strong().color(egui::Color32::GRAY)));
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
                                ui.add_sized([60.0, 18.0], egui::Label::new(&q.band));
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
    let host = cpal::default_host();
    let mut ins = host.input_devices().map(|d| d.map(|x| x.name().unwrap_or_default()).collect()).unwrap_or(vec![]);
    let mut outs = host.output_devices().map(|d| d.map(|x| x.name().unwrap_or_default()).collect()).unwrap_or(vec![]);
    ins.sort(); ins.dedup(); outs.sort(); outs.dedup();
    (ins, outs)
}