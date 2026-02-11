// src/gui/app.rs
use cpal::traits::{DeviceTrait, HostTrait};
use eframe::egui;
use std::collections::HashMap;

use crate::engine::{EngineHandle, SlotParity, SlotPeriod, UiCmd, UiEvent};
use crate::engine::report_calc::report_from_correlation;
use crate::qso::adif::{AdifLogger, QsoRecord};

pub fn run_gui() -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 400.0])
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
    custom_band_input: String, // 🟢 NEW: Separate input field for custom band
    slot_parity: SlotParity,
    slot_period: SlotPeriod,
    saved_slot_parity: Option<SlotParity>, // 🟢 NEW: Save original slot when entering QSO
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
}

impl Msk2kEguiApp {
    fn new(engine: EngineHandle) -> Self {
        let (in_devs, out_devs) = enumerate_audio_devices();
        let adif_path = AdifLogger::default_path();
        let adif_logger = AdifLogger::new(&adif_path);
        let qso_log = adif_logger.read_all().unwrap_or_default();
        let config_path = crate::settings::default_config_path();
        let config = crate::settings::Config::load(&config_path).unwrap_or_default();

        Self {
            engine,
            my_call: "NOCALL".to_string(),
            their_call: String::new(),
            // 🟢 Band is UI-local only, defaults to "2M"
            // TODO: Add `pub band: Option<String>` to StationConfig to enable persistence
            band: "2M".to_string(),
            custom_band_input: String::new(),
            slot_parity: SlotParity::Odd,
            slot_period: SlotPeriod::S15,
            saved_slot_parity: None, // 🟢 NEW: No saved slot initially
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

                    // 🟢 Don't send Listen command - runtime already auto-starts if config is valid
                    // Just set UI state to reflect that we're listening
                    if !self.my_call.is_empty() && self.my_call != "NOCALL" && self.sel_in.is_some() {
                        self.is_listening = true;
                        self.current_state = "Listening".to_string();
                        log::info!("[UI] Auto-start detected, UI is_listening = true");
                    }
                },

                // src/gui/app.rs

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

    // 🟢 STRICT ROUTING LOGIC
    if is_cq {
        // 1. CQs go ONLY to the SPOTS Log
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
        // 2. Private Messages go ONLY to the RX Log (QSO Window)
        // This keeps your QSO window clean of random CQs.
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
                    // Grid is tracked in runtime, just ignore here for now
                    let _ = grid; 
                }
                UiEvent::TxSlotChanged { slot } => {
                    // 🟢 Save the original slot before changing (only once per QSO)
                    if self.saved_slot_parity.is_none() {
                        self.saved_slot_parity = Some(self.slot_parity);
                    }
                    // Update UI to show we're now on their slot
                    self.slot_parity = slot;
                }
                UiEvent::QsoLogged { record } => {
                    // 🟢 Update record with UI's current band value (purely local, not in runtime)
                    let mut updated_record = record;
                    updated_record.band = self.band.clone();
                    
                    let _ = self.adif_logger.log_qso(&updated_record); 
                    self.qso_log.insert(0, updated_record);
                    self.their_call = String::new(); 
                    self.in_active_qso = false;
                    self.reset_dedupe();
                    
                    // 🟢 Restore original slot after QSO ends
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
        self.drain_events();

        if self.settings_open {
            let mut close = false;
            egui::Window::new("⚙ Settings").collapsible(false).resizable(false).show(ctx, |ui| {
                ui.heading("Station Setup");
                ui.horizontal(|ui| { 
                    ui.label("My Callsign:"); 
                    
                    // 🟢 Determine max length based on grid mode
                    let max_len = if self.config.station.use_grid_in_cq { 7 } else { 10 };
                    
                    if ui.text_edit_singleline(&mut self.my_call).changed() { 
                        self.my_call = self.my_call.to_uppercase();
                        // 🟢 Enforce character limit
                        if self.my_call.len() > max_len {
                            self.my_call.truncate(max_len);
                        }
                    }
                    
                    // 🟢 Show character count
                    let count_color = if self.my_call.len() == max_len { 
                        egui::Color32::from_rgb(255, 180, 0) // Orange when at limit
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
                    // Field 1 (A-R)
                    egui::ComboBox::from_id_salt("grid_f1")
                        .selected_text(alphabet[self.config.station.grid_indices[0]].to_string())
                        .width(40.0)
                        .show_ui(ui, |ui| {
                            for (i, c) in alphabet.iter().enumerate() {
                                ui.selectable_value(&mut self.config.station.grid_indices[0], i, c.to_string());
                            }
                        });
                    // Field 2 (A-R)
                    egui::ComboBox::from_id_salt("grid_f2")
                        .selected_text(alphabet[self.config.station.grid_indices[1]].to_string())
                        .width(40.0)
                        .show_ui(ui, |ui| {
                            for (i, c) in alphabet.iter().enumerate() {
                                ui.selectable_value(&mut self.config.station.grid_indices[1], i, c.to_string());
                            }
                        });
                    // Square 1 (0-9)
                    egui::ComboBox::from_id_salt("grid_s1")
                        .selected_text(self.config.station.grid_indices[2].to_string())
                        .width(40.0)
                        .show_ui(ui, |ui| {
                            for n in 0..10 {
                                ui.selectable_value(&mut self.config.station.grid_indices[2], n, n.to_string());
                            }
                        });
                    // Square 2 (0-9)
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
                
                // 🟢 Re-validate callsign length when grid mode changes
                let max_call_len = if self.config.station.use_grid_in_cq { 7 } else { 10 };
                if self.my_call.len() > max_call_len {
                    self.my_call.truncate(max_call_len);
                }
                
                ui.add_space(5.0);
                ui.label(egui::RichText::new("Note: Grid mode limits callsign to 7 characters.").small().color(egui::Color32::GRAY));

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
                // Send device selections to runtime FIRST (they're consumed before Listen runs)
                let _ = self.engine.cmds.send(UiCmd::SetInputDevice(self.sel_in.clone()));
                let _ = self.engine.cmds.send(UiCmd::SetOutputDevice(self.sel_out.clone()));
                
                // 🟢 Listen handles: ApplyAudio + save config + restart RX — all in one.
                // Do NOT send ApplyAudio separately, it causes a double-restart race
                // that kills the receiver immediately after it starts.
                let _ = self.engine.cmds.send(UiCmd::Listen {
                    my_call: self.my_call.clone(),
                    their_call: String::new(),
                    auto_slots: true,
                });
                self.is_listening = true;
                self.is_calling_cq = false;
                self.current_state = "Listening".to_string();
                
                // Save UI-side config to disk (grid settings etc.)
                let _ = self.config.save(&crate::settings::default_config_path());
                self.settings_open = false; 
            }
        }

        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(&self.my_call).strong().size(22.0).color(egui::Color32::GRAY));
                ui.add_space(20.0); ui.label("BAND:");
                
                // 🟢 Band selector with Custom option
                let display_band = self.band.clone(); // Just show the band value directly
                let saved_selection = ui.visuals().selection.bg_fill;
                let saved_inactive_bg = ui.visuals().widgets.inactive.weak_bg_fill;
                let slate = egui::Color32::from_rgb(70, 90, 110);
                ui.visuals_mut().selection.bg_fill = slate;
                ui.visuals_mut().widgets.inactive.weak_bg_fill = slate;
                egui::ComboBox::from_id_salt("band").selected_text(&display_band).width(100.0).show_ui(ui, |ui| {
                    for b in &["6M", "4M", "2M", "70CM"] { 
                        // 🟢 Band is UI-only, just update local state
                        // TODO: Auto-save once StationConfig.band exists
                        ui.selectable_value(&mut self.band, b.to_string(), *b);
                    }
                    ui.separator();
                    
                    // 🟢 Custom input
                    ui.label("Custom:");
                    let text_edit = egui::TextEdit::singleline(&mut self.custom_band_input)
                        .hint_text("e.g. 144.350")
                        .desired_width(80.0);
                    
                    let response = ui.add(text_edit);
                    
                    // Automatically give focus to keep dropdown open
                    response.request_focus();
                    
                    // When user types
                    if response.changed() {
                        // Limit to 7 characters and uppercase
                        if self.custom_band_input.len() > 7 {
                            self.custom_band_input.truncate(7);
                        }
                        self.custom_band_input = self.custom_band_input.to_uppercase();
                    }
                    
                    // Check for Enter key press
                    if ui.input(|i| i.key_pressed(egui::Key::Enter)) && !self.custom_band_input.is_empty() {
                        self.band = self.custom_band_input.clone();
                        self.custom_band_input.clear();
                        // 🟢 TODO: Auto-save once StationConfig.band exists
                        // self.config.station.band = Some(self.band.clone());
                        // let _ = self.config.save(&crate::settings::default_config_path());
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
                        // 🟢 Stop listening - toggle OFF
                        log::info!("[UI] LISTEN button clicked - STOPPING");
                        self.is_listening = false;
                        self.is_calling_cq = false;
                        let _ = self.engine.cmds.send(UiCmd::Stop);
                    } else {
                        // 🟢 Start listening - toggle ON
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
                
                // 🟢 FIXED: "CALL CQ" now respects the Grid Mode toggle
                if ui.add_sized([90.0, 30.0], egui::Button::new("📢 CALL CQ").fill(if self.is_calling_cq { blue } else { egui::Color32::from_rgb(45, 45, 45) })).clicked() {
                    if self.is_calling_cq { 
                        self.is_calling_cq = false; 
                        let _ = self.engine.cmds.send(UiCmd::Stop); 
                    } else { 
                        self.is_listening = false; 
                        self.is_calling_cq = true; 
                        self.their_call = String::new(); 
                        
                        // Check if user enabled Grid Mode in settings
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
                    // 🟢 TARGET callsign always limited to 10 characters
                    if ui.text_edit_singleline(&mut self.their_call).changed() {
                        self.their_call = self.their_call.to_uppercase();
                        if self.their_call.len() > 10 {
                            self.their_call.truncate(10);
                        }
                    }
                    
                    // 🟢 Show character count
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
            let mut clicked_grid: Option<String> = None; // 🟢 NEW: Store grid if CQ has one
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
                                // 🟢 TX column (i==1) - compact background, right-aligned text
                                if i == 1 {
                                    // Allocate space for the row
                                    ui.allocate_ui(egui::vec2(ui.available_width(), 18.0), |ui| {
                                        // Right-align the content
                                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                            // Show text with compact background (only around text, not full width)
                                            if entry.colored {
                                                egui::Frame::none()
                                                    .fill(egui::Color32::from_rgba_unmultiplied(31, 111, 235, 6))
                                                    .inner_margin(egui::Margin::symmetric(6.0, 2.0))
                                                    .show(ui, |ui| {
                                                        ui.monospace(&entry.text);
                                                    });
                                                // Draw right-side bar indicator
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
                                    // RX and SPOTS columns - clickable with full-width allocation
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
                                                    // 🟢 Extract both call and grid
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
                    grid: clicked_grid, // 🟢 NEW: Pass grid if found
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
                        // Fixed widths for each column
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

// src/gui/app.rs

fn extract_callsign_and_grid(text: &str) -> Option<(String, Option<String>)> {
    let t = text.to_uppercase();
    
    // Remove count suffix like "(181)" first
    let clean_text = if let Some(pos) = t.find('(') {
        t[..pos].trim()
    } else {
        t.trim()
    };
    
    let parts: Vec<&str> = clean_text.split_whitespace().collect();
    
    // 1. "CQ DE CALL" or "CQ DE CALL GRID" format - skip the "DE" token
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
    
    // 2. "CQ CALL GRID" format (no DE - e.g. grid CQ: "CQ GW4WND IO82")
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
    
    // 3. "CQ CALL" format (2 tokens only, no grid)
    if parts.len() == 2 && parts[0] == "CQ" {
        return Some((parts[1].to_string(), None));
    }
    
    // 4. "A de B" format (directed messages, no grid)
    if clean_text.contains(" DE ") { 
        let call = clean_text.split(" DE ").nth(1)?.split_whitespace().next().map(|s| s.to_string())?;
        return Some((call, None));
    }
    
    // 5. "CALL GRID" format without CQ (e.g. from accumulated decode)
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

// 🟢 Keep legacy function for compatibility - just wraps new function
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