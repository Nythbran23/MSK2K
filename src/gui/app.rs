// src/gui/app.rs
use cpal::traits::{DeviceTrait, HostTrait};
use eframe::egui;
use std::collections::HashMap;

use crate::engine::{EngineHandle, SlotParity, SlotPeriod, UiCmd, UiEvent};
use crate::engine::report_calc::report_from_correlation;
use crate::qso::adif::{AdifLogger, QsoRecord};

pub fn run_gui() -> anyhow::Result<()> {
    let options = eframe::NativeOptions::default();
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
}

struct Msk2kEguiApp {
    engine: EngineHandle,
    my_call: String,
    their_call: String,
    band: String,
    slot_parity: SlotParity,
    slot_period: SlotPeriod,
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
}

impl Msk2kEguiApp {
    fn new(engine: EngineHandle) -> Self {
        let (in_devs, out_devs) = enumerate_audio_devices();
        let adif_path = AdifLogger::default_path();
        let adif_logger = AdifLogger::new(&adif_path);
        let qso_log = adif_logger.read_all().unwrap_or_default();

        Self {
            engine,
            my_call: "NOCALL".to_string(),
            their_call: String::new(),
            band: "2M".to_string(),
            slot_parity: SlotParity::Odd,
            slot_period: SlotPeriod::S15,
            in_devs,
            out_devs,
            sel_in: None,
            sel_out: None,
            settings_open: false,
            is_listening: false, // Default off, waits for ConfigLoaded
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
                // 🟢 NEW: Handle Config Loading
                UiEvent::ConfigLoaded { my_call, input_device, output_device } => {
                    // 1. Populate Fields
                    if !my_call.is_empty() { self.my_call = my_call; }
                    
                    if let Some(in_d) = input_device {
                        if self.in_devs.contains(&in_d) { self.sel_in = Some(in_d); }
                    }
                    if let Some(out_d) = output_device {
                        if self.out_devs.contains(&out_d) { self.sel_out = Some(out_d); }
                    }

                    // 2. Sync Listen State (Auto-Start)
                    // If we have a callsign and valid input, the Engine has already auto-started.
                    if !self.my_call.is_empty() && self.my_call != "NOCALL" && self.sel_in.is_some() {
                        self.is_listening = true;
                        self.current_state = "Listening".to_string();
                    }
                },

                UiEvent::RxText { text, snr, utc_ms: _, rx_slot } => {
                    self.last_corr = snr.unwrap_or(self.last_corr);
                    self.last_rx_slot = Some(rx_slot);
                    let key = text.to_uppercase().trim().to_string();
                    let count = self.decode_counts.entry(key.clone()).or_insert(0);
                    *count += 1;
                    let display = format!("{} ({})", text, count);
                    let stamp = self.in_active_qso || !self.their_call.is_empty();
                    if *count == 1 {
                        let _ = push_cap_entry(&mut self.rx_log, LogEntry { text: display.clone(), colored: stamp });
                        self.decode_log_index.insert(key.clone(), self.rx_log.len().saturating_sub(1));
                        if text.to_uppercase().contains("CQ") {
                            let _ = push_cap_entry(&mut self.cq_log, LogEntry { text: display, colored: stamp });
                            self.cq_log_index.insert(key, self.cq_log.len().saturating_sub(1));
                        }
                    } else if let Some(&idx) = self.decode_log_index.get(&key) {
                        if idx < self.rx_log.len() { self.rx_log[idx].text = display; if stamp { self.rx_log[idx].colored = true; } }
                    }
                }
                UiEvent::TxText { text } => { push_cap_entry(&mut self.tx_log, LogEntry { text, colored: self.in_active_qso || !self.their_call.is_empty() }); }
                UiEvent::State(s) => {
                    self.current_state = s.clone();
                    if s.contains("Listening") { self.is_listening = true; }
                    else if s.contains("CallingCq") { self.is_listening = false; self.is_calling_cq = true; }
                    else if s.contains("Sending") || s.contains("CallingStn") { self.is_listening = false; self.is_calling_cq = false; self.in_active_qso = true; }
                }
                UiEvent::TheirCallChanged { callsign } => { self.their_call = callsign.clone(); if !callsign.is_empty() { self.color_history_for_call(&callsign); } }
                UiEvent::QsoLogged { record } => {
                    let _ = self.adif_logger.log_qso(&record); self.qso_log.insert(0, record);
                    self.their_call = String::new(); self.in_active_qso = false;
                    self.reset_dedupe();
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
                ui.horizontal(|ui| { ui.label("My Callsign:"); if ui.text_edit_singleline(&mut self.my_call).changed() { self.my_call = self.my_call.to_uppercase(); } });
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
                if ui.button("Apply & Close").clicked() { close = true; }
            });
            if close { self.send_apply_audio(); self.settings_open = false; }
        }

        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(&self.my_call).strong().size(22.0).color(egui::Color32::GRAY));
                ui.add_space(20.0); ui.label("BAND:");
                egui::ComboBox::from_id_salt("band").selected_text(&self.band).width(70.0).show_ui(ui, |ui| {
                    for b in &["6M", "4M", "2M", "70CM"] { if ui.selectable_value(&mut self.band, b.to_string(), *b).changed() { let _ = self.engine.cmds.send(UiCmd::SetBand(self.band.clone())); } }
                });
                ui.add_space(15.0); ui.label("PERIOD:");
                if ui.selectable_value(&mut self.slot_period, SlotPeriod::S15, "15s").changed() || ui.selectable_value(&mut self.slot_period, SlotPeriod::S30, "30s").changed() { let _ = self.engine.cmds.send(UiCmd::SetSlotPeriod(self.slot_period)); }
                ui.add_space(15.0); ui.label("TX SLOT:");
                if ui.selectable_value(&mut self.slot_parity, SlotParity::Even, "Even").changed() || ui.selectable_value(&mut self.slot_parity, SlotParity::Odd, "Odd").changed() { let _ = self.engine.cmds.send(UiCmd::SetSlotParity(self.slot_parity)); }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("⚙").clicked() { self.settings_open = true; }
                    ui.label(egui::RichText::new(chrono::Utc::now().format("%H:%M:%S Z").to_string()).monospace());
                });
            });
        });

        egui::TopBottomPanel::top("actions").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let blue = egui::Color32::from_rgb(31, 111, 235);
                if ui.add_sized([90.0, 30.0], egui::Button::new("📻 LISTEN").fill(if self.is_listening { blue } else { egui::Color32::from_rgb(45, 45, 45) })).clicked() {
                    self.is_listening = true; self.is_calling_cq = false; self.their_call = String::new();
                    let _ = self.engine.cmds.send(UiCmd::Listen { my_call: self.my_call.clone(), their_call: String::new(), auto_slots: true });
                }
                if ui.add_sized([90.0, 30.0], egui::Button::new("📢 CALL CQ").fill(if self.is_calling_cq { blue } else { egui::Color32::from_rgb(45, 45, 45) })).clicked() {
                    if self.is_calling_cq { self.is_calling_cq = false; let _ = self.engine.cmds.send(UiCmd::Stop); }
                    else { self.is_listening = false; self.is_calling_cq = true; self.their_call = String::new(); let _ = self.engine.cmds.send(UiCmd::StartCq { my_call: self.my_call.clone(), auto_slots: true }); }
                }
                ui.add_space(20.0); ui.label("TARGET:");
                ui.text_edit_singleline(&mut self.their_call);
                if ui.button("CALL").clicked() {
                    let t = self.their_call.clone(); self.color_history_for_call(&t);
                    let _ = self.engine.cmds.send(UiCmd::CallStation { my_call: self.my_call.clone(), their_call: t });
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("⏹ STOP").clicked() { self.their_call = String::new(); self.is_calling_cq = false; self.is_listening = false; let _ = self.engine.cmds.send(UiCmd::Stop); }
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let mut clicked_call = None;
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
                        egui::ScrollArea::vertical().id_salt(id).stick_to_bottom(true).show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            let log = match i { 0 => &self.rx_log, 1 => &self.tx_log, _ => &self.cq_log };
                            for entry in log {
                                ui.allocate_ui(egui::vec2(ui.available_width(), 18.0), |ui| {
                                    let rect = ui.available_rect_before_wrap();
                                    let color = if entry.colored { 
                                        if i == 0 { egui::Color32::from_rgba_unmultiplied(31, 143, 58, 6) } 
                                        else { egui::Color32::from_rgba_unmultiplied(31, 111, 235, 6) } 
                                    } else { egui::Color32::TRANSPARENT };
                                    
                                    egui::Frame::none().fill(color).inner_margin(egui::Margin::symmetric(10.0, 2.0)).show(ui, |ui| {
                                        if entry.colored {
                                            let bar_rect = if i == 0 { egui::Rect::from_min_max(egui::pos2(rect.min.x, rect.min.y), egui::pos2(rect.min.x + 3.0, rect.max.y)) } 
                                                           else { egui::Rect::from_min_max(egui::pos2(rect.max.x - 3.0, rect.min.y), egui::pos2(rect.max.x, rect.max.y)) };
                                            ui.painter().rect_filled(bar_rect, 0.0, if i == 0 { egui::Color32::from_rgb(31, 143, 58) } else { egui::Color32::from_rgb(31, 111, 235) });
                                        }
                                        if i == 1 { ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| { ui.monospace(&entry.text); }); }
                                        else if ui.selectable_label(false, &entry.text).clicked() { if let Some(c) = extract_callsign(&entry.text) { clicked_call = Some(c); } }
                                    });
                                });
                            }
                        });
                    });
                }
            });

            if let Some(target) = clicked_call { 
                self.their_call = target.clone(); self.color_history_for_call(&target);
                let _ = self.engine.cmds.send(UiCmd::AnswerCq { my_call: self.my_call.clone(), their_call: target, rpt: self.calc_report(), rx_slot: self.last_rx_slot }); 
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
                    .num_columns(7)
                    .spacing([10.0, 4.0])
                    .min_col_width(ui.available_width() / 7.2)
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("DATE").strong().color(egui::Color32::GRAY));
                        ui.label(egui::RichText::new("START (Z)").strong().color(egui::Color32::GRAY));
                        ui.label(egui::RichText::new("END (Z)").strong().color(egui::Color32::GRAY));
                        ui.label(egui::RichText::new("BAND").strong().color(egui::Color32::GRAY));
                        ui.label(egui::RichText::new("STATION").strong().color(egui::Color32::GRAY));
                        ui.label(egui::RichText::new("SENT").strong().color(egui::Color32::GRAY));
                        ui.label(egui::RichText::new("RCVD").strong().color(egui::Color32::GRAY));
                        ui.end_row();
                    });

                ui.add_space(2.0);
                ui.separator();

                egui::ScrollArea::vertical().id_salt("log_sc_data").max_height(200.0).show(ui, |ui| {
                    egui::Grid::new("log_data_grid")
                        .striped(true)
                        .num_columns(7)
                        .spacing([10.0, 4.0])
                        .min_col_width(ui.available_width() / 7.2)
                        .show(ui, |ui| {
                            for q in &self.qso_log {
                                ui.label(q.display_date());
                                ui.label(q.display_time_on());
                                ui.label(q.display_time_off());
                                ui.label(&q.band);
                                ui.label(egui::RichText::new(&q.call).strong().color(egui::Color32::LIGHT_BLUE));
                                ui.label(&q.rst_sent);
                                ui.label(&q.rst_rcvd);
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

fn extract_callsign(text: &str) -> Option<String> {
    let t = text.to_uppercase();
    if t.contains(" DE ") { t.split(" DE ").nth(1)?.split_whitespace().next().map(|s| s.to_string()) }
    else if t.starts_with("CQ ") { t.split_whitespace().nth(1).map(|s| s.to_string()) }
    else { None }
}

fn push_cap_entry(v: &mut Vec<LogEntry>, s: LogEntry) { if v.len() >= 100 { v.remove(0); } v.push(s); }

fn enumerate_audio_devices() -> (Vec<String>, Vec<String>) {
    let host = cpal::default_host();
    let mut ins = host.input_devices().map(|d| d.map(|x| x.name().unwrap_or_default()).collect()).unwrap_or(vec![]);
    let mut outs = host.output_devices().map(|d| d.map(|x| x.name().unwrap_or_default()).collect()).unwrap_or(vec![]);
    ins.sort(); ins.dedup(); outs.sort(); outs.dedup();
    (ins, outs)
}