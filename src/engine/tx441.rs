// src/engine/tx441.rs
//
// FSK441 TX engine — manages the 30-second TX/RX period, audio output,
// and PTT keying via hamlib/rigctld.
//
// Period structure (30s periods):
//   Period 1 (TX first):  TX 00-29s, RX 30-59s  (western station)
//   Period 2 (RX first):  RX 00-29s, TX 30-59s  (eastern station)

use std::time::Duration;
use chrono::Utc;
use tokio::sync::mpsc;
use anyhow::Result;

use crate::fsk441rx::gen::{encode_message, generate_audio};
use crate::fsk441rx::params::SAMPLE_RATE_F;

// ─── Period ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TxPeriod {
    First,   // TX 00-29s
    Second,  // TX 30-59s
}

impl TxPeriod {
    pub fn is_tx_time(&self) -> bool {
        let secs = Utc::now().timestamp() % 60;
        match self {
            TxPeriod::First  => secs < 30,
            TxPeriod::Second => secs >= 30,
        }
    }

    pub fn secs_remaining_in_window(&self) -> i64 {
        let secs = Utc::now().timestamp() % 60;
        match self {
            TxPeriod::First  => if secs < 30  { 29 - secs } else { 0 },
            TxPeriod::Second => if secs >= 30 { 59 - secs } else { 0 },
        }
    }

    pub fn secs_to_next_tx(&self) -> i64 {
        let secs = Utc::now().timestamp() % 60;
        match self {
            TxPeriod::First  => if secs >= 30 { 60 - secs } else { 0 },
            TxPeriod::Second => if secs < 30  { 30 - secs } else { 0 },
        }
    }

    pub fn label(&self) -> &str {
        match self {
            TxPeriod::First  => "1st (TX 00-29s)",
            TxPeriod::Second => "2nd (TX 30-59s)",
        }
    }
}

// ─── Commands / Events ───────────────────────────────────────────────────────

#[derive(Debug)]
pub enum TxCommand {
    /// Queue a message to start at the next TX window
    StartTx { message: String, period: TxPeriod },
    /// Stop after current pass completes
    StopTx,
    /// Stop immediately — mid-pass if necessary
    HaltTx,
    /// Swap message at next pass boundary
    UpdateMessage(String),
}

#[derive(Debug, Clone)]
pub enum TxEvent {
    TxWindowOpen,
    PassCompleted { pass_number: u32, message: String },
    TxWindowClosed,
    PttError(String),
}

// ─── Engine ───────────────────────────────────────────────────────────────────

pub struct TxEngine {
    /// rigctld address e.g. "localhost:4532"
    rigctld_addr: Option<String>,
    output_device: Option<String>,
}

impl TxEngine {
    pub fn new(rigctld_addr: Option<String>, output_device: Option<String>) -> Self {
        Self { rigctld_addr, output_device }
    }

    pub async fn run(
        self,
        mut cmd_rx: mpsc::UnboundedReceiver<TxCommand>,
        event_tx:   mpsc::UnboundedSender<TxEvent>,
    ) {
        let mut queued_message: Option<(String, TxPeriod)> = None;
        let mut halt_requested = false;

        loop {
            // Drain commands
            while let Ok(cmd) = cmd_rx.try_recv() {
                match cmd {
                    TxCommand::StartTx { message, period } => {
                        queued_message = Some((message, period));
                        halt_requested = false;
                        log::info!("[TX] Message queued for {} period", period.label());
                    }
                    TxCommand::StopTx  => { halt_requested = true; }
                    TxCommand::HaltTx  => {
                        halt_requested = true;
                        queued_message = None;
                        self.ptt_off().await;
                        let _ = event_tx.send(TxEvent::TxWindowClosed);
                    }
                    TxCommand::UpdateMessage(msg) => {
                        if let Some((_, p)) = queued_message.take() {
                            queued_message = Some((msg, p));
                        }
                    }
                }
            }

            // Check if it's time to TX
            if let Some((ref msg, period)) = queued_message {
                if period.is_tx_time() && !halt_requested {
                    let msg_clone   = msg.clone();
                    let dev_clone   = self.output_device.clone();
                    let event_clone = event_tx.clone();
                    let addr_clone  = self.rigctld_addr.clone();

                    // Key PTT
                    if let Err(e) = self.ptt_on().await {
                        log::error!("[TX] PTT on failed: {}", e);
                        let _ = event_tx.send(TxEvent::PttError(e.to_string()));
                        queued_message = None;
                        tokio::time::sleep(Duration::from_millis(200)).await;
                        continue;
                    }
                    let _ = event_tx.send(TxEvent::TxWindowOpen);

                    // Stream audio on blocking thread
                    tokio::task::spawn_blocking(move || {
                        stream_audio(&msg_clone, &dev_clone, period, &event_clone);
                    }).await.ok();

                    // Un-key PTT
                    self.ptt_off().await;
                    let _ = event_tx.send(TxEvent::TxWindowClosed);

                    if halt_requested {
                        queued_message = None;
                        halt_requested = false;
                    }
                }
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn ptt_on(&self) -> Result<()> {
        if let Some(ref addr) = self.rigctld_addr {
            use tokio::net::TcpStream;
            use tokio::io::AsyncWriteExt;
            let mut s = TcpStream::connect(addr).await?;
            s.write_all(b"T 1\n").await?;
            log::debug!("[TX] PTT ON via rigctld {}", addr);
        } else {
            log::debug!("[TX] No rigctld — PTT via VOX or manual");
        }
        Ok(())
    }

    async fn ptt_off(&self) {
        if let Some(ref addr) = self.rigctld_addr {
            use tokio::net::TcpStream;
            use tokio::io::AsyncWriteExt;
            if let Ok(mut s) = TcpStream::connect(addr).await {
                let _ = s.write_all(b"T 0\n").await;
                log::debug!("[TX] PTT OFF via rigctld {}", addr);
            }
        }
    }
}

// ─── Blocking audio output ────────────────────────────────────────────────────

fn stream_audio(
    message:  &str,
    device:   &Option<String>,
    period:   TxPeriod,
    event_tx: &mpsc::UnboundedSender<TxEvent>,
) {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use std::sync::{Arc, Mutex};

    let host = cpal::default_host();

    let output = match device {
        Some(name) => host.output_devices().ok()
            .and_then(|mut d| d.find(|dev|
                dev.name().ok().map(|n| n.contains(name)).unwrap_or(false)))
            .or_else(|| host.default_output_device()),
        None => host.default_output_device(),
    };

    let output = match output {
        Some(d) => d,
        None    => { log::error!("[TX] No audio output device"); return; }
    };

    let config = match output.default_output_config() {
        Ok(c)  => c,
        Err(e) => { log::error!("[TX] Output config error: {}", e); return; }
    };

    // Pre-generate one pass of audio, resampled if needed
    let tones    = encode_message(message);
    let mut buf  = generate_audio(&tones);

    // Resample from 11025 to device rate if needed
    let dev_rate = config.sample_rate().0;
    if dev_rate != crate::fsk441rx::params::SAMPLE_RATE {
        buf = resample_audio(&buf, crate::fsk441rx::params::SAMPLE_RATE, dev_rate);
    }

    let pass_len = buf.len();
    let buf      = Arc::new(buf);
    let pos      = Arc::new(Mutex::new(0usize));
    let pass_cnt = Arc::new(Mutex::new(0u32));

    let buf2     = buf.clone();
    let pos2     = pos.clone();
    let cnt2     = pass_cnt.clone();
    let evt2     = event_tx.clone();
    let msg2     = message.to_string();

    let stream = match output.build_output_stream(
        &config.into(),
        move |data: &mut [f32], _| {
            let mut p = pos2.lock().unwrap();
            for s in data.iter_mut() {
                *s = buf2[*p];
                *p += 1;
                if *p >= pass_len {
                    *p = 0;
                    let mut n = cnt2.lock().unwrap();
                    *n += 1;
                    let _ = evt2.send(TxEvent::PassCompleted {
                        pass_number: *n,
                        message: msg2.clone(),
                    });
                }
            }
        },
        |e| log::error!("[TX] Stream error: {}", e),
        None,
    ) {
        Ok(s)  => s,
        Err(e) => { log::error!("[TX] Build stream error: {}", e); return; }
    };

    if let Err(e) = stream.play() {
        log::error!("[TX] Play error: {}", e);
        return;
    }

    // Run until the TX window closes
    while period.is_tx_time() {
        std::thread::sleep(Duration::from_millis(50));
    }
    // stream drops here, audio stops
}

/// Simple linear interpolation resampler for TX audio
/// (quality not critical for FSK — we just need the right rate)
fn resample_audio(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    if in_rate == out_rate { return input.to_vec(); }
    let ratio  = out_rate as f64 / in_rate as f64;
    let n_out  = (input.len() as f64 * ratio) as usize;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let src = i as f64 / ratio;
        let lo  = src.floor() as usize;
        let hi  = (lo + 1).min(input.len() - 1);
        let frac = src - lo as f64;
        out.push(input[lo] * (1.0 - frac as f32) + input[hi] * frac as f32);
    }
    out
}
