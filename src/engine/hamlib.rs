// src/engine/hamlib.rs

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use std::time::Duration;
use log::{info, error, debug};
#[allow(dead_code)]
#[derive(Debug)]
pub enum HamlibCmd {
    Ptt(bool),
    GetFreq,
    GetMode,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct HamlibUpdate {
    pub freq: Option<u64>,
    pub mode: Option<String>,
    pub ptt_active: bool,
}

pub struct HamlibClient {
    cmd_tx: mpsc::UnboundedSender<HamlibCmd>,
}

impl HamlibClient {
    pub fn new(address: String, update_tx: mpsc::UnboundedSender<HamlibUpdate>) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();

        tokio::spawn(async move {
            run_hamlib_client(address, cmd_rx, update_tx).await;
        });

        Self { cmd_tx }
    }

    pub fn set_ptt(&self, active: bool) {
        let _ = self.cmd_tx.send(HamlibCmd::Ptt(active));
    }

    pub fn refresh(&self) {
        let _ = self.cmd_tx.send(HamlibCmd::GetFreq);
    }
}

async fn run_hamlib_client(
    addr: String,
    mut cmd_rx: mpsc::UnboundedReceiver<HamlibCmd>,
    update_tx: mpsc::UnboundedSender<HamlibUpdate>,
) {
    loop {
        match TcpStream::connect(&addr).await {
            Ok(mut stream) => {
                info!("[Hamlib] Connected to rigctld at {}!", addr);
                let (reader, mut writer) = stream.split();
                let mut reader = BufReader::new(reader);
                let mut buf = String::new();

                if let Err(_) = writer.write_all(b"f\n").await { continue; }

                loop {
                    tokio::select! {
                        cmd_opt = cmd_rx.recv() => {
                            let cmd = match cmd_opt {
                                Some(c) => c,
                                None => {
                                    info!("[Hamlib] Command channel closed, shutting down");
                                    return; // Exit the entire task
                                }
                            };

                            let cmd_str = match cmd {
                                HamlibCmd::Ptt(true) => "T 1\n",
                                HamlibCmd::Ptt(false) => "T 0\n",
                                HamlibCmd::GetFreq => "f\n",
                                HamlibCmd::GetMode => "m\n",
                            };
                            
                            debug!("[Hamlib] TX: {:?}", cmd_str.trim());
                            if let Err(e) = writer.write_all(cmd_str.as_bytes()).await {
                                error!("[Hamlib] Write error: {}", e);
                                break; // Break inner loop to trigger reconnect
                            }
                            
                            if matches!(cmd, HamlibCmd::Ptt(_)) {
                                buf.clear();
                                let _ = reader.read_line(&mut buf).await;
                            } 
                            else if matches!(cmd, HamlibCmd::GetFreq) {
                                buf.clear();
                                if reader.read_line(&mut buf).await.is_ok() {
                                    if let Ok(freq) = buf.trim().parse::<u64>() {
                                        let _ = update_tx.send(HamlibUpdate { freq: Some(freq), mode: None, ptt_active: false });
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Err(_) => {
                // Check if channel is closed before retrying
                if cmd_rx.is_closed() {
                    info!("[Hamlib] Command channel closed, shutting down");
                    return;
                }
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    }
}