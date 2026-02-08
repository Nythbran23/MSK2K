//! Terminal User Interface

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Terminal,
};
use std::io;
use tokio::sync::mpsc;

use crate::rx::DecodedMessage;

/// UI Events
#[derive(Debug, Clone)]
pub enum UiEvent {
    Quit,
    ToggleTx,
    SendCq,
    SendMessage(String),
}

/// Terminal UI state
pub struct Ui {
    messages: Vec<DecodedMessage>,
    input_buffer: String,
    tx_active: bool,
}

impl Ui {
    /// Create a new UI
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            input_buffer: String::new(),
            tx_active: false,
        }
    }

    /// Run the UI
    pub async fn run(
        &mut self,
        mut decoded_rx: mpsc::UnboundedReceiver<DecodedMessage>,
        event_tx: mpsc::UnboundedSender<UiEvent>,
    ) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        stdout.execute(EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        loop {
            // Handle decoded messages
            while let Ok(msg) = decoded_rx.try_recv() {
                self.messages.push(msg);
                if self.messages.len() > 100 {
                    self.messages.remove(0);
                }
            }

            // Draw UI
            terminal.draw(|f| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Min(3),    // Messages
                        Constraint::Length(3), // Input
                        Constraint::Length(3), // Status
                    ])
                    .split(f.area());

                // Messages window
                let messages: Vec<ListItem> = self
                    .messages
                    .iter()
                    .rev()
                    .take(20)
                    .map(|m| {
                        let time = m.timestamp.format("%H:%M:%S");
                        let content = format!(
                            "{} | SNR: {:.1}dB | {} -> {}: {}",
                            time,
                            m.snr,
                            m.from_call,
                            m.to_call.as_deref().unwrap_or("ALL"),
                            m.text
                        );
                        ListItem::new(content)
                    })
                    .collect();

                let messages_list = List::new(messages).block(
                    Block::default()
                        .title("Decoded Messages")
                        .borders(Borders::ALL),
                );

                f.render_widget(messages_list, chunks[0]);

                // Input window
                let input_text = format!("> {}", self.input_buffer);
                let input = Paragraph::new(input_text)
                    .block(Block::default().title("Input").borders(Borders::ALL));

                f.render_widget(input, chunks[1]);

                // Status bar
                let status_text = if self.tx_active {
                    "TX | Press ESC to stop | F1: CQ | F2: Toggle TX/RX"
                } else {
                    "RX | Press ESC to quit | F1: CQ | F2: Toggle TX/RX"
                };

                let status = Paragraph::new(status_text)
                    .style(Style::default().fg(if self.tx_active {
                        Color::Red
                    } else {
                        Color::Green
                    }))
                    .block(Block::default().borders(Borders::ALL));

                f.render_widget(status, chunks[2]);
            })?;

            // Handle keyboard input
            if event::poll(std::time::Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Esc => {
                            event_tx.send(UiEvent::Quit)?;
                            break;
                        }
                        KeyCode::F(1) => {
                            event_tx.send(UiEvent::SendCq)?;
                        }
                        KeyCode::F(2) => {
                            event_tx.send(UiEvent::ToggleTx)?;
                            self.tx_active = !self.tx_active;
                        }
                        KeyCode::Char(c) => {
                            self.input_buffer.push(c);
                        }
                        KeyCode::Backspace => {
                            self.input_buffer.pop();
                        }
                        KeyCode::Enter => {
                            if !self.input_buffer.is_empty() {
                                event_tx.send(UiEvent::SendMessage(self.input_buffer.clone()))?;
                                self.input_buffer.clear();
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Cleanup terminal
        disable_raw_mode()?;
        terminal.backend_mut().execute(LeaveAlternateScreen)?;

        Ok(())
    }
}

impl Default for Ui {
    fn default() -> Self {
        Self::new()
    }
}
