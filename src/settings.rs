//! Application configuration

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// MSK2K application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Audio configuration
    pub audio: AudioConfig,

    /// Station configuration
    pub station: StationConfig,

    /// UI configuration
    pub ui: UiConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Input device name (None = default)
    pub input_device: Option<String>,

    /// Output device name (None = default)
    pub output_device: Option<String>,

    /// Sample rate (fixed at 48kHz for MSK2K)
    pub sample_rate: u32,

    /// Buffer size in samples
    pub buffer_size: usize,

    /// Input audio level (0.0 - 1.0)
    pub input_level: f32,

    /// Output audio level (0.0 - 1.0)
    pub output_level: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationConfig {
    /// Station callsign
    pub callsign: String,
    /// Station grid square (e.g., "IO91")
    pub grid: Option<String>,
    pub grid_indices: [usize; 4], // Stores Field1, Field2, Sq1, Sq2 indices
    pub use_grid_in_cq: bool,     // Toggle for MSK2K Grid mode
    /// Auto-reply to CQ
    pub auto_reply_cq: bool,
    /// Auto-send 73 after exchange
    pub auto_73: bool,
    /// Band selection (e.g., "2M", "70CM", "144.350")
    pub band: Option<String>,     // 🟢 ADD THIS FIELD
}

impl Default for StationConfig {
    fn default() -> Self {
        Self {
            callsign: String::new(),
            grid: None,
            grid_indices: [9, 14, 5, 4], // IO54
            use_grid_in_cq: false,
            auto_reply_cq: false,
            auto_73: false,
            band: Some("2M".to_string()), // 🟢 ADD THIS LINE
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    /// Show waterfall display
    pub show_waterfall: bool,

    /// Waterfall height in lines
    pub waterfall_height: u16,

    /// Show decode statistics
    pub show_stats: bool,

    /// Maximum decoded messages to display
    pub max_messages: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            audio: AudioConfig::default(),
            station: StationConfig::default(),
            ui: UiConfig::default(),
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            input_device: None,
            output_device: None,
            sample_rate: 48000,
            buffer_size: 1024,
            input_level: 0.8,
            output_level: 0.8,
        }
    }
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            show_waterfall: true,
            waterfall_height: 10,
            show_stats: true,
            max_messages: 100,
        }
    }
}

impl Config {
    /// Load configuration from file
    pub fn load(path: &PathBuf) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn save(&self, path: &PathBuf) -> anyhow::Result<()> {
        let contents = toml::to_string_pretty(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Get default config file path
    pub fn default_path() -> PathBuf {
        let mut path = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        path.push(".msk2k");
        path.push("config.toml");
        path
    }

    /// Create default config directory
    pub fn create_config_dir() -> anyhow::Result<PathBuf> {
        let mut path = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        path.push(".msk2k");
        std::fs::create_dir_all(&path)?;
        Ok(path)
    }

    /// Load config if present; otherwise default.
    /// Always applies environment overrides afterwards.
    pub fn load_or_default_with_env() -> Self {
        let path = Self::default_path();
        let mut cfg = match Self::load(&path) {
            Ok(c) => c,
            Err(_) => Self::default(),
        };
        cfg.apply_env_overrides();
        cfg
    }

    /// Apply environment overrides (non-destructive; only overrides if env var is present).
    ///
    /// Supported:
    /// - MSK2K_CALLSIGN
    /// - MSK2K_RX_IN   (audio input device name or substring)
    /// - MSK2K_TX_OUT  (audio output device name or substring)
    pub fn apply_env_overrides(&mut self) {
        if let Ok(cs) = std::env::var("MSK2K_CALLSIGN") {
            let cs = cs.trim();
            if !cs.is_empty() {
                self.station.callsign = cs.to_uppercase();
            }
        }

        if let Ok(rx) = std::env::var("MSK2K_RX_IN") {
            let rx = rx.trim();
            if !rx.is_empty() {
                self.audio.input_device = Some(rx.to_string());
            }
        }

        if let Ok(tx) = std::env::var("MSK2K_TX_OUT") {
            let tx = tx.trim();
            if !tx.is_empty() {
                self.audio.output_device = Some(tx.to_string());
            }
        }
    }
}
pub fn default_config_path() -> std::path::PathBuf {
    let mut path = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    path.push(".msk2k");
    std::fs::create_dir_all(&path).ok();
    path.push("config.toml");
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.audio.sample_rate, 48000);
        assert_eq!(config.station.callsign, "N0CALL");
    }

    #[test]
    fn test_serialize_config() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        assert!(toml_str.contains("sample_rate"));
    }
}