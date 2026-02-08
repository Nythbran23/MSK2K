// src/qso/adif.rs
//
// ADIF (Amateur Data Interchange Format) logging for QSO records.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use chrono::{Utc, TimeZone};

/// A completed QSO record with all required fields for ADIF logging
#[derive(Debug, Clone)]
pub struct QsoRecord {
    /// Their callsign
    pub call: String,
    /// Our callsign
    pub operator: String,
    /// QSO date in YYYYMMDD format
    pub qso_date: String,
    /// Start time in HHMMSS format
    pub time_on: String,
    /// End time in HHMMSS format
    pub time_off: String,
    /// Band (e.g., "2M", "70CM")
    pub band: String,
    /// Frequency in MHz (optional)
    pub freq: Option<f64>,
    /// Mode (e.g., "MSK144")
    pub mode: String,
    /// RST sent
    pub rst_sent: String,
    /// RST received
    pub rst_rcvd: String,
}

impl QsoRecord {
    /// Create a new QSO record from timestamps and QSO data
    pub fn new(
        call: String,
        operator: String,
        start_utc_ms: i64,
        end_utc_ms: i64,
        band: String,
        freq: Option<f64>,
        rst_sent: i16,
        rst_rcvd: Option<i16>,
    ) -> Self {
        let start_dt = Utc.timestamp_millis_opt(start_utc_ms).unwrap();
        let end_dt = Utc.timestamp_millis_opt(end_utc_ms).unwrap();
        
        Self {
            call,
            operator,
            qso_date: start_dt.format("%Y%m%d").to_string(),
            time_on: start_dt.format("%H%M%S").to_string(),
            time_off: end_dt.format("%H%M%S").to_string(),
            band,
            freq,
            mode: "MSK144".to_string(),
            rst_sent: rst_sent.to_string(),
            rst_rcvd: rst_rcvd.map(|r| r.to_string()).unwrap_or_default(),
        }
    }
    
    /// Format as ADIF record string
    pub fn to_adif(&self) -> String {
        let mut parts = Vec::new();
        
        parts.push(adif_field("CALL", &self.call));
        parts.push(adif_field("OPERATOR", &self.operator));
        parts.push(adif_field("QSO_DATE", &self.qso_date));
        parts.push(adif_field("TIME_ON", &self.time_on));
        parts.push(adif_field("TIME_OFF", &self.time_off));
        parts.push(adif_field("BAND", &self.band));
        if let Some(f) = self.freq {
            parts.push(adif_field("FREQ", &format!("{:.6}", f)));
        }
        parts.push(adif_field("MODE", &self.mode));
        parts.push(adif_field("RST_SENT", &self.rst_sent));
        if !self.rst_rcvd.is_empty() {
            parts.push(adif_field("RST_RCVD", &self.rst_rcvd));
        }
        parts.push("<EOR>".to_string());
        
        parts.join(" ")
    }
    
    /// Format date for display (YYYY-MM-DD)
    pub fn display_date(&self) -> String {
        if self.qso_date.len() == 8 {
            format!("{}-{}-{}", 
                &self.qso_date[0..4], 
                &self.qso_date[4..6], 
                &self.qso_date[6..8])
        } else {
            self.qso_date.clone()
        }
    }
    
    /// Format time for display (HH:MM:SS)
    pub fn display_time_on(&self) -> String {
        format_time(&self.time_on)
    }
    
    /// Format time for display (HH:MM:SS)
    pub fn display_time_off(&self) -> String {
        format_time(&self.time_off)
    }
}

fn format_time(t: &str) -> String {
    if t.len() >= 6 {
        format!("{}:{}:{}", &t[0..2], &t[2..4], &t[4..6])
    } else if t.len() >= 4 {
        format!("{}:{}", &t[0..2], &t[2..4])
    } else {
        t.to_string()
    }
}

fn adif_field(name: &str, value: &str) -> String {
    format!("<{}:{}>{}",name, value.len(), value)
}

/// ADIF file writer - appends QSO records to a file
pub struct AdifLogger {
    path: String,
}

impl AdifLogger {
    pub fn new(path: &str) -> Self {
        Self { path: path.to_string() }
    }
    
    /// Get the default ADIF log path
    pub fn default_path() -> String {
        // Use home directory or current directory
        if let Some(home) = dirs::home_dir() {
            home.join("msk2k_log.adi").to_string_lossy().to_string()
        } else {
            "msk2k_log.adi".to_string()
        }
    }
    
    /// Append a QSO record to the ADIF file
    pub fn log_qso(&self, record: &QsoRecord) -> std::io::Result<()> {
        let path = Path::new(&self.path);
        let needs_header = !path.exists() || path.metadata().map(|m| m.len() == 0).unwrap_or(true);
        
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        
        if needs_header {
            writeln!(file, "ADIF Export from MSK2K")?;
            writeln!(file, "<ADIF_VER:5>3.1.4")?;
            writeln!(file, "<PROGRAMID:5>MSK2K")?;
            writeln!(file, "<EOH>")?;
            writeln!(file)?;
        }
        
        writeln!(file, "{}", record.to_adif())?;
        
        log::info!("📝 Logged QSO to {}: {} at {}", self.path, record.call, record.time_on);
        
        Ok(())
    }
    
    /// Read all QSO records from the ADIF file
    pub fn read_all(&self) -> std::io::Result<Vec<QsoRecord>> {
        let path = Path::new(&self.path);
        if !path.exists() {
            return Ok(Vec::new());
        }
        
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();
        
        let mut past_header = false;
        
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            
            if line.to_uppercase().contains("<EOH>") {
                past_header = true;
                continue;
            }
            
            if !past_header || line.is_empty() {
                continue;
            }
            
            if let Some(record) = parse_adif_record(line) {
                records.push(record);
            }
        }
        
        // Return in reverse order (newest first)
        records.reverse();
        Ok(records)
    }
}

/// Parse a single ADIF record line
fn parse_adif_record(line: &str) -> Option<QsoRecord> {
    let get_field = |name: &str| -> Option<String> {
        let upper = line.to_uppercase();
        let search = format!("<{}:", name.to_uppercase());
        if let Some(start) = upper.find(&search) {
            let after_name = start + search.len();
            if let Some(gt_pos) = line[after_name..].find('>') {
                let len_str = &line[after_name..after_name + gt_pos];
                if let Ok(len) = len_str.parse::<usize>() {
                    let value_start = after_name + gt_pos + 1;
                    let value_end = (value_start + len).min(line.len());
                    return Some(line[value_start..value_end].to_string());
                }
            }
        }
        None
    };
    
    let call = get_field("CALL")?;
    let qso_date = get_field("QSO_DATE").unwrap_or_default();
    let time_on = get_field("TIME_ON").unwrap_or_default();
    let time_off = get_field("TIME_OFF").unwrap_or_default();
    let band = get_field("BAND").unwrap_or_default();
    let freq = get_field("FREQ").and_then(|f| f.parse().ok());
    let mode = get_field("MODE").unwrap_or_else(|| "MSK144".to_string());
    let rst_sent = get_field("RST_SENT").unwrap_or_default();
    let rst_rcvd = get_field("RST_RCVD").unwrap_or_default();
    let operator = get_field("OPERATOR").unwrap_or_default();
    
    Some(QsoRecord {
        call,
        operator,
        qso_date,
        time_on,
        time_off,
        band,
        freq,
        mode,
        rst_sent,
        rst_rcvd,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_adif_format() {
        let record = QsoRecord {
            call: "DJ5HJ".to_string(),
            operator: "GW4WND".to_string(),
            qso_date: "20260202".to_string(),
            time_on: "051200".to_string(),
            time_off: "051345".to_string(),
            band: "2M".to_string(),
            freq: Some(144.360),
            mode: "MSK144".to_string(),
            rst_sent: "28".to_string(),
            rst_rcvd: "27".to_string(),
        };
        
        let adif = record.to_adif();
        assert!(adif.contains("<CALL:5>DJ5HJ"));
        assert!(adif.contains("<QSO_DATE:8>20260202"));
        assert!(adif.contains("<TIME_ON:6>051200"));
        assert!(adif.contains("<BAND:2>2M"));
        assert!(adif.contains("<MODE:6>MSK144"));
        assert!(adif.contains("<EOR>"));
    }
    
    #[test]
    fn test_parse_adif_record() {
        let line = "<CALL:5>DJ5HJ <QSO_DATE:8>20260202 <TIME_ON:6>051200 <TIME_OFF:6>051345 <BAND:2>2M <MODE:6>MSK144 <RST_SENT:2>28 <RST_RCVD:2>27 <EOR>";
        let record = parse_adif_record(line).unwrap();
        assert_eq!(record.call, "DJ5HJ");
        assert_eq!(record.qso_date, "20260202");
        assert_eq!(record.time_on, "051200");
        assert_eq!(record.rst_sent, "28");
    }
}
