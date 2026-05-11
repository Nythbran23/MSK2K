// src/fsk441rx/store.rs
//
// SQLite persistence. WAL mode for safe concurrent reads (e.g. from DB Browser
// or DBeaver while the monitor is running).
//
// Storage philosophy: store metadata only — soft_dits NOT stored.
// At 3.5KB per ping they balloon to 1GB+ quickly. The raw_decode,
// confidence and CCF columns contain everything needed for analysis.
// soft_dits can be re-enabled per-session for targeted experiments.
// preserve all soft information for future off-line analysis.

use std::path::Path;
use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{Connection, params};

use crate::detector::DetectedPing;
use crate::demod::DemodResult;
use crate::filter::ParsedMessage;

pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("Cannot open database: {}", path.display()))?;

        conn.execute_batch("
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous  = NORMAL;
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at  TEXT    NOT NULL,
                ended_at    TEXT,
                device      TEXT,
                notes       TEXT
            );

            CREATE TABLE IF NOT EXISTS pings (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      INTEGER NOT NULL REFERENCES sessions(id),
                -- Timing
                detected_at     TEXT    NOT NULL,
                df_hz           REAL,
                ccf_ratio       REAL,
                duration_ms     REAL,
                -- Hard decode
                raw_decode      TEXT,
                validity_score  INTEGER NOT NULL DEFAULT 0,
                message_type    TEXT,
                callsign_a      TEXT,
                callsign_b      TEXT,
                locator         TEXT,
                report          TEXT,
                is_cq           INTEGER NOT NULL DEFAULT 0,
                -- Soft metric summary
                mean_confidence REAL,
                min_confidence  REAL,
                n_ambiguous     INTEGER,
                -- Soft dit matrix: Vec<[f32;4]> stored as little-endian bytes.
                -- 100 dits × 4 tones × 4 bytes = 1600 bytes typical.
                soft_dits       BLOB
            );

            CREATE INDEX IF NOT EXISTS idx_pings_session
                ON pings(session_id, detected_at);
            CREATE INDEX IF NOT EXISTS idx_pings_score
                ON pings(validity_score DESC);
            CREATE INDEX IF NOT EXISTS idx_pings_callsigns
                ON pings(callsign_a, callsign_b);
        ").context("Database schema creation")?;

        Ok(Self { conn })
    }

    pub fn new_session(&self, device: Option<&str>, notes: Option<&str>) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO sessions (started_at, device, notes) VALUES (?1,?2,?3)",
            params![Utc::now().to_rfc3339(), device, notes],
        ).context("Insert session")?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn close_session(&self, session_id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE sessions SET ended_at = ?1 WHERE id = ?2",
            params![Utc::now().to_rfc3339(), session_id],
        ).context("Close session")?;
        Ok(())
    }

    pub fn insert_ping(
        &self,
        session_id: i64,
        ping:       &DetectedPing,
        result:     &DemodResult,
        parsed:     &ParsedMessage,
    ) -> Result<i64> {
        // soft_dits NOT stored — 3.5KB/ping = 1GB+ at scale
        let soft_blob: Option<Vec<u8>> = None;

        self.conn.execute(
            "INSERT INTO pings (
                session_id, detected_at, df_hz, ccf_ratio, duration_ms,
                raw_decode, validity_score, message_type,
                callsign_a, callsign_b, locator, report, is_cq,
                mean_confidence, min_confidence, n_ambiguous,
                soft_dits
            ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17)",
            params![
                session_id,
                ping.timestamp.to_rfc3339(),
                result.df_hz,
                ping.ccf_ratio,
                ping.duration_ms,
                if result.raw_decode.is_empty() { None } else { Some(&result.raw_decode) },
                parsed.validity_score,
                format!("{:?}", parsed.message_type),
                parsed.callsign_a(),
                parsed.callsign_b(),
                parsed.locator.as_deref(),
                parsed.report.as_deref(),
                parsed.is_cq as i32,
                result.mean_confidence,
                result.min_confidence,
                result.n_ambiguous as i64,
                soft_blob,
            ],
        ).context("Insert ping")?;

        Ok(self.conn.last_insert_rowid())
    }

    pub fn session_summary(&self, session_id: i64) -> Result<SessionSummary> {
        let q = |sql: &str| -> i64 {
            self.conn.query_row(sql, params![session_id], |r| r.get(0)).unwrap_or(0)
        };

        Ok(SessionSummary {
            total_pings:       q("SELECT COUNT(*) FROM pings WHERE session_id=?1"),
            valid_pings:       q("SELECT COUNT(*) FROM pings WHERE session_id=?1 AND validity_score>=60"),
            high_confidence:   q("SELECT COUNT(*) FROM pings WHERE session_id=?1 AND validity_score>=80"),
            unique_callsigns:  q("SELECT COUNT(DISTINCT callsign_a) FROM pings WHERE session_id=?1 AND callsign_a IS NOT NULL"),
            unique_locators:   q("SELECT COUNT(DISTINCT locator) FROM pings WHERE session_id=?1 AND locator IS NOT NULL"),
        })
    }
}

#[derive(Debug)]
pub struct SessionSummary {
    pub total_pings:      i64,
    pub valid_pings:      i64,
    pub high_confidence:  i64,
    pub unique_callsigns: i64,
    pub unique_locators:  i64,
}
