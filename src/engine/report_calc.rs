// src/engine/report_calc.rs
use std::time::{SystemTime, UNIX_EPOCH};

/// Calculate signal reports based on decode quality (correlation)
/// Maps correlation to report codes: 26, 27, 28, 29, 36, 37
pub struct ReportCalculator {
    window_secs: f64,
    corr_min: f32,
    corr_max: f32,
    max_hits: usize,
    corr_weight: f32,
    hit_weight: f32,
    hysteresis: f32,
    last_report: String,
    last_q: f32,
}

impl Default for ReportCalculator {
    fn default() -> Self {
        Self {
            window_secs: 30.0,
            corr_min: 0.40,
            corr_max: 0.85,
            max_hits: 6,
            corr_weight: 0.7,
            hit_weight: 0.3,
            hysteresis: 0.05,
            last_report: "26".to_string(),
            last_q: 0.0,
        }
    }
}

impl ReportCalculator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a successful decode to history
    pub fn add_decode(&self, history: &mut Vec<(f64, f32)>, correlation: f32) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        history.push((now, correlation));

        // Prune old entries
        let cutoff = now - self.window_secs;
        history.retain(|(t, _)| *t >= cutoff);
    }

    /// Compute quality score Q from decode history
    /// Returns (Q, max_corr, hit_count)
    pub fn compute_quality(&self, history: &[(f64, f32)]) -> (f32, f32, usize) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let cutoff = now - self.window_secs;

        // Filter to window
        let recent: Vec<_> = history
            .iter()
            .filter(|(t, _)| *t >= cutoff)
            .collect();

        if recent.is_empty() {
            return (0.0, 0.0, 0);
        }

        // Correlation quality: best correlation in window, scaled to [0,1]
        let max_corr = recent
            .iter()
            .map(|(_, c)| *c)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let c = ((max_corr - self.corr_min) / (self.corr_max - self.corr_min))
            .max(0.0)
            .min(1.0);

        // Hit quality: count of decodes, scaled to [0,1]
        let hit_count = recent.len();
        let h = (hit_count as f32 / self.max_hits as f32).min(1.0);

        // Combined quality score
        let q = self.corr_weight * c + self.hit_weight * h;

        (q, max_corr, hit_count)
    }

    /// Compute report string with hysteresis
    /// Returns: report string ("26", "27", "28", "29", "36", or "37")
    pub fn compute_report(&mut self, history: &[(f64, f32)]) -> String {
        let (q, _max_corr, _hit_count) = self.compute_quality(history);

        // Q thresholds for report bins
        let thresholds = [
            (0.75, "37"), // Strong/consistent
            (0.60, "36"),
            (0.45, "29"),
            (0.30, "28"),
            (0.15, "27"),
            (0.00, "26"), // Barely any/very weak
        ];

        // Find what report the raw Q would give
        let mut new_report = "26";
        for (threshold, report) in &thresholds {
            if q >= *threshold {
                new_report = report;
                break;
            }
        }

        // Check if we should change (with hysteresis)
        if new_report != self.last_report {
            // Find the threshold for the new report
            let new_threshold = thresholds
                .iter()
                .find(|(_, r)| *r == new_report)
                .map(|(t, _)| *t)
                .unwrap_or(0.0);

            // Only change if we've crossed by hysteresis margin
            if new_report > self.last_report.as_str() {
                // Moving up: Q must be above threshold + hysteresis
                if q >= new_threshold + self.hysteresis {
                    self.last_report = new_report.to_string();
                    self.last_q = q;
                }
            } else {
                // Moving down: Q must be below threshold - hysteresis
                if q <= new_threshold - self.hysteresis {
                    self.last_report = new_report.to_string();
                    self.last_q = q;
                }
            }
        } else {
            self.last_q = q;
        }

        self.last_report.clone()
    }
}

/// Simple quality -> report mapping from correlation percentage (0-100)
pub fn report_from_qpct(qpct: f32) -> String {
    let q = qpct.round() as i32;
    if q >= 71 {
        "37".to_string()
    } else if q >= 61 {
        "36".to_string()
    } else if q >= 51 {
        "29".to_string()
    } else if q >= 41 {
        "28".to_string()
    } else if q >= 21 {
        "27".to_string()
    } else {
        "26".to_string()
    }
}

/// Quick report from single correlation value (0.0 to 1.0)
/// Used when clicking on a CQ to immediately compute report
pub fn report_from_correlation(corr: f32) -> String {
    // Map correlation 0.0-1.0 to Q% 0-100
    let qpct = (corr * 100.0).max(0.0).min(100.0);
    report_from_qpct(qpct)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_from_qpct() {
        assert_eq!(report_from_qpct(75.0), "37");
        assert_eq!(report_from_qpct(65.0), "36");
        assert_eq!(report_from_qpct(55.0), "29");
        assert_eq!(report_from_qpct(45.0), "28");
        assert_eq!(report_from_qpct(25.0), "27");
        assert_eq!(report_from_qpct(15.0), "26");
    }

    #[test]
    fn test_report_from_correlation() {
        assert_eq!(report_from_correlation(0.75), "37");
        assert_eq!(report_from_correlation(0.50), "29");
        assert_eq!(report_from_correlation(0.25), "27");
    }
}
