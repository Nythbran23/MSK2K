//! MSK2K Soft-Decision Accumulator
//!
//! Implements multi-ping accumulation for meteor scatter reception

use std::collections::HashMap;

/// Soft-decision accumulator for combining multiple partial pings
pub struct Accumulator {
    /// Accumulated soft bits (258 bits)
    soft_sum: Vec<f32>,

    /// Confidence weights for each bit
    confidence: Vec<f32>,

    /// Number of pings accumulated
    num_pings: usize,

    /// Total weight accumulated
    total_weight: f32,

    /// Diagnostics
    last_dominance_ratio: f32,
    last_valid_bits_mean: f32,
    last_canonical_corr_mean: f32,
}

impl Accumulator {
    /// Create new accumulator
    pub fn new() -> Self {
        Self {
            soft_sum: vec![0.0; 258],
            confidence: vec![0.0; 258],
            num_pings: 0,
            total_weight: 0.0,
            last_dominance_ratio: 0.0,
            last_valid_bits_mean: 0.0,
            last_canonical_corr_mean: 0.0,
        }
    }

    /// Reset accumulator
    pub fn reset(&mut self) {
        self.soft_sum.fill(0.0);
        self.confidence.fill(0.0);
        self.num_pings = 0;
        self.total_weight = 0.0;
        self.last_dominance_ratio = 0.0;
        self.last_valid_bits_mean = 0.0;
        self.last_canonical_corr_mean = 0.0;
    }

    /// Accumulate a soft packet with optional masking and confidence override
    pub fn accumulate_soft_packet(
        &mut self,
        packet_soft: &[f32],
        weight: f32,
        valid_mask: Option<&[bool]>,
        conf_override: Option<&[f32]>,
    ) {
        if packet_soft.len() != 258 {
            return;
        }

        self.num_pings += 1;
        self.total_weight += weight;

        for i in 0..258 {
            // Check if this bit is valid (if mask provided)
            let is_valid = valid_mask.map_or(true, |mask| mask.get(i).copied().unwrap_or(false));

            if !is_valid {
                continue;
            }

            // Get confidence for this bit
            let conf = if let Some(conf_arr) = conf_override {
                conf_arr.get(i).copied().unwrap_or(1.0)
            } else {
                1.0
            };

            let weighted_conf = weight * conf;

            // Accumulate
            self.soft_sum[i] += packet_soft[i] * weighted_conf;
            self.confidence[i] += weighted_conf;
        }
    }

    /// Get averaged soft bits
    pub fn get_averaged_soft(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(258);

        for i in 0..258 {
            if self.confidence[i] > 0.0 {
                result.push(self.soft_sum[i] / self.confidence[i]);
            } else {
                result.push(0.0);
            }
        }

        result
    }

    /// Get number of accumulated pings
    pub fn num_pings(&self) -> usize {
        self.num_pings
    }

    /// Get diagnostics
    pub fn diagnostics(&self) -> AccumulatorDiagnostics {
        AccumulatorDiagnostics {
            num_pings: self.num_pings,
            total_weight: self.total_weight,
            dominance_ratio: self.last_dominance_ratio,
            valid_bits_mean: self.last_valid_bits_mean,
            canonical_corr_mean: self.last_canonical_corr_mean,
        }
    }

    /// Set diagnostics (called by decoder)
    pub fn set_diagnostics(
        &mut self,
        dominance_ratio: f32,
        valid_bits_mean: f32,
        canonical_corr_mean: f32,
    ) {
        self.last_dominance_ratio = dominance_ratio;
        self.last_valid_bits_mean = valid_bits_mean;
        self.last_canonical_corr_mean = canonical_corr_mean;
    }
}

impl Default for Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Diagnostics from accumulation
#[derive(Debug, Clone)]
pub struct AccumulatorDiagnostics {
    pub num_pings: usize,
    pub total_weight: f32,
    pub dominance_ratio: f32,
    pub valid_bits_mean: f32,
    pub canonical_corr_mean: f32,
}

/// Phase clustering for multi-ping accumulation
pub struct PhaseClustering {
    bins: HashMap<i32, PhaseBin>,
}

#[derive(Debug, Clone)]
struct PhaseBin {
    indices: Vec<usize>,
    total_weight: f32,
}

impl PhaseClustering {
    pub fn new() -> Self {
        Self {
            bins: HashMap::new(),
        }
    }

    /// Add a candidate to phase clustering
    pub fn add_candidate(&mut self, phase: i32, index: usize, weight: f32) {
        const TOLERANCE: i32 = 6; // ±6 bits tolerance

        // Find existing bin within tolerance
        let mut found_bin = None;
        for &bin_center in self.bins.keys() {
            if (phase - bin_center).abs() <= TOLERANCE {
                found_bin = Some(bin_center);
                break;
            }
        }

        if let Some(center) = found_bin {
            let bin = self.bins.get_mut(&center).unwrap();
            bin.indices.push(index);
            bin.total_weight += weight;
        } else {
            self.bins.insert(
                phase,
                PhaseBin {
                    indices: vec![index],
                    total_weight: weight,
                },
            );
        }
    }

    /// Get dominant phase bin
    pub fn get_dominant_bin(&self) -> Option<(i32, Vec<usize>, f32)> {
        if self.bins.is_empty() {
            return None;
        }

        // Find bin with highest total weight
        let (&dominant_center, dominant_bin) = self
            .bins
            .iter()
            .max_by(|a, b| a.1.total_weight.partial_cmp(&b.1.total_weight).unwrap())?;

        // Calculate dominance ratio
        let total_weight: f32 = self.bins.values().map(|b| b.total_weight).sum();
        let dominance_ratio = if total_weight > 0.0 {
            dominant_bin.total_weight / total_weight
        } else {
            0.0
        };

        Some((
            dominant_center,
            dominant_bin.indices.clone(),
            dominance_ratio,
        ))
    }
}

impl Default for PhaseClustering {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_basic() {
        let mut acc = Accumulator::new();

        // Accumulate two packets
        let packet1 = vec![1.0; 258];
        let packet2 = vec![-1.0; 258];

        acc.accumulate_soft_packet(&packet1, 1.0, None, None);
        acc.accumulate_soft_packet(&packet2, 1.0, None, None);

        assert_eq!(acc.num_pings(), 2);

        let averaged = acc.get_averaged_soft();
        // Should average to 0
        assert!(averaged[0].abs() < 0.001);
    }

    #[test]
    fn test_phase_clustering() {
        let mut clustering = PhaseClustering::new();

        // Add candidates with similar phases
        clustering.add_candidate(10, 0, 1.0);
        clustering.add_candidate(12, 1, 1.0); // Within tolerance
        clustering.add_candidate(100, 2, 0.5); // Different phase

        let (center, indices, ratio) = clustering.get_dominant_bin().unwrap();

        // Should select the bin with phases 10,12
        assert_eq!(indices.len(), 2);
        assert!(ratio > 0.6); // Should be dominant
    }

    #[test]
    fn test_accumulator_with_mask() {
        let mut acc = Accumulator::new();

        let packet = vec![1.0; 258];
        let mut mask = vec![false; 258];
        mask[0..100].fill(true); // Only first 100 bits valid

        acc.accumulate_soft_packet(&packet, 1.0, Some(&mask), None);

        let averaged = acc.get_averaged_soft();
        assert_eq!(averaged[0], 1.0); // Valid bit
        assert_eq!(averaged[200], 0.0); // Invalid bit
    }
}
