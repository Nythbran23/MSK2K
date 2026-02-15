//! Audio ring buffer for packet extraction.
//!
//! Maintains a sliding window of raw audio samples that can be sliced
//! around detected sync peaks for decode.

use std::collections::VecDeque;
#[allow(dead_code)]
/// Ring buffer for raw audio samples.
///
/// Supports efficient push and extraction of slices around specified positions.
pub struct AudioRing {
    buffer: VecDeque<f32>,
    capacity: usize,
    /// Total samples ever pushed (global stream position)
    total_pushed: u64,
}

impl AudioRing {
    /// Create a new ring buffer with given capacity in samples.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            total_pushed: 0,
        }
    }
    
    /// Push new audio samples into the buffer.
    pub fn push(&mut self, samples: &[f32]) {
        for &s in samples {
            self.buffer.push_back(s);
            self.total_pushed += 1;
            
            // Trim if over capacity
            if self.buffer.len() > self.capacity {
                self.buffer.pop_front();
            }
        }
    }
    
    /// Get current global stream position (total samples pushed).
    pub fn stream_position(&self) -> u64 {
        self.total_pushed
    }
    
    /// Get the earliest global position still in buffer.
    pub fn earliest_position(&self) -> u64 {
        self.total_pushed.saturating_sub(self.buffer.len() as u64)
    }
    
    /// Extract a slice around a global stream position.
    ///
    /// Returns None if the requested range is not fully available in the buffer.
    ///
    /// # Arguments
    /// * `t_end` - Global stream position of packet end
    /// * `pre` - Samples before t_end to include
    /// * `post` - Samples after t_end to include
    ///
    /// # Returns
    /// * `Some((audio, t_end_offset))` - The audio slice and offset of t_end within it
    /// * `None` - If the range is not available
    pub fn extract_around(&self, t_end: u64, pre: usize, post: usize) -> Option<(Vec<f32>, usize)> {
        let earliest = self.earliest_position();
        let latest = self.total_pushed;
        
        // Calculate required range in global coordinates
        let start_global = t_end.saturating_sub(pre as u64);
        let end_global = t_end + post as u64;
        
        // Check if range is available
        if start_global < earliest || end_global > latest {
            log::debug!(
                "AudioRing: range [{}, {}] not available, buffer holds [{}, {}]",
                start_global, end_global, earliest, latest
            );
            return None;
        }
        
        // Convert to buffer indices
        let buffer_start = (start_global - earliest) as usize;
        let buffer_end = (end_global - earliest) as usize;
        
        // Extract slice
        let slice: Vec<f32> = self.buffer.iter()
            .skip(buffer_start)
            .take(buffer_end - buffer_start)
            .copied()
            .collect();
        
        // t_end offset within the slice
        let t_end_offset = (t_end - start_global) as usize;
        
        Some((slice, t_end_offset))
    }
    
    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
        // Don't reset total_pushed - maintain global stream position
    }
    
    /// Current buffer length.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_push_and_position() {
        let mut ring = AudioRing::new(100);
        
        assert_eq!(ring.stream_position(), 0);
        assert_eq!(ring.len(), 0);
        
        ring.push(&[1.0, 2.0, 3.0]);
        assert_eq!(ring.stream_position(), 3);
        assert_eq!(ring.len(), 3);
    }
    
    #[test]
    fn test_capacity_limit() {
        let mut ring = AudioRing::new(5);
        
        ring.push(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        
        assert_eq!(ring.len(), 5);
        assert_eq!(ring.stream_position(), 7);
        assert_eq!(ring.earliest_position(), 2);
    }
    
    #[test]
    fn test_extract_around() {
        let mut ring = AudioRing::new(100);
        
        // Push some samples
        let samples: Vec<f32> = (0..50).map(|i| i as f32).collect();
        ring.push(&samples);
        
        // Extract around position 30 with pre=10, post=5
        let result = ring.extract_around(30, 10, 5);
        assert!(result.is_some());
        
        let (slice, offset) = result.unwrap();
        assert_eq!(slice.len(), 15);  // pre + post = 10 + 5
        assert_eq!(offset, 10);  // t_end is at position 10 in the slice
        assert_eq!(slice[0], 20.0);  // starts at global pos 20
        assert_eq!(slice[10], 30.0);  // t_end value
    }
    
    #[test]
    fn test_extract_out_of_range() {
        let mut ring = AudioRing::new(50);
        
        let samples: Vec<f32> = (0..30).map(|i| i as f32).collect();
        ring.push(&samples);
        
        // Try to extract beyond what's available
        let result = ring.extract_around(25, 30, 10);
        assert!(result.is_none());
    }
}
