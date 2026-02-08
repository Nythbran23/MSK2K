/// Sample rate conversion using rubato
use rubato::{FftFixedIn, FftFixedInOut, FftFixedOut, Resampler, ResamplerConstructionError};

use crate::types::{AudioSample, SampleRate};

/// Errors related to resampling
#[derive(Debug, thiserror::Error)]
pub enum ResamplerError {
    #[error("Failed to create resampler: {0}")]
    ConstructionError(#[from] ResamplerConstructionError),

    #[error("Resampling failed: {0}")]
    ProcessingError(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
}

/// Resampler type enum for different use cases
pub enum ResamplerType {
    /// Fixed input size, variable output (for consistent input buffers)
    FixedIn(FftFixedIn<f32>),

    /// Variable input size, fixed output (for consistent output buffers)
    FixedOut(FftFixedOut<f32>),

    /// Fixed input and output (most efficient when both are known)
    FixedInOut(FftFixedInOut<f32>),
}

/// Audio resampler wrapper
pub struct AudioResampler {
    resampler: ResamplerType,
    input_rate: SampleRate,
    output_rate: SampleRate,
}

impl AudioResampler {
    /// Create a new resampler with fixed input chunk size
    ///
    /// Best for reading from audio input at a fixed buffer size
    pub fn new_fixed_in(
        input_rate: SampleRate,
        output_rate: SampleRate,
        chunk_size: usize,
    ) -> Result<Self, ResamplerError> {
        let resampler = FftFixedIn::<f32>::new(
            input_rate as usize,
            output_rate as usize,
            chunk_size,
            2, // sub_chunks (lower = less latency, higher = more efficient)
            1, // num_channels
        )?;

        Ok(Self {
            resampler: ResamplerType::FixedIn(resampler),
            input_rate,
            output_rate,
        })
    }

    /// Create a new resampler with fixed output chunk size
    ///
    /// Best for writing to audio output at a fixed buffer size
    pub fn new_fixed_out(
        input_rate: SampleRate,
        output_rate: SampleRate,
        chunk_size: usize,
    ) -> Result<Self, ResamplerError> {
        let resampler = FftFixedOut::<f32>::new(
            input_rate as usize,
            output_rate as usize,
            chunk_size,
            2, // sub_chunks
            1, // num_channels
        )?;

        Ok(Self {
            resampler: ResamplerType::FixedOut(resampler),
            input_rate,
            output_rate,
        })
    }

    /// Create a new resampler with both input and output fixed
    ///
    /// Most efficient when both buffer sizes are known and constant
    pub fn new_fixed_in_out(
        input_rate: SampleRate,
        output_rate: SampleRate,
        input_chunk_size: usize,
        output_chunk_size: usize,
    ) -> Result<Self, ResamplerError> {
        // FftFixedInOut only takes 4 parameters in rubato 0.15
        let resampler = FftFixedInOut::<f32>::new(
            input_rate as usize,
            output_rate as usize,
            input_chunk_size,
            output_chunk_size,
        )?;

        Ok(Self {
            resampler: ResamplerType::FixedInOut(resampler),
            input_rate,
            output_rate,
        })
    }

    /// Process a chunk of samples
    ///
    /// Returns the resampled output. The output size depends on the resampler type.
    pub fn process(&mut self, input: &[AudioSample]) -> Result<Vec<AudioSample>, ResamplerError> {
        // Wrap input in Vec<Vec<f32>> as required by rubato (for multi-channel support)
        let input_channels = vec![input.to_vec()];

        let output_channels = match &mut self.resampler {
            ResamplerType::FixedIn(r) => r
                .process(&input_channels, None)
                .map_err(|e| ResamplerError::ProcessingError(e.to_string()))?,

            ResamplerType::FixedOut(r) => r
                .process(&input_channels, None)
                .map_err(|e| ResamplerError::ProcessingError(e.to_string()))?,

            ResamplerType::FixedInOut(r) => r
                .process(&input_channels, None)
                .map_err(|e| ResamplerError::ProcessingError(e.to_string()))?,
        };

        // Extract mono channel
        Ok(output_channels.into_iter().next().unwrap_or_default())
    }

    /// Get the input sample rate
    pub fn input_rate(&self) -> SampleRate {
        self.input_rate
    }

    /// Get the output sample rate
    pub fn output_rate(&self) -> SampleRate {
        self.output_rate
    }

    /// Get the resampling ratio
    pub fn ratio(&self) -> f64 {
        self.output_rate as f64 / self.input_rate as f64
    }

    /// Calculate expected output size for a given input size
    pub fn output_size_for_input(&self, input_size: usize) -> usize {
        ((input_size as f64) * self.ratio()).ceil() as usize
    }

    /// Calculate expected input size for a given output size
    pub fn input_size_for_output(&self, output_size: usize) -> usize {
        ((output_size as f64) / self.ratio()).ceil() as usize
    }
}

/// Helper function to determine if resampling is needed
pub fn needs_resampling(input_rate: SampleRate, output_rate: SampleRate) -> bool {
    input_rate != output_rate
}

/// Helper function to calculate the GCD (for optimal chunk sizes)
pub fn calculate_optimal_chunk_size(
    input_rate: SampleRate,
    output_rate: SampleRate,
    max_size: usize,
) -> usize {
    let gcd = gcd(input_rate as usize, output_rate as usize);
    let base_chunk = input_rate as usize / gcd;

    // Find largest multiple of base_chunk that's <= max_size
    let multiplier = max_size / base_chunk;
    if multiplier > 0 {
        base_chunk * multiplier
    } else {
        base_chunk
    }
}

/// Calculate GCD using Euclidean algorithm
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_needs_resampling() {
        assert!(needs_resampling(48000, 12000));
        assert!(!needs_resampling(12000, 12000));
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(48000, 12000), 12000);
        assert_eq!(gcd(44100, 48000), 300);
    }

    #[test]
    fn test_optimal_chunk_size() {
        let chunk = calculate_optimal_chunk_size(48000, 12000, 1024);
        assert!(chunk > 0);
        assert!(chunk <= 1024);
    }

    #[test]
    fn test_resampler_48k_to_12k() -> Result<(), ResamplerError> {
        let mut resampler = AudioResampler::new_fixed_in(48000, 12000, 4800)?;

        // Create test signal (1 second at 48kHz)
        let input: Vec<f32> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin())
            .collect();

        let output = resampler.process(&input)?;

        // Should be approximately 1200 samples (1 second at 12kHz)
        assert!(output.len() >= 1190 && output.len() <= 1210);

        Ok(())
    }
}
