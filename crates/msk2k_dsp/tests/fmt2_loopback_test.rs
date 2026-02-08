// Format-2 Loopback Test
// 
// This test creates a known Format-2 transmission and verifies it decodes correctly.
//
// This test will:
// 1. Create a known 18-bit info pattern
// 2. Encode it through the REAL Format-2 pipeline
// 3. Transmit it through the REAL modem
// 4. Receive it through the REAL modem  
// 5. Decode it and verify the info bits match
//
// Add this to your test suite to prove the modem works end-to-end

#[cfg(test)]
mod format2_loopback_tests {
    use msk2k_dsp::callsign::CallsignCodec;
    use msk2k_dsp::{fec, fmt1, fmt2};
    use msk2k_dsp::rx::{demodulate_msk_soft, find_sync_soft, extract_packet_soft_format1};
    use msk2k_dsp::decode::decode_packet_soft;
    use std::collections::HashMap;
    use std::f32::consts::PI;

    const SAMPLE_RATE: f32 = 48_000.0;
    const SAMPLES_PER_BIT: usize = 24;
    const CENTER_FREQ: f32 = 1500.0;
    const FREQ_DEV: f32 = 500.0;

    // Known test pattern for 18 info bits  
    const TEST_INFO_BITS: [i32; 18] = [
        1, 0, 1, 0, 1, 0, 1, 0, 1,  // First 9 bits: alternating 1,0
        0, 1, 0, 1, 0, 1, 0, 1, 0   // Last 9 bits: alternating 0,1
    ];

    const TEST_TO_CALL: &str = "GW4WND";

    #[test]
    fn test_format2_full_loopback() {
        println!("\n========================================");
        println!("FORMAT-2 FULL LOOPBACK TEST");
        println!("========================================\n");
        
        let codec = CallsignCodec::new();
        
        // ==========================================
        // TRANSMIT SIDE
        // ==========================================
        println!("--- TRANSMIT SIDE ---");
        println!("Info bits to transmit: {:?}", &TEST_INFO_BITS);
        
        // Step 1: FEC encode the 18 info bits
        let poly_dict = fec::encode_format2(&TEST_INFO_BITS);
        println!("FEC encoded to 9 polynomial streams");
        
        // Step 2: Generate address for TO callsign
        let addr_vec = codec.generate_private_address(TEST_TO_CALL)
            .expect("Failed to generate address");
        let mut addr49 = [0i32; 49];
        addr49.copy_from_slice(&addr_vec[..49]);
        println!("Address for {}: (first 10) {:?}", TEST_TO_CALL, &addr49[..10]);
        
        // Step 3: Use Format-1 sync pattern (same for both formats!)
        let sync43 = fmt1::SYNC_PATTERN_HD43;
        
        // Step 4: Interleave using Format-2 table
        let packet258_vec = fmt2::interleave_format2(&sync43, &addr49, &poly_dict);
        let mut packet258 = [0i32; 258];
        packet258.copy_from_slice(&packet258_vec);
        
        println!("Interleaved packet (first 20 bits): {:?}", &packet258[..20]);
        println!("Interleaved packet (bits 100-120): {:?}", &packet258[100..120]);
        
        // Step 5: Convert to symbols using TX mapping: 0 → +1, 1 → -1
        let symbols: Vec<i32> = packet258
            .iter()
            .map(|&b| if b == 0 { 1 } else { -1 })
            .collect();
        
        println!("TX Symbol mapping: bit 0 → +1, bit 1 → -1");
        println!("Symbols (first 20): {:?}", &symbols[..20]);
        
        // Step 6: Generate MSK waveform
        let audio = generate_msk_waveform(&symbols);
        println!("Generated {} audio samples\n", audio.len());
        
        // ==========================================
        // RECEIVE SIDE
        // ==========================================
        println!("--- RECEIVE SIDE ---");
        
        // Step 1: Demodulate to soft bits
        let soft_bits = demodulate_msk_soft(&audio);
        println!("Demodulated {} soft bits", soft_bits.len());
        println!("Soft bits (first 20): {:?}", 
                 &soft_bits[..20].iter().map(|&x| format!("{:.2}", x)).collect::<Vec<_>>());
        
        // Step 2: Find sync
        let sync = find_sync_soft(&soft_bits);
        println!("\nSync detection:");
        println!("  found: {}", sync.found);
        println!("  position: {}", sync.position);
        println!("  correlation: {:.3}", sync.correlation);
        println!("  sync_bits: {}", sync.sync_bits);
        println!("  polarity: {}", sync.polarity);
        println!("  sync_shift: {}", sync.sync_shift);
        println!("  format_hint: {}", sync.format_hint);
        
        assert!(sync.found, "Sync should be found!");
        assert_eq!(sync.format_hint, 2, "Should detect Format-2!");
        
        // Step 3: Extract packet with polarity correction
        let packet_soft = extract_packet_soft_format1(&soft_bits, &sync)
            .expect("Failed to extract packet");
        println!("\nExtracted packet with polarity={}", sync.polarity);
        println!("Packet soft (first 20): {:?}", 
                 &packet_soft[..20].iter().map(|&x| format!("{:.2}", x)).collect::<Vec<_>>());
        
        // Step 4: Decode packet
        let decoded = decode_packet_soft(&packet_soft, &sync)
            .expect("Failed to decode packet");
        
        println!("\nDecoded packet:");
        println!("  format: {}", decoded.format);
        println!("  info_bits length: {}", decoded.info_bits.len());
        println!("  info_bits: {:?}", decoded.info_bits);
        println!("  addr_bits (first 10): {:?}", &decoded.addr_bits[..10]);
        
        // ==========================================
        // VERIFICATION
        // ==========================================
        println!("\n--- VERIFICATION ---");
        
        assert_eq!(decoded.format, 2, "Should decode as Format-2");
        assert_eq!(decoded.info_bits.len(), 18, "Should have 18 info bits");
        
        // Check if info bits match
        let mut matches = 0;
        let mut mismatches = Vec::new();
        for i in 0..18 {
            if decoded.info_bits[i] == TEST_INFO_BITS[i] {
                matches += 1;
            } else {
                mismatches.push((i, TEST_INFO_BITS[i], decoded.info_bits[i]));
            }
        }
        
        println!("Info bit comparison:");
        println!("  Matches: {}/18", matches);
        if !mismatches.is_empty() {
            println!("  Mismatches:");
            for (idx, expected, got) in &mismatches {
                println!("    Bit {}: expected {}, got {}", idx, expected, got);
            }
        }
        
        // Check address
        let decoded_call = codec.decode_private_address(&decoded.addr_bits[..49]);
        println!("\nAddress comparison:");
        println!("  Expected: {}", TEST_TO_CALL);
        println!("  Decoded:  {}", decoded_call);
        
        // Final assertions
        assert_eq!(matches, 18, "All info bits should match! Got {} mismatches", 18 - matches);
        assert_eq!(decoded_call, TEST_TO_CALL, "Decoded callsign should match!");
        
        println!("\n✅ FORMAT-2 LOOPBACK TEST PASSED!");
        println!("========================================\n");
    }

    fn generate_msk_waveform(symbols: &[i32]) -> Vec<f32> {
        let mut samples = Vec::with_capacity(symbols.len() * SAMPLES_PER_BIT);
        let mut phase = 0.0f32;

        for &symbol in symbols {
            let freq = CENTER_FREQ + (symbol as f32 * FREQ_DEV);
            let phase_step = 2.0 * PI * freq / SAMPLE_RATE;

            for _ in 0..SAMPLES_PER_BIT {
                phase += phase_step;
                samples.push(0.5 * phase.sin());
                if phase > 2.0 * PI {
                    phase -= 2.0 * PI;
                }
            }
        }
        
        samples
    }
}
