// Standalone Format-2 roundtrip test
// SAFE TO ADD - Does NOT modify any existing code
// 
// To use:
// 1. Copy this file to: crates/msk2k_dsp/tests/format2_test.rs
// 2. Run: cargo test --test format2_test -- --nocapture

use msk2k_dsp::callsign::CallsignCodec;
use msk2k_dsp::fec;
use msk2k_dsp::fmt2;
use msk2k_dsp::message::Message;

#[test]
fn test_format2_encode_decode_rr() {
    println!("\n══════════════════════════════════════════════════════");
    println!("  FORMAT-2 ROUNDTRIP TEST: GW4WND -> G4YNL : RR");
    println!("══════════════════════════════════════════════════════\n");

    let codec = CallsignCodec::new();
    let from = "GW4WND";
    let to = "G4YNL";
    
    // === TRANSMIT ===
    println!("TRANSMIT:");
    let msg = Message::format2(from, to, "RR").unwrap();
    println!("  Message: {:?}", msg);
    
    let info_bits = msg.to_format2_bits(&codec).unwrap();
    println!("  Info bits (18): {:?}", info_bits);
    assert_eq!(info_bits.len(), 18);
    
    // FEC encode
    let poly_streams = fec::encode_format2(&info_bits);
    let mut codeword = Vec::with_capacity(162);
    for i in 0..18 {
        for name in ["Pa", "Pb", "Pc", "Pd", "Pe", "Pf", "Pg", "Ph", "Pi"] {
            codeword.push(poly_streams[name][i]);
        }
    }
    println!("  FEC encoded: {} bits", codeword.len());
    
    // === RECEIVE ===
    println!("\nRECEIVE:");
    
    // FEC decode
    let decoded_bits = fec::decode_format2(&codeword);
    println!("  FEC decoded: {} bits", decoded_bits.len());
    println!("  Original:    {:?}", info_bits);
    println!("  Decoded:     {:?}", decoded_bits);
    
    // Check bit errors
    let mut errors = 0;
    for i in 0..18 {
        if decoded_bits[i] != info_bits[i] {
            println!("  ❌ BIT ERROR at position {}: expected {}, got {}", 
                     i, info_bits[i], decoded_bits[i]);
            errors += 1;
        }
    }
    
    if errors == 0 {
        println!("  ✅ FEC PERFECT: {} bits match!", decoded_bits.len());
    } else {
        panic!("FEC decode failed with {} bit errors!", errors);
    }
    
    // Decode message
    let addr_bits = codec.generate_private_address(to).unwrap();
    let mut addr_49 = addr_bits;
    while addr_49.len() < 49 { addr_49.push(0); }
    addr_49.truncate(49);
    
    let decoded_msg = Message::from_format2_bits(
        &codec,
        &decoded_bits,
        &addr_49,
        to,   // receiver
        from, // transmitter
    ).unwrap();
    
    println!("\nVERIFY:");
    println!("  Original: format={}, from={}, to={:?}, text='{}'",
             msg.format, msg.from_call, msg.to_call, msg.text);
    println!("  Decoded:  format={}, from={}, to={:?}, text='{}'",
             decoded_msg.format, decoded_msg.from_call, 
             decoded_msg.to_call, decoded_msg.text);
    
    assert_eq!(decoded_msg.format, msg.format);
    assert_eq!(decoded_msg.from_call, msg.from_call);
    assert_eq!(decoded_msg.to_call, msg.to_call);
    assert_eq!(decoded_msg.text, msg.text);
    
    println!("\n🎉 COMPLETE ROUNDTRIP SUCCESS!");
    println!("══════════════════════════════════════════════════════\n");
}

#[test]
fn test_format2_encode_decode_73() {
    let codec = CallsignCodec::new();
    
    println!("\nTesting: GW4WND -> G4YNL : 73");
    
    let msg = Message::format2("GW4WND", "G4YNL", "73").unwrap();
    let info_bits = msg.to_format2_bits(&codec).unwrap();
    
    // FEC roundtrip
    let poly_streams = fec::encode_format2(&info_bits);
    let mut codeword = Vec::with_capacity(162);
    for i in 0..18 {
        for name in ["Pa", "Pb", "Pc", "Pd", "Pe", "Pf", "Pg", "Ph", "Pi"] {
            codeword.push(poly_streams[name][i]);
        }
    }
    
    let decoded_bits = fec::decode_format2(&codeword);
    assert_eq!(decoded_bits, info_bits, "FEC decode failed for 73");
    
    // Message decode
    let addr_bits = codec.generate_private_address("G4YNL").unwrap();
    let mut addr_49 = addr_bits;
    while addr_49.len() < 49 { addr_49.push(0); }
    addr_49.truncate(49);
    
    let decoded_msg = Message::from_format2_bits(
        &codec, &decoded_bits, &addr_49, "G4YNL", "GW4WND"
    ).unwrap();
    
    assert_eq!(decoded_msg.text, "73");
    println!("  ✅ PASSED");
}

#[test]
fn test_format2_all_callsigns() {
    let codec = CallsignCodec::new();
    
    let tests = vec![
        ("GW4WND", "G4YNL"),
        ("G4YNL", "GW4WND"),
        ("SM2CEW", "W1AW"),
        ("K1ABC", "VE3XYZ"),
    ];
    
    for (from, to) in tests {
        println!("\nTesting: {} -> {} : RR", from, to);
        
        let msg = Message::format2(from, to, "RR").unwrap();
        let info_bits = msg.to_format2_bits(&codec).unwrap();
        
        // FEC roundtrip
        let poly_streams = fec::encode_format2(&info_bits);
        let mut codeword = Vec::with_capacity(162);
        for i in 0..18 {
            for name in ["Pa", "Pb", "Pc", "Pd", "Pe", "Pf", "Pg", "Ph", "Pi"] {
                codeword.push(poly_streams[name][i]);
            }
        }
        
        let decoded_bits = fec::decode_format2(&codeword);
        assert_eq!(decoded_bits, info_bits);
        
        println!("  ✅ PASSED");
    }
    
    println!("\n✅ ALL CALLSIGN TESTS PASSED!");
}
