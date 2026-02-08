// crates/msk2k_dsp/tests/callsign_test.rs
//
// Integration tests for callsign encoding/decoding
// Verifies against Python implementation

use msk2k_dsp::callsign::CallsignCodec;

#[test]
fn test_common_us_callsigns() {
    let codec = CallsignCodec::new();
    let callsigns = vec!["W1AW", "K1A", "N2MH", "KA1ABC", "WB6JJJ", "N0CALL"];

    for call in callsigns {
        let encoded = codec
            .encode_callsign(call)
            .expect(&format!("Failed to encode {}", call));
        assert_eq!(encoded.len(), 54, "Wrong encoding length for {}", call);

        let decoded = codec.decode_callsign(&encoded);
        assert_eq!(decoded, call, "Roundtrip failed for {}", call);
    }
}

#[test]
fn test_international_callsigns() {
    let codec = CallsignCodec::new();
    let callsigns = vec![
        "VE3ABC", // Canada
        "G4ABC",  // UK
        "JA1ABC", // Japan
        "ZL1ABC", // New Zealand
        "VK3ABC", // Australia
    ];

    for call in callsigns {
        let encoded = codec
            .encode_callsign(call)
            .expect(&format!("Failed to encode {}", call));
        let decoded = codec.decode_callsign(&encoded);
        assert_eq!(decoded, call, "Roundtrip failed for {}", call);
    }
}

#[test]
fn test_special_event_callsign() {
    let codec = CallsignCodec::new();

    // 10-character special event callsign
    let callsign = "VE3SPECIAL";
    let encoded = codec.encode_callsign(callsign).unwrap();

    assert_eq!(encoded.len(), 54);
    assert_eq!(encoded[53], 0, "10-char callsign should have bit 53 = 0");

    let decoded = codec.decode_callsign(&encoded);
    assert_eq!(decoded, callsign);
}

#[test]
fn test_3char_minimum() {
    let codec = CallsignCodec::new();

    let callsign = "K1A";
    let encoded = codec.encode_callsign(callsign).unwrap();
    assert_eq!(encoded.len(), 54);

    let decoded = codec.decode_callsign(&encoded);
    assert_eq!(decoded, callsign);
}

#[test]
fn test_9char_maximum_normal() {
    let codec = CallsignCodec::new();

    let callsign = "VE3ABCDEF"; // 9 chars
    let encoded = codec.encode_callsign(callsign).unwrap();
    assert_eq!(encoded.len(), 54);
    assert_eq!(encoded[53], 1, "9-char callsign should have bit 53 = 1");

    let decoded = codec.decode_callsign(&encoded);
    assert_eq!(decoded, callsign);
}

#[test]
fn test_private_address_encoding() {
    let codec = CallsignCodec::new();

    // Test with a 6-character callsign
    let callsign = "KA1ABC";
    let address = codec.generate_private_address(callsign).unwrap();

    // Should generate 50-bit address
    assert_eq!(address.len(), 50);

    // Decode from first 49 bits (per protocol)
    let decoded = codec.decode_private_address(&address[..49]);
    assert_eq!(decoded, callsign);
}

#[test]
fn test_private_address_various_lengths() {
    let codec = CallsignCodec::new();
    let callsigns = vec!["K1A", "W1AW", "KA1ABC", "VE3ABCD", "N0CALLSIG"];

    for call in callsigns {
        let address = codec.generate_private_address(call).unwrap();
        assert_eq!(address.len(), 50, "Address length wrong for {}", call);

        let decoded = codec.decode_private_address(&address[..49]);
        assert_eq!(decoded, call, "Address decode failed for {}", call);
    }
}

#[test]
fn test_parity_generation_various_primes() {
    let codec = CallsignCodec::new();

    // Test with different source values and primes
    let test_cases = vec![
        (vec![0, 0, 0, 0, 0], 7, 3, vec![0, 0, 0]), // 0 % 7 = 0
        (vec![1, 0, 1, 0, 1], 7, 3, vec![0, 0, 0]), // 21 % 7 = 0
        (vec![1, 1, 1, 1, 1], 7, 3, vec![1, 0, 0]), // 31 % 7 = 3 = 0b011
    ];

    for (source, prime, num_bits, expected) in test_cases {
        let parity = codec.generate_parity(&source, prime, num_bits);
        assert_eq!(parity, expected, "Parity mismatch for source {:?}", source);
    }
}

#[test]
fn test_lowercase_handling() {
    let codec = CallsignCodec::new();

    // Should auto-convert to uppercase
    let encoded_upper = codec.encode_callsign("KA1ABC").unwrap();
    let encoded_lower = codec.encode_callsign("ka1abc").unwrap();
    let encoded_mixed = codec.encode_callsign("Ka1AbC").unwrap();

    assert_eq!(encoded_upper, encoded_lower);
    assert_eq!(encoded_upper, encoded_mixed);
}

#[test]
fn test_whitespace_trimming() {
    let codec = CallsignCodec::new();

    let encoded1 = codec.encode_callsign("KA1ABC").unwrap();
    let encoded2 = codec.encode_callsign("  KA1ABC  ").unwrap();

    assert_eq!(encoded1, encoded2);
}

#[test]
fn test_error_cases() {
    let codec = CallsignCodec::new();

    // Too short
    assert!(codec.encode_callsign("AB").is_err());
    assert!(codec.encode_callsign("K").is_err());

    // Too long
    assert!(codec.encode_callsign("VE3TOOLONGCALL").is_err());

    // Empty
    assert!(codec.encode_callsign("").is_err());
}

#[test]
fn test_decode_error_cases() {
    let codec = CallsignCodec::new();

    // Wrong length
    let bits = vec![0; 50];
    assert_eq!(codec.decode_callsign(&bits), "ERROR");

    // Invalid length code
    let mut bits = vec![0; 54];
    bits[50..54].copy_from_slice(&[1, 1, 1, 0]); // Invalid code
    assert_eq!(codec.decode_callsign(&bits), "ERROR");
}
