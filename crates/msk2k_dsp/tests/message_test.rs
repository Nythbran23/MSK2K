// crates/msk2k_dsp/tests/message_test.rs
//
// Integration tests for message encoding/decoding

use msk2k_dsp::callsign::CallsignCodec;
use msk2k_dsp::message::{Message, MessageType};

#[test]
fn test_cq_encode_decode() {
    let codec = CallsignCodec::new();

    // Create CQ message
    let msg = Message::cq("GW4WND");
    assert_eq!(msg.format, 1);
    assert_eq!(msg.text, "CQ de GW4WND");

    // Encode
    let (info_bits, addr_bits) = msg.to_format1_bits(&codec).unwrap();
    assert_eq!(info_bits.len(), 71);
    assert_eq!(addr_bits.len(), 49);

    // Decode
    let decoded = Message::from_format1_bits(&codec, &info_bits, &addr_bits, true).unwrap();
    assert_eq!(decoded.from_call, "GW4WND");
    assert_eq!(decoded.to_call, None);
    match decoded.message_type {
        MessageType::Cq => {}
        _ => panic!("Expected CQ message type"),
    }
}

#[test]
fn test_cold_call_encode_decode() {
    let codec = CallsignCodec::new();

    // Create cold call
    let msg = Message::cold_call("GW4WND", "DJ5HG");
    assert_eq!(msg.format, 1);
    assert_eq!(msg.text, "DJ5HG de GW4WND");

    // Encode
    let (info_bits, addr_bits) = msg.to_format1_bits(&codec).unwrap();
    assert_eq!(info_bits.len(), 71);
    assert_eq!(addr_bits.len(), 49);

    // Decode
    let decoded = Message::from_format1_bits(&codec, &info_bits, &addr_bits, false).unwrap();
    assert_eq!(decoded.from_call, "GW4WND");
    assert_eq!(decoded.to_call, Some("DJ5HG".to_string()));
    match decoded.message_type {
        MessageType::ColdCall => {}
        _ => panic!("Expected ColdCall message type"),
    }
}

#[test]
fn test_call_with_report_26() {
    let codec = CallsignCodec::new();

    let msg = Message::call_with_report("GW4WND", "DJ5HG", "26");
    assert_eq!(msg.text, "DJ5HG de GW4WND 26");

    let (info_bits, addr_bits) = msg.to_format1_bits(&codec).unwrap();
    let decoded = Message::from_format1_bits(&codec, &info_bits, &addr_bits, false).unwrap();

    assert_eq!(decoded.from_call, "GW4WND");
    assert_eq!(decoded.to_call, Some("DJ5HG".to_string()));
    match decoded.message_type {
        MessageType::CallWithReport { report } => assert_eq!(report, "26"),
        _ => panic!("Expected CallWithReport"),
    }
}

#[test]
fn test_call_with_report_27() {
    let codec = CallsignCodec::new();

    let msg = Message::call_with_report("KA1ABC", "W1AW", "27");
    let (info_bits, addr_bits) = msg.to_format1_bits(&codec).unwrap();
    let decoded = Message::from_format1_bits(&codec, &info_bits, &addr_bits, false).unwrap();

    match decoded.message_type {
        MessageType::CallWithReport { report } => assert_eq!(report, "27"),
        _ => panic!("Expected CallWithReport"),
    }
}

#[test]
fn test_call_with_report_37() {
    let codec = CallsignCodec::new();

    let msg = Message::call_with_report("VE3ABC", "G4ABC", "37");
    let (info_bits, addr_bits) = msg.to_format1_bits(&codec).unwrap();
    let decoded = Message::from_format1_bits(&codec, &info_bits, &addr_bits, false).unwrap();

    match decoded.message_type {
        MessageType::CallWithReport { report } => assert_eq!(report, "37"),
        _ => panic!("Expected CallWithReport"),
    }
}

#[test]
fn test_r_report_26() {
    let codec = CallsignCodec::new();

    let msg = Message::r_report("GW4WND", "DJ5HG", "26");
    assert_eq!(msg.format, 2);
    assert_eq!(msg.text, "R26");

    let info_bits = msg.to_format2_bits(&codec).unwrap();
    assert_eq!(info_bits.len(), 18);

    // Message code should be 000
    assert_eq!(info_bits[0], 0);
    assert_eq!(info_bits[1], 0);
    assert_eq!(info_bits[2], 0);

    // Decode
    let addr_bits = vec![0; 49];
    let decoded =
        Message::from_format2_bits(&codec, &info_bits, &addr_bits, "GW4WND", "DJ5HG").unwrap();

    match decoded.message_type {
        MessageType::RReport { report } => assert_eq!(report, "26"),
        _ => panic!("Expected RReport"),
    }
}

#[test]
fn test_r_report_27() {
    let codec = CallsignCodec::new();

    let msg = Message::r_report("KA1ABC", "W1AW", "27");
    let info_bits = msg.to_format2_bits(&codec).unwrap();

    // Message code should be 001
    assert_eq!(info_bits[0], 0);
    assert_eq!(info_bits[1], 0);
    assert_eq!(info_bits[2], 1);

    let addr_bits = vec![0; 49];
    let decoded =
        Message::from_format2_bits(&codec, &info_bits, &addr_bits, "KA1ABC", "W1AW").unwrap();

    match decoded.message_type {
        MessageType::RReport { report } => assert_eq!(report, "27"),
        _ => panic!("Expected RReport"),
    }
}

#[test]
fn test_r_report_all_codes() {
    let codec = CallsignCodec::new();

    let test_cases = vec![
        ("26", 0b000),
        ("27", 0b001),
        ("28", 0b010),
        ("29", 0b011),
        ("36", 0b100),
        ("37", 0b101),
    ];

    for (report, expected_code) in test_cases {
        let msg = Message::r_report("TEST", "PEER", report);
        let info_bits = msg.to_format2_bits(&codec).unwrap();

        let actual_code = (info_bits[0] << 2) | (info_bits[1] << 1) | info_bits[2];
        assert_eq!(actual_code, expected_code, "Wrong code for R{}", report);
    }
}

#[test]
fn test_roger_roger() {
    let codec = CallsignCodec::new();

    let msg = Message::roger_roger("GW4WND", "DJ5HG");
    assert_eq!(msg.format, 2);
    assert_eq!(msg.text, "RR");

    let info_bits = msg.to_format2_bits(&codec).unwrap();

    // Message code should be 110
    assert_eq!(info_bits[0], 1);
    assert_eq!(info_bits[1], 1);
    assert_eq!(info_bits[2], 0);

    let addr_bits = vec![0; 49];
    let decoded =
        Message::from_format2_bits(&codec, &info_bits, &addr_bits, "GW4WND", "DJ5HG").unwrap();

    match decoded.message_type {
        MessageType::RogerRoger => {}
        _ => panic!("Expected RogerRoger"),
    }
}

#[test]
fn test_seventy_three() {
    let codec = CallsignCodec::new();

    let msg = Message::seventy_three("GW4WND", "DJ5HG");
    assert_eq!(msg.format, 2);
    assert_eq!(msg.text, "73");

    let info_bits = msg.to_format2_bits(&codec).unwrap();

    // Message code should be 111
    assert_eq!(info_bits[0], 1);
    assert_eq!(info_bits[1], 1);
    assert_eq!(info_bits[2], 1);

    let addr_bits = vec![0; 49];
    let decoded =
        Message::from_format2_bits(&codec, &info_bits, &addr_bits, "GW4WND", "DJ5HG").unwrap();

    match decoded.message_type {
        MessageType::SeventyThree => {}
        _ => panic!("Expected SeventyThree"),
    }
}

#[test]
fn test_various_callsigns() {
    let codec = CallsignCodec::new();

    let test_calls = vec![
        ("W1AW", "K1A"),
        ("GW4WND", "DJ5HG"),
        ("VE3ABC", "G4XYZ"),
        ("JA1ABC", "ZL1XYZ"),
    ];

    for (my_call, their_call) in test_calls {
        // Test Format 1
        let msg = Message::call_with_report(my_call, their_call, "26");
        let (info_bits, addr_bits) = msg.to_format1_bits(&codec).unwrap();
        let decoded = Message::from_format1_bits(&codec, &info_bits, &addr_bits, false).unwrap();
        assert_eq!(decoded.from_call, my_call);
        assert_eq!(decoded.to_call, Some(their_call.to_string()));

        // Test Format 2
        let msg = Message::r_report(my_call, their_call, "27");
        let info_bits = msg.to_format2_bits(&codec).unwrap();
        let addr_bits = vec![0; 49];
        let decoded =
            Message::from_format2_bits(&codec, &info_bits, &addr_bits, my_call, their_call)
                .unwrap();
        assert_eq!(decoded.from_call, my_call);
        assert_eq!(decoded.to_call, Some(their_call.to_string()));
    }
}

#[test]
fn test_complete_qso_sequence() {
    let codec = CallsignCodec::new();

    // Typical QSO sequence
    let messages = vec![
        Message::cq("GW4WND"),
        Message::call_with_report("DJ5HG", "GW4WND", "26"),
        Message::r_report("GW4WND", "DJ5HG", "27"),
        Message::roger_roger("DJ5HG", "GW4WND"),
        Message::seventy_three("GW4WND", "DJ5HG"),
    ];

    // Verify all messages encode and decode correctly
    for msg in messages {
        match msg.format {
            1 => {
                let (info_bits, addr_bits) = msg.to_format1_bits(&codec).unwrap();
                let is_general = msg.to_call.is_none();
                let decoded =
                    Message::from_format1_bits(&codec, &info_bits, &addr_bits, is_general).unwrap();
                assert_eq!(decoded.from_call, msg.from_call);
            }
            2 => {
                let info_bits = msg.to_format2_bits(&codec).unwrap();
                let addr_bits = vec![0; 49];
                let their_call = msg.to_call.as_ref().unwrap();
                let decoded = Message::from_format2_bits(
                    &codec,
                    &info_bits,
                    &addr_bits,
                    &msg.from_call,
                    their_call,
                )
                .unwrap();
                assert_eq!(decoded.from_call, msg.from_call);
            }
            _ => panic!("Invalid format"),
        }
    }
}

#[test]
fn test_parity_validation() {
    let codec = CallsignCodec::new();

    // Create valid message
    let msg = Message::call_with_report("TEST", "PEER", "26");
    let (mut info_bits, addr_bits) = msg.to_format1_bits(&codec).unwrap();

    // Corrupt parity bit
    info_bits[70] = 1 - info_bits[70];

    // Should fail parity check
    let result = Message::from_format1_bits(&codec, &info_bits, &addr_bits, false);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Parity"));
}

#[test]
fn test_invalid_callsign_rejection() {
    let codec = CallsignCodec::new();

    // Create message with valid callsign
    let msg = Message::cq("TEST");
    let (mut info_bits, addr_bits) = msg.to_format1_bits(&codec).unwrap();

    // Corrupt callsign bits to produce invalid callsign
    for i in 0..54 {
        info_bits[i] = 0;
    }

    // Recalculate parity for corrupted data
    let parity = codec.generate_parity(&info_bits[..56], 32749, 15);
    info_bits[56..71].copy_from_slice(&parity);

    // Should fail callsign validation
    let result = Message::from_format1_bits(&codec, &info_bits, &addr_bits, true);
    // Will either fail on invalid callsign or produce empty/ERROR
    assert!(result.is_err() || result.unwrap().from_call.len() < 3);
}
