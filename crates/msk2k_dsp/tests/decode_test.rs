// crates/msk2k_dsp/tests/decode_test.rs
use serde::Deserialize;
use std::{fs, path::PathBuf};

fn vectors_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("vectors")
        .join(rel)
}

fn load_json(rel: &str) -> String {
    let p = vectors_path(rel);
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("read json {:?}: {}", p, e))
}

fn load_npy_f32(rel: &str) -> Vec<f32> {
    // Minimal .npy loader for little-endian float32, C-contiguous 1D arrays.
    let p = vectors_path(rel);
    let data = fs::read(&p).unwrap_or_else(|e| panic!("read npy {:?}: {}", p, e));

    assert!(data.len() > 10, "npy too small: {:?}", p);
    assert!(&data[0..6] == b"\x93NUMPY", "bad npy magic in {:?}", p);

    let major = data[6];
    let minor = data[7];
    assert!(
        (major == 1 && minor == 0) || (major == 2 && minor == 0) || (major == 3 && minor == 0),
        "unsupported npy version {}.{} in {:?}",
        major,
        minor,
        p
    );

    let (header_len, header_start) = if major == 1 {
        (u16::from_le_bytes([data[8], data[9]]) as usize, 10usize)
    } else {
        (
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize,
            12usize,
        )
    };

    let header_end = header_start + header_len;
    assert!(data.len() >= header_end, "truncated npy header in {:?}", p);

    let header = std::str::from_utf8(&data[header_start..header_end])
        .unwrap_or_else(|_| panic!("npy header not utf8 in {:?}", p));

    assert!(
        header.contains("'descr': '<f4'")
            || header.contains("\"descr\": \"<f4\"")
            || header.contains("'descr': '|f4'"),
        "expected float32 dtype in {:?}, header={}",
        p,
        header
    );
    assert!(
        header.contains("fortran_order") && header.contains("False"),
        "expected C-order (fortran_order False) in {:?}, header={}",
        p,
        header
    );

    let raw = &data[header_end..];
    assert!(
        raw.len() % 4 == 0,
        "float32 npy data not multiple of 4 bytes in {:?}",
        p
    );

    let mut out = Vec::with_capacity(raw.len() / 4);
    for chunk in raw.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

#[derive(Debug, Deserialize)]
struct RxSyncGolden {
    found: bool,
    correlation: f64,
    position: i32,
    sync_bits: i32,
    polarity: i32,
    sync_shift: i32,
    sync_rotation: i32,
}

#[derive(Debug, Deserialize)]
struct EncodeGolden {
    info_bits: Vec<i32>,
    sync_bits: Vec<i32>,
    addr_bits: Vec<i32>,
    packet_bits: Vec<i32>,
}

fn hard_packet_to_soft(packet_bits: &[i32]) -> Vec<f32> {
    packet_bits
        .iter()
        .map(|&b| if b == 1 { 1.0 } else { -1.0 })
        .collect()
}

#[test]
fn test_decode_format1_pattern() {
    let encode_json = load_json("fmt1_encode_pattern.json");
    let encode_golden: EncodeGolden =
        serde_json::from_str(&encode_json).expect("parse fmt1_encode_pattern.json");

    eprintln!("[decode_test] Loaded fmt1_encode_pattern:");
    eprintln!("  info_bits len: {}", encode_golden.info_bits.len());
    eprintln!("  packet_bits len: {}", encode_golden.packet_bits.len());

    let packet_soft = hard_packet_to_soft(&encode_golden.packet_bits);

    // For pure decode tests we can provide a "known-good" sync context.
    let sync = msk2k_dsp::rx::RxSync {
        found: true,
        correlation: 1.0,
        position: 0,
        sync_bits: 43,
        polarity: 1,
        sync_shift: 0,
        sync_rotation: 0,
    };

    let decoded =
        msk2k_dsp::decode::decode_packet_soft(&packet_soft, &sync).expect("decode failed");

    eprintln!("[decode_test] Decoded:");
    eprintln!("  format: {}", decoded.format);
    eprintln!("  info_bits len: {}", decoded.info_bits.len());
    eprintln!("  addr_bits len: {}", decoded.addr_bits.len());

    assert_eq!(decoded.format, 1);
    assert_eq!(decoded.info_bits.len(), 71); // Format 1 uses 71 info bits
    assert_eq!(
        decoded.info_bits, encode_golden.info_bits,
        "decoded info_bits don't match original!"
    );
    assert_eq!(
        decoded.addr_bits, encode_golden.addr_bits,
        "decoded addr_bits don't match original!"
    );
    assert_eq!(
        decoded.sync_bits, encode_golden.sync_bits,
        "decoded sync_bits don't match original!"
    );

    eprintln!("[decode_test] ✓ Format 1 pattern decode: PASS");
}

#[test]
fn test_decode_format1_prng() {
    let encode_json = load_json("fmt1_encode_prng_12345.json");
    let encode_golden: EncodeGolden =
        serde_json::from_str(&encode_json).expect("parse fmt1_encode_prng_12345.json");

    let packet_soft = hard_packet_to_soft(&encode_golden.packet_bits);

    let sync = msk2k_dsp::rx::RxSync {
        found: true,
        correlation: 1.0,
        position: 0,
        sync_bits: 43,
        polarity: 1,
        sync_shift: 0,
        sync_rotation: 0,
    };

    let decoded =
        msk2k_dsp::decode::decode_packet_soft(&packet_soft, &sync).expect("decode failed");

    assert_eq!(decoded.format, 1);
    assert_eq!(decoded.info_bits, encode_golden.info_bits);
    assert_eq!(decoded.addr_bits, encode_golden.addr_bits);

    eprintln!("[decode_test] ✓ Format 1 PRNG decode: PASS");
}

#[test]
fn test_decode_format2_pattern() {
    let encode_json = load_json("fmt2_encode_pattern.json");
    let encode_golden: EncodeGolden =
        serde_json::from_str(&encode_json).expect("parse fmt2_encode_pattern.json");

    eprintln!("[decode_test] Loaded fmt2_encode_pattern:");
    eprintln!("  info_bits len: {}", encode_golden.info_bits.len());
    eprintln!("  packet_bits len: {}", encode_golden.packet_bits.len());

    let packet_soft = hard_packet_to_soft(&encode_golden.packet_bits);

    let sync = msk2k_dsp::rx::RxSync {
        found: true,
        correlation: 1.0,
        position: 0,
        sync_bits: 43,
        polarity: 1,
        sync_shift: 14,
        sync_rotation: 0,
    };

    let decoded =
        msk2k_dsp::decode::decode_packet_soft(&packet_soft, &sync).expect("decode failed");

    eprintln!("[decode_test] Decoded:");
    eprintln!("  format: {}", decoded.format);
    eprintln!("  info_bits len: {}", decoded.info_bits.len());

    assert_eq!(decoded.format, 2);
    assert_eq!(decoded.info_bits.len(), 18);
    assert_eq!(
        decoded.info_bits, encode_golden.info_bits,
        "decoded info_bits don't match original!"
    );
    assert_eq!(
        decoded.addr_bits, encode_golden.addr_bits,
        "decoded addr_bits don't match original!"
    );

    eprintln!("[decode_test] ✓ Format 2 pattern decode: PASS");
}

#[test]
fn test_decode_format2_prng() {
    let encode_json = load_json("fmt2_encode_prng_12345.json");
    let encode_golden: EncodeGolden =
        serde_json::from_str(&encode_json).expect("parse fmt2_encode_prng_12345.json");

    let packet_soft = hard_packet_to_soft(&encode_golden.packet_bits);

    let sync = msk2k_dsp::rx::RxSync {
        found: true,
        correlation: 1.0,
        position: 0,
        sync_bits: 43,
        polarity: 1,
        sync_shift: 29,
        sync_rotation: 0,
    };

    let decoded =
        msk2k_dsp::decode::decode_packet_soft(&packet_soft, &sync).expect("decode failed");

    assert_eq!(decoded.format, 2);
    assert_eq!(decoded.info_bits, encode_golden.info_bits);
    assert_eq!(decoded.addr_bits, encode_golden.addr_bits);

    eprintln!("[decode_test] ✓ Format 2 PRNG decode: PASS");
}

#[test]
fn test_decode_from_real_rx() {
    let packet_soft = load_npy_f32("rx_packet_soft.npy");
    let sync_json = load_json("rx_sync.json");
    let sync_py: RxSyncGolden = serde_json::from_str(&sync_json).expect("parse rx_sync.json");

    eprintln!("[decode_test] Loaded RX outputs:");
    eprintln!("  packet_soft len: {}", packet_soft.len());
    eprintln!(
        "  sync: found={}, shift={}, rot={}",
        sync_py.found, sync_py.sync_shift, sync_py.sync_rotation
    );

    let sync_rs = msk2k_dsp::rx::RxSync {
        found: sync_py.found,
        correlation: sync_py.correlation as f32,
        position: sync_py.position,
        sync_bits: sync_py.sync_bits,
        polarity: sync_py.polarity,
        sync_shift: sync_py.sync_shift,
        sync_rotation: sync_py.sync_rotation,
    };

    let decoded = msk2k_dsp::decode::decode_packet_soft(&packet_soft, &sync_rs)
        .expect("real RX decode failed");

    eprintln!("[decode_test] Decoded from real RX:");
    eprintln!("  format: {}", decoded.format);
    eprintln!("  info_bits len: {}", decoded.info_bits.len());
    eprintln!(
        "  info_bits (head): {:?}",
        &decoded.info_bits[..decoded.info_bits.len().min(20)]
    );

    if decoded.format == 1 {
        assert_eq!(decoded.info_bits.len(), 71);
    } else {
        assert_eq!(decoded.info_bits.len(), 18);
    }
    assert_eq!(decoded.addr_bits.len(), 49);
    assert_eq!(decoded.sync_bits.len(), 43);

    eprintln!("[decode_test] ✓ Real RX decode: structure valid");
}
