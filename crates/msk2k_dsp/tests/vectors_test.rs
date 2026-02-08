use serde::Deserialize;
use std::collections::HashMap;
use std::{fs, path::PathBuf};

// ============================================================
// Interleave vectors
// ============================================================

#[derive(Debug, Deserialize)]
struct Fmt1Vector {
    name: Option<String>,
    seed: Option<u64>,
    sync_bits: Vec<i32>,
    addr_bits: Vec<i32>,
    poly1_bits: Vec<i32>,
    poly2_bits: Vec<i32>,
    packet_bits: Vec<i32>,
}

#[derive(Debug, Deserialize)]
struct Fmt2Vector {
    name: Option<String>,
    seed: Option<u64>,
    sync_bits: Vec<i32>,
    addr_bits: Vec<i32>,
    poly_keys: Vec<String>,
    poly_len: i32,
    poly_bits_dict: HashMap<String, Vec<i32>>,
    packet_bits: Vec<i32>,
}

// ============================================================
// Encode vectors (info_bits -> FEC -> polys -> (optional) interleave)
// ============================================================

#[derive(Debug, Deserialize)]
struct Fmt1EncodeVector {
    name: Option<String>,
    seed: Option<u64>,
    info_len: i32,
    info_bits: Vec<i32>,
    sync_bits: Vec<i32>,
    addr_bits: Vec<i32>,
    poly1_bits: Vec<i32>,
    poly2_bits: Vec<i32>,
    packet_bits: Vec<i32>,
}

#[derive(Debug, Deserialize)]
struct Fmt2EncodeVector {
    name: Option<String>,
    seed: Option<u64>,
    info_len: i32,
    info_bits: Vec<i32>,
    sync_bits: Vec<i32>,
    addr_bits: Vec<i32>,
    poly_keys: Vec<String>,
    poly_len: i32,
    poly_bits_dict: HashMap<String, Vec<i32>>,
    packet_bits: Vec<i32>,
}

// ============================================================
// Packet vectors (Step 3): info_bits -> FEC -> interleave == packet_bits
// These reuse the encode JSONs, but the tests are now "full packet build".
// ============================================================

type Fmt1PacketVector = Fmt1EncodeVector;
type Fmt2PacketVector = Fmt2EncodeVector;

fn vectors_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("vectors")
        .join(rel)
}

fn load_json<T: for<'de> Deserialize<'de>>(rel: &str) -> T {
    let p = vectors_path(rel);
    let s = fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!("read json file {:?}: {}", p, e);
    });
    serde_json::from_str(&s).unwrap_or_else(|e| {
        panic!("parse json {:?}: {}", p, e);
    })
}

// ============================================================
// Helper checks
// ============================================================

fn assert_bits01(name: &str, rel: &str, bits: &[i32]) {
    for (i, &b) in bits.iter().enumerate() {
        assert!(
            b == 0 || b == 1,
            "{} contains non-bit value at idx {} in {}: {}",
            name,
            i,
            rel,
            b
        );
    }
}

fn assert_fmt2_poly_map_shape(
    rel: &str,
    keys: &[String],
    poly_len: i32,
    map: &HashMap<String, Vec<i32>>,
) {
    assert_eq!(poly_len, 18, "unexpected poly_len in {}", rel);
    assert_eq!(keys.len(), 9, "unexpected poly_keys count in {}", rel);

    for k in keys {
        let v = map
            .get(k)
            .unwrap_or_else(|| panic!("missing key {} in {}", k, rel));
        assert_eq!(
            v.len(),
            poly_len as usize,
            "unexpected poly length for key {} in {}",
            k,
            rel
        );
    }
}

// ============================================================
// Step 1: Interleave tests (polys -> packet)
// ============================================================

fn run_fmt1_vector(rel: &str) {
    let v: Fmt1Vector = load_json(rel);

    assert_bits01("sync_bits", rel, &v.sync_bits);
    assert_bits01("addr_bits", rel, &v.addr_bits);
    assert_bits01("poly1_bits", rel, &v.poly1_bits);
    assert_bits01("poly2_bits", rel, &v.poly2_bits);
    assert_bits01("packet_bits", rel, &v.packet_bits);

    let rust_packet = msk2k_dsp::fmt1::interleave_format1(
        &v.sync_bits,
        &v.addr_bits,
        &v.poly1_bits,
        &v.poly2_bits,
    );

    assert_eq!(
        rust_packet.len(),
        v.packet_bits.len(),
        "length mismatch for {}",
        rel
    );

    assert_eq!(
        rust_packet, v.packet_bits,
        "packet mismatch for {} (vector {:?}, seed {:?})",
        rel, v.name, v.seed
    );
}

fn run_fmt2_vector(rel: &str) {
    let v: Fmt2Vector = load_json(rel);

    assert_bits01("sync_bits", rel, &v.sync_bits);
    assert_bits01("addr_bits", rel, &v.addr_bits);
    assert_bits01("packet_bits", rel, &v.packet_bits);
    for (k, vv) in &v.poly_bits_dict {
        assert_bits01(&format!("poly_bits_dict[{k}]"), rel, vv);
    }

    assert_fmt2_poly_map_shape(rel, &v.poly_keys, v.poly_len, &v.poly_bits_dict);

    let rust_packet =
        msk2k_dsp::fmt2::interleave_format2(&v.sync_bits, &v.addr_bits, &v.poly_bits_dict);

    assert_eq!(
        rust_packet.len(),
        v.packet_bits.len(),
        "length mismatch for {}",
        rel
    );

    assert_eq!(
        rust_packet, v.packet_bits,
        "packet mismatch for {} (vector {:?}, seed {:?})",
        rel, v.name, v.seed
    );
}

#[test]
fn fmt1_interleave_matches_python_pattern() {
    run_fmt1_vector("fmt1_pattern.json");
}

#[test]
fn fmt1_interleave_matches_python_prng_12345() {
    run_fmt1_vector("fmt1_prng_12345.json");
}

#[test]
fn fmt2_interleave_matches_python_pattern() {
    run_fmt2_vector("fmt2_pattern.json");
}

#[test]
fn fmt2_interleave_matches_python_prng_12345() {
    run_fmt2_vector("fmt2_prng_12345.json");
}

// ============================================================
// Step 2: Encode tests (info_bits -> polys)  [kept]
// ============================================================

fn run_fmt1_encode_vector(rel: &str) {
    let v: Fmt1EncodeVector = load_json(rel);

    assert_eq!(
        v.info_len as usize,
        v.info_bits.len(),
        "info_len mismatch for {} (vector {:?}, seed {:?})",
        rel,
        v.name,
        v.seed
    );

    assert_bits01("info_bits", rel, &v.info_bits);
    assert_bits01("poly1_bits", rel, &v.poly1_bits);
    assert_bits01("poly2_bits", rel, &v.poly2_bits);

    // Your API (as per Option B)
    let (poly1, poly2) = msk2k_dsp::fec::encode_format1(&v.info_bits);

    assert_eq!(
        poly1, v.poly1_bits,
        "fmt1 encode poly1 mismatch for {} (vector {:?}, seed {:?})",
        rel, v.name, v.seed
    );
    assert_eq!(
        poly2, v.poly2_bits,
        "fmt1 encode poly2 mismatch for {} (vector {:?}, seed {:?})",
        rel, v.name, v.seed
    );
}

fn run_fmt2_encode_vector(rel: &str) {
    let v: Fmt2EncodeVector = load_json(rel);

    assert_eq!(
        v.info_len as usize,
        v.info_bits.len(),
        "info_len mismatch for {} (vector {:?}, seed {:?})",
        rel,
        v.name,
        v.seed
    );

    assert_bits01("info_bits", rel, &v.info_bits);
    for (k, vv) in &v.poly_bits_dict {
        assert_bits01(&format!("poly_bits_dict[{k}]"), rel, vv);
    }

    assert_fmt2_poly_map_shape(rel, &v.poly_keys, v.poly_len, &v.poly_bits_dict);

    let poly_map = msk2k_dsp::fec::encode_format2(&v.info_bits);

    // Validate we produced the same key set and same bit-vectors
    for k in &v.poly_keys {
        let got = poly_map
            .get(k)
            .unwrap_or_else(|| panic!("missing key {} in {}", k, rel));
        let exp = v.poly_bits_dict.get(k).unwrap();
        assert_eq!(
            got, exp,
            "fmt2 encode poly {} mismatch for {} (vector {:?}, seed {:?})",
            k, rel, v.name, v.seed
        );
    }
}

#[test]
fn fmt1_encode_matches_python_pattern() {
    run_fmt1_encode_vector("fmt1_encode_pattern.json");
}

#[test]
fn fmt1_encode_matches_python_prng_12345() {
    run_fmt1_encode_vector("fmt1_encode_prng_12345.json");
}

#[test]
fn fmt2_encode_matches_python_pattern() {
    run_fmt2_encode_vector("fmt2_encode_pattern.json");
}

#[test]
fn fmt2_encode_matches_python_prng_12345() {
    run_fmt2_encode_vector("fmt2_encode_prng_12345.json");
}

// ============================================================
// Step 3: Full packet build tests
//   Format1: info_bits -> encode_format1 -> interleave_format1 -> packet_bits
//   Format2: info_bits -> encode_format2 -> interleave_format2 -> packet_bits
// These are the “TX bitstream” equivalence tests (pre-modulation).
// ============================================================

fn run_fmt1_packet_vector(rel: &str) {
    let v: Fmt1PacketVector = load_json(rel);

    assert_eq!(
        v.info_len as usize,
        v.info_bits.len(),
        "info_len mismatch for {} (vector {:?}, seed {:?})",
        rel,
        v.name,
        v.seed
    );

    assert_bits01("info_bits", rel, &v.info_bits);
    assert_bits01("sync_bits", rel, &v.sync_bits);
    assert_bits01("addr_bits", rel, &v.addr_bits);
    assert_bits01("packet_bits", rel, &v.packet_bits);

    let (poly1, poly2) = msk2k_dsp::fec::encode_format1(&v.info_bits);

    // Optional sanity: compare the polys too (helps pinpoint failures)
    assert_eq!(
        poly1, v.poly1_bits,
        "fmt1 packet-build poly1 mismatch for {} (vector {:?}, seed {:?})",
        rel, v.name, v.seed
    );
    assert_eq!(
        poly2, v.poly2_bits,
        "fmt1 packet-build poly2 mismatch for {} (vector {:?}, seed {:?})",
        rel, v.name, v.seed
    );

    let pkt = msk2k_dsp::fmt1::interleave_format1(&v.sync_bits, &v.addr_bits, &poly1, &poly2);

    assert_eq!(
        pkt.len(),
        v.packet_bits.len(),
        "fmt1 packet-build length mismatch for {} (vector {:?}, seed {:?})",
        rel,
        v.name,
        v.seed
    );

    assert_eq!(
        pkt, v.packet_bits,
        "fmt1 packet-build packet mismatch for {} (vector {:?}, seed {:?})",
        rel, v.name, v.seed
    );
}

fn run_fmt2_packet_vector(rel: &str) {
    let v: Fmt2PacketVector = load_json(rel);

    assert_eq!(
        v.info_len as usize,
        v.info_bits.len(),
        "info_len mismatch for {} (vector {:?}, seed {:?})",
        rel,
        v.name,
        v.seed
    );

    assert_bits01("info_bits", rel, &v.info_bits);
    assert_bits01("sync_bits", rel, &v.sync_bits);
    assert_bits01("addr_bits", rel, &v.addr_bits);
    assert_bits01("packet_bits", rel, &v.packet_bits);

    assert_fmt2_poly_map_shape(rel, &v.poly_keys, v.poly_len, &v.poly_bits_dict);

    let poly_map = msk2k_dsp::fec::encode_format2(&v.info_bits);

    // Optional sanity: compare polys too (pinpoint failures by key)
    for k in &v.poly_keys {
        let got = poly_map
            .get(k)
            .unwrap_or_else(|| panic!("missing key {} in {}", k, rel));
        let exp = v.poly_bits_dict.get(k).unwrap();
        assert_eq!(
            got, exp,
            "fmt2 packet-build poly {} mismatch for {} (vector {:?}, seed {:?})",
            k, rel, v.name, v.seed
        );
    }

    let pkt = msk2k_dsp::fmt2::interleave_format2(&v.sync_bits, &v.addr_bits, &poly_map);

    assert_eq!(
        pkt.len(),
        v.packet_bits.len(),
        "fmt2 packet-build length mismatch for {} (vector {:?}, seed {:?})",
        rel,
        v.name,
        v.seed
    );

    assert_eq!(
        pkt, v.packet_bits,
        "fmt2 packet-build packet mismatch for {} (vector {:?}, seed {:?})",
        rel, v.name, v.seed
    );
}

#[test]
fn fmt1_packet_build_matches_python_pattern() {
    run_fmt1_packet_vector("fmt1_encode_pattern.json");
}

#[test]
fn fmt1_packet_build_matches_python_prng_12345() {
    run_fmt1_packet_vector("fmt1_encode_prng_12345.json");
}

#[test]
fn fmt2_packet_build_matches_python_pattern() {
    run_fmt2_packet_vector("fmt2_encode_pattern.json");
}

#[test]
fn fmt2_packet_build_matches_python_prng_12345() {
    run_fmt2_packet_vector("fmt2_encode_prng_12345.json");
}
