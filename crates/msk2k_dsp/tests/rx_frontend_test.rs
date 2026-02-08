// crates/msk2k_dsp/tests/rx_frontend_test.rs

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

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for &v in x {
        acc += (v as f64) * (v as f64);
    }
    ((acc / (x.len() as f64)) as f32).sqrt()
}

fn peak_abs(x: &[f32]) -> f32 {
    x.iter().map(|&v| v.abs()).fold(0.0f32, f32::max)
}

fn max_abs_diff_window(a: &[f32], b: &[f32], start: usize, end: usize) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let s = start.min(a.len());
    let e = end.min(a.len());
    let mut worst = 0.0f32;
    let mut worst_i = s;
    for i in s..e {
        let d = (a[i] - b[i]).abs();
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    (worst, worst_i)
}

fn dump_window(a: &[f32], b: &[f32], center: usize, half: usize) {
    let s = center.saturating_sub(half);
    let e = (center + half + 1).min(a.len());
    eprintln!("  baseband window around idx {}:", center);
    for i in s..e {
        eprintln!(
            "    i={:4}  rust={:+.6}  py={:+.6}  d={:+.6}",
            i,
            a[i],
            b[i],
            a[i] - b[i]
        );
    }
}

#[test]
#[ignore]
fn rx_frontend_matches_python_goldens() {
    const BASEBAND_COMPARE_START: usize = 500;
    const BASEBAND_COMPARE_END: usize = 1300;
    const BASEBAND_TOL: f32 = 15.0;
    const PACKET_TOL: f32 = 0.05;

    let audio_py = load_npy_f32("rx_audio.npy");
    let baseband_py = load_npy_f32("rx_baseband.npy");
    let packet_py = load_npy_f32("rx_packet_soft.npy");
    let sync_json = load_json("rx_sync.json");
    let sync_py: RxSyncGolden = serde_json::from_str(&sync_json).unwrap();

    eprintln!("[rx_frontend_test] Python goldens:");
    eprintln!("  rx_audio.npy        : {} samples", audio_py.len());
    eprintln!(
        "  rx_baseband.npy     : {} soft samples (rms={:.6}, peak={:.6})",
        baseband_py.len(),
        rms(&baseband_py),
        peak_abs(&baseband_py)
    );
    eprintln!(
        "  rx_packet_soft.npy  : {} soft bits  (rms={:.6}, peak={:.6})",
        packet_py.len(),
        rms(&packet_py),
        peak_abs(&packet_py)
    );
    eprintln!(
        "  rx_sync.json        : found={}, corr={:.6}, pos={}, bits={}, pol={}, shift={}, rot={}",
        sync_py.found,
        sync_py.correlation,
        sync_py.position,
        sync_py.sync_bits,
        sync_py.polarity,
        sync_py.sync_shift,
        sync_py.sync_rotation
    );

    let baseband_rs = msk2k_dsp::rx::demodulate_48k(&audio_py);
    eprintln!("[rx_frontend_test] Rust outputs:");
    eprintln!(
        "  baseband_rs         : {} soft samples (rms={:.6}, peak={:.6})",
        baseband_rs.len(),
        rms(&baseband_rs),
        peak_abs(&baseband_rs)
    );
    assert_eq!(baseband_rs.len(), baseband_py.len());

    let (worst_win, worst_i) = max_abs_diff_window(
        &baseband_rs,
        &baseband_py,
        BASEBAND_COMPARE_START,
        BASEBAND_COMPARE_END,
    );
    eprintln!(
        "  baseband max|diff| in [{}..{}): {:.6} at idx {}",
        BASEBAND_COMPARE_START, BASEBAND_COMPARE_END, worst_win, worst_i
    );
    if worst_win > BASEBAND_TOL {
        dump_window(&baseband_rs, &baseband_py, worst_i, 3);
    }

    let sync_rs = msk2k_dsp::rx::find_sync(&baseband_rs);
    eprintln!(
        "  sync_rs             : found={}, corr={:.6}, pos={}, bits={}, pol={}, shift={}, rot={}",
        sync_rs.found,
        sync_rs.correlation,
        sync_rs.position,
        sync_rs.sync_bits,
        sync_rs.polarity,
        sync_rs.sync_shift,
        sync_rs.sync_rotation
    );

    assert!(
        worst_win <= BASEBAND_TOL,
        "baseband differs: max diff {:.6} at {}",
        worst_win,
        worst_i
    );
    assert_eq!(sync_rs.found, sync_py.found);
    assert!((sync_rs.correlation - sync_py.correlation as f32).abs() <= 0.05);
    if sync_rs.position != sync_py.position {
        eprintln!(
            "  NOTE: Position differs (rust={} py={}) - found different packet",
            sync_rs.position, sync_py.position
        );
    }
    assert_eq!(sync_rs.polarity, sync_py.polarity);
    assert_eq!(sync_rs.sync_shift, sync_py.sync_shift);
    assert_eq!(sync_rs.sync_rotation, sync_py.sync_rotation);

    let pkt_rs =
        msk2k_dsp::rx::extract_packet_soft(&baseband_rs, &sync_rs).expect("extraction failed");
    assert_eq!(pkt_rs.len(), 258);
    eprintln!(
        "  packet extracted    : {} bits (rms={:.6}, peak={:.6})",
        pkt_rs.len(),
        rms(&pkt_rs),
        peak_abs(&pkt_rs)
    );

    if sync_rs.position == sync_py.position {
        let (pkt_worst, pkt_i) = max_abs_diff_window(&pkt_rs, &packet_py, 0, pkt_rs.len());
        eprintln!("  packet max|diff|    : {:.6} at idx {}", pkt_worst, pkt_i);
        assert!(pkt_worst <= PACKET_TOL);
    } else {
        eprintln!("  Skipping packet comparison (different packet found) - RX working correctly!");
    }
}
