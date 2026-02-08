use std::fs;
use std::path::PathBuf;

fn vectors_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("vectors")
        .join(rel)
}

fn load_npy_f32(rel: &str) -> Vec<f32> {
    // Minimal .npy loader for little-endian float32, C-contiguous 1D arrays.
    // Works for: np.save("msk_wave.npy", y.astype(np.float32))
    let p = vectors_path(rel);
    let data = fs::read(&p).unwrap_or_else(|e| panic!("read npy {:?}: {}", p, e));

    // NPY format: magic \x93NUMPY, version, header_len, header, then raw data.
    assert!(data.len() > 10, "npy too small: {:?}", p);
    assert_eq!(&data[0..6], b"\x93NUMPY", "bad npy magic in {:?}", p);

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
        let hl = u16::from_le_bytes([data[8], data[9]]) as usize;
        (hl, 10)
    } else {
        let hl = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        (hl, 12)
    };

    let header_end = header_start + header_len;
    assert!(data.len() >= header_end, "truncated npy header in {:?}", p);
    let header = std::str::from_utf8(&data[header_start..header_end])
        .unwrap_or_else(|_| panic!("npy header not utf8 in {:?}", p));

    // Very small header parsing: check dtype and fortran_order
    let is_f4 = header.contains("'descr': '<f4'")
        || header.contains("\"descr\": \"<f4\"")
        || header.contains("'descr': '|f4'")
        || header.contains("\"descr\": \"|f4\"");
    assert!(
        is_f4,
        "expected float32 dtype in {:?}, header={}",
        p, header
    );

    // Ensure C-order
    assert!(
        header.contains("fortran_order") && header.contains("False"),
        "expected C-order (fortran_order False) in {:?}, header={}",
        p,
        header
    );

    assert!(
        header.contains("shape"),
        "expected shape in npy header {:?}, header={}",
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

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for &v in x {
        let vv = v as f64;
        acc += vv * vv;
    }
    ((acc / (x.len() as f64)) as f32).sqrt()
}

fn peak_abs(x: &[f32]) -> f32 {
    let mut p = 0.0f32;
    for &v in x {
        let a = v.abs();
        if a > p {
            p = a;
        }
    }
    p
}

fn make_reference_bits_258() -> Vec<i32> {
    // Exactly matches Python:
    // bits = np.array(([0,1] * 129), dtype=int)[:258]
    let mut bits = Vec::with_capacity(258);
    for i in 0..258 {
        bits.push((i % 2) as i32);
    }
    bits
}

#[test]
#[ignore] // Remove this once the Rust MSK modulator exists and matches Python
fn msk_modulator_matches_python_wave_summary() {
    let y_py = load_npy_f32("msk_wave.npy");
    assert!(!y_py.is_empty(), "msk_wave.npy empty");

    let bits = make_reference_bits_258();

    // ============================================================
    // TODO: REPLACE THIS CALL with your actual Rust modulator API.
    //
    // It must return Vec<f32> at 48000 Hz and match Python’s behavior.
    //
    // Examples of what this could look like:
    //   let y_rs = msk2k_dsp::modem::MSK2KModulator::new(48_000).modulate(&bits);
    //   let y_rs = msk2k_dsp::msk::modulate_packet_48k(&bits);
    // ============================================================
    let y_rs: Vec<f32> = msk2k_dsp::msk::modulate_48k(&bits);
    // ============================================================

    assert_eq!(
        y_rs.len(),
        y_py.len(),
        "wave length mismatch (rust={} python={})",
        y_rs.len(),
        y_py.len()
    );

    // Compare summary stats with sensible tolerances.
    let rms_py = rms(&y_py);
    let rms_rs = rms(&y_rs);
    let pk_py = peak_abs(&y_py);
    let pk_rs = peak_abs(&y_rs);

    let rms_tol = 1e-4f32;
    let pk_tol = 1e-4f32;

    assert!(
        (rms_rs - rms_py).abs() <= rms_tol,
        "RMS mismatch: rust={} python={} (tol={})",
        rms_rs,
        rms_py,
        rms_tol
    );

    assert!(
        (pk_rs - pk_py).abs() <= pk_tol,
        "Peak mismatch: rust={} python={} (tol={})",
        pk_rs,
        pk_py,
        pk_tol
    );

    // Compare first N samples loosely
    let n = 64usize.min(y_py.len());
    let samp_tol = 2e-4f32;

    for i in 0..n {
        let d = (y_rs[i] - y_py[i]).abs();
        assert!(
            d <= samp_tol,
            "sample[{}] mismatch: rust={} python={} diff={} tol={}",
            i,
            y_rs[i],
            y_py[i],
            d,
            samp_tol
        );
    }
}
