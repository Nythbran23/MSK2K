// crates/msk2k_dsp/src/fmt2.rs

use std::collections::HashMap;

pub const SYNC_PATTERN_FORMAT2: [i32; 43] = [
    0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 
    0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0
];

pub const FORMAT2_TABLE: [(&str, usize); 258] = [
    ("s", 1), ("pa", 1), ("a", 1), ("pc", 13), ("pe", 8), ("ph", 11),
    ("s", 2), ("pa", 10), ("a", 2), ("pc", 5), ("pe", 17), ("ph", 3),
    ("s", 3), ("pa", 2), ("a", 3), ("pc", 14), ("pe", 9), ("ph", 12),
    ("s", 4), ("pa", 11), ("a", 4), ("pc", 6), ("pe", 18), ("ph", 4),
    ("s", 5), ("pa", 3), ("a", 5), ("pc", 15), ("pf", 1), ("ph", 13),
    ("s", 6), ("pa", 12), ("a", 6), ("pc", 7), ("pf", 10), ("ph", 5),
    ("s", 7), ("pa", 4), ("a", 7), ("pc", 16), ("pf", 2), ("ph", 14),
    ("s", 8), ("pa", 13), ("a", 8), ("pc", 8), ("pf", 11), ("ph", 6),
    ("s", 9), ("pa", 5), ("a", 9), ("pc", 17), ("pf", 3), ("ph", 15),
    ("s", 10), ("pa", 14), ("a", 10), ("pc", 9), ("pf", 12), ("ph", 7),
    ("s", 11), ("pa", 6), ("a", 11), ("pc", 18), ("pf", 4), ("ph", 16),
    ("s", 12), ("pa", 15), ("a", 12), ("pd", 1), ("pf", 13), ("ph", 8),
    ("s", 13), ("pa", 7), ("a", 13), ("pd", 10), ("pf", 5), ("ph", 17),
    ("s", 14), ("pa", 16), ("a", 14), ("pd", 2), ("pf", 14), ("ph", 9),
    ("s", 15), ("pa", 8), ("a", 15), ("pd", 11), ("pf", 6), ("ph", 18),
    ("s", 16), ("pa", 17), ("a", 16), ("pd", 3), ("pf", 15), ("pi", 1),
    ("s", 17), ("pa", 9), ("a", 17), ("pd", 12), ("pf", 7), ("pi", 10),
    ("s", 18), ("pa", 18), ("a", 18), ("pd", 4), ("pf", 16), ("pi", 2),
    ("s", 19), ("pb", 1), ("a", 19), ("pd", 13), ("pf", 8), ("pi", 11),
    ("s", 20), ("pb", 10), ("a", 20), ("pd", 5), ("pf", 17), ("pi", 3),
    ("s", 21), ("pb", 2), ("a", 21), ("pd", 14), ("pf", 9), ("pi", 12),
    ("s", 22), ("pb", 11), ("a", 22), ("pd", 6), ("pf", 18), ("pi", 4),
    ("s", 23), ("pb", 3), ("a", 23), ("pd", 15), ("pg", 1), ("pi", 13),
    ("s", 24), ("pb", 12), ("a", 24), ("pd", 7), ("pg", 10), ("pi", 5),
    ("s", 25), ("pb", 4), ("a", 25), ("pd", 16), ("pg", 2), ("pi", 14),
    ("s", 26), ("pb", 13), ("a", 26), ("pd", 8), ("pg", 11), ("pi", 6),
    ("s", 27), ("pb", 5), ("a", 27), ("pd", 17), ("pg", 3), ("pi", 15),
    ("s", 28), ("pb", 14), ("a", 28), ("pd", 9), ("pg", 12), ("pi", 7),
    ("s", 29), ("pb", 6), ("a", 29), ("pd", 18), ("pg", 4), ("pi", 16),
    ("s", 30), ("pb", 15), ("a", 30), ("pe", 1), ("pg", 13), ("pi", 8),
    ("s", 31), ("pb", 7), ("a", 31), ("pe", 10), ("pg", 5), ("pi", 17),
    ("s", 32), ("pb", 16), ("a", 32), ("pe", 2), ("pg", 14), ("pi", 9),
    ("s", 33), ("pb", 8), ("a", 33), ("pe", 11), ("pg", 6), ("pi", 18),
    ("s", 34), ("pb", 17), ("a", 34), ("pe", 3), ("pg", 15), ("_", 0),
    ("s", 35), ("pb", 9), ("a", 35), ("pe", 12), ("pg", 7), ("_", 0),
    ("s", 36), ("pb", 18), ("a", 36), ("pe", 4), ("pg", 16), ("_", 0),
    ("s", 37), ("pc", 1), ("a", 37), ("pe", 13), ("pg", 8), ("_", 0),
    ("s", 38), ("pc", 10), ("a", 38), ("pe", 5), ("pg", 17), ("a", 44),
    ("s", 39), ("pc", 2), ("a", 39), ("pe", 14), ("pg", 9), ("a", 45),
    ("s", 40), ("pc", 11), ("a", 40), ("pe", 6), ("pg", 18), ("a", 46),
    ("s", 41), ("pc", 3), ("a", 41), ("pe", 15), ("ph", 1), ("a", 47),
    ("s", 42), ("pc", 12), ("a", 42), ("pe", 7), ("ph", 10), ("a", 48),
    ("s", 43), ("pc", 4), ("a", 43), ("pe", 16), ("ph", 2), ("a", 49),
];

fn get_poly<'a>(poly_bits: &'a HashMap<String, Vec<i32>>, key: &str) -> &'a [i32] {
    poly_bits.get(key).unwrap_or_else(|| panic!("Missing polynomial {}", key)).as_slice()
}

pub fn interleave_format2(
    sync_bits: &[i32],
    addr_bits: &[i32],
    poly_bits: &HashMap<String, Vec<i32>>,
) -> Vec<i32> {
    if sync_bits.len() != 43 || addr_bits.len() != 49 {
        panic!("Invalid lengths: sync={} addr={}", sync_bits.len(), addr_bits.len());
    }

    for k in ["Pa", "Pb", "Pc", "Pd", "Pe", "Pf", "Pg", "Ph", "Pi"] {
        let v = poly_bits.get(k).unwrap_or_else(|| panic!("Invalid polynomial {}", k));
        if v.len() != 18 { panic!("Invalid polynomial {} length: {}", k, v.len()); }
    }

    let pa = get_poly(poly_bits, "Pa"); let pb = get_poly(poly_bits, "Pb");
    let pc = get_poly(poly_bits, "Pc"); let pd = get_poly(poly_bits, "Pd");
    let pe = get_poly(poly_bits, "Pe"); let pf = get_poly(poly_bits, "Pf");
    let pg = get_poly(poly_bits, "Pg"); let ph = get_poly(poly_bits, "Ph");
    let pi = get_poly(poly_bits, "Pi");

    let mut packet = vec![0i32; 258];

    for (pos, (typ, index)) in FORMAT2_TABLE.iter().enumerate() {
        if *typ == "_" { packet[pos] = 0; continue; }
        let idx = index - 1;
        packet[pos] = match *typ {
            "s" => sync_bits[idx], "a" => addr_bits[idx], "pa" => pa[idx],
            "pb" => pb[idx], "pc" => pc[idx], "pd" => pd[idx], "pe" => pe[idx],
            "pf" => pf[idx], "pg" => pg[idx], "ph" => ph[idx], "pi" => pi[idx],
            other => panic!("Unknown type code in FORMAT2_TABLE: {}", other),
        };
    }
    packet
}

pub fn deinterleave_format2(packet: &[i32]) -> (Vec<i32>, Vec<i32>, HashMap<String, Vec<i32>>) {
    if packet.len() != 258 { panic!("deinterleave_format2 requires 258 bits"); }

    let mut sync_bits = vec![0i32; 43]; let mut addr_bits = vec![0i32; 49];
    let mut pa = vec![0i32; 18]; let mut pb = vec![0i32; 18]; let mut pc = vec![0i32; 18];
    let mut pd = vec![0i32; 18]; let mut pe = vec![0i32; 18]; let mut pf = vec![0i32; 18];
    let mut pg = vec![0i32; 18]; let mut ph = vec![0i32; 18]; let mut pi = vec![0i32; 18];

    for (pos, (typ, index)) in FORMAT2_TABLE.iter().enumerate() {
        if *typ == "_" { continue; }
        let idx = index - 1;
        match *typ {
            "s" => sync_bits[idx] = packet[pos], "a" => addr_bits[idx] = packet[pos],
            "pa" => pa[idx] = packet[pos], "pb" => pb[idx] = packet[pos], "pc" => pc[idx] = packet[pos],
            "pd" => pd[idx] = packet[pos], "pe" => pe[idx] = packet[pos], "pf" => pf[idx] = packet[pos],
            "pg" => pg[idx] = packet[pos], "ph" => ph[idx] = packet[pos], "pi" => pi[idx] = packet[pos],
            _ => panic!("Unknown type code: {}", typ),
        }
    }

    let mut poly_dict = HashMap::new();
    poly_dict.insert("Pa".to_string(), pa); poly_dict.insert("Pb".to_string(), pb);
    poly_dict.insert("Pc".to_string(), pc); poly_dict.insert("Pd".to_string(), pd);
    poly_dict.insert("Pe".to_string(), pe); poly_dict.insert("Pf".to_string(), pf);
    poly_dict.insert("Pg".to_string(), pg); poly_dict.insert("Ph".to_string(), ph);
    poly_dict.insert("Pi".to_string(), pi);

    (sync_bits, addr_bits, poly_dict)
}

// 🟢 NEW: Soft Deinterleaver
pub fn deinterleave_format2_soft(packet: &[f32]) -> (Vec<i32>, Vec<i32>, HashMap<String, Vec<f32>>) {
    if packet.len() != 258 { panic!("deinterleave_format2_soft requires 258 bits"); }

    let mut sync_bits = vec![0i32; 43]; let mut addr_bits = vec![0i32; 49];
    let mut pa = vec![0.0f32; 18]; let mut pb = vec![0.0f32; 18]; let mut pc = vec![0.0f32; 18];
    let mut pd = vec![0.0f32; 18]; let mut pe = vec![0.0f32; 18]; let mut pf = vec![0.0f32; 18];
    let mut pg = vec![0.0f32; 18]; let mut ph = vec![0.0f32; 18]; let mut pi = vec![0.0f32; 18];

    for (pos, (typ, index)) in FORMAT2_TABLE.iter().enumerate() {
        if *typ == "_" { continue; }
        let idx = index - 1;
        match *typ {
            "s" => sync_bits[idx] = if packet[pos] > 0.0 { 1 } else { 0 },
            "a" => addr_bits[idx] = if packet[pos] > 0.0 { 1 } else { 0 },
            "pa" => pa[idx] = packet[pos], "pb" => pb[idx] = packet[pos], "pc" => pc[idx] = packet[pos],
            "pd" => pd[idx] = packet[pos], "pe" => pe[idx] = packet[pos], "pf" => pf[idx] = packet[pos],
            "pg" => pg[idx] = packet[pos], "ph" => ph[idx] = packet[pos], "pi" => pi[idx] = packet[pos],
            _ => panic!("Unknown type code: {}", typ),
        }
    }

    let mut poly_dict = HashMap::new();
    poly_dict.insert("Pa".to_string(), pa); poly_dict.insert("Pb".to_string(), pb);
    poly_dict.insert("Pc".to_string(), pc); poly_dict.insert("Pd".to_string(), pd);
    poly_dict.insert("Pe".to_string(), pe); poly_dict.insert("Pf".to_string(), pf);
    poly_dict.insert("Pg".to_string(), pg); poly_dict.insert("Ph".to_string(), ph);
    poly_dict.insert("Pi".to_string(), pi);

    (sync_bits, addr_bits, poly_dict)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format2_roundtrip() {
        let sync = SYNC_PATTERN_FORMAT2.to_vec();
        let addr: Vec<i32> = (0..49).map(|i| i % 2).collect();

        let mut polys = HashMap::new();
        for name in &["Pa", "Pb", "Pc", "Pd", "Pe", "Pf", "Pg", "Ph", "Pi"] {
            let stream: Vec<i32> = (0..18).map(|i| ((i + name.len()) % 2) as i32).collect();
            polys.insert(name.to_string(), stream);
        }

        let packet = interleave_format2(&sync, &addr, &polys);
        assert_eq!(packet.len(), 258);

        let (sync2, addr2, polys2) = deinterleave_format2(&packet);
        assert_eq!(sync, sync2);
        assert_eq!(addr, addr2);

        for name in &["Pa", "Pb", "Pc", "Pd", "Pe", "Pf", "Pg", "Ph", "Pi"] {
            assert_eq!(polys[*name], polys2[*name], "Mismatch in {}", name);
        }
    }

    #[test]
    fn test_sync_pattern_length() {
        assert_eq!(SYNC_PATTERN_FORMAT2.len(), 43);
    }
}