
import os, json, re, inspect, sys
import numpy as np

import msk2k_complete as m

OUTDIR = os.path.dirname(__file__)

import glob

def clear_stale_errors():
    for p in glob.glob(os.path.join(OUTDIR, "*_error.json")):
        try:
            os.remove(p)
        except Exception:
            pass

clear_stale_errors()

def save_json(name, obj):
    with open(os.path.join(OUTDIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_npy(name, arr):
    np.save(os.path.join(OUTDIR, name), arr)

def to_int_list(a):
    return [int(x) for x in np.asarray(a).astype(int).tolist()]

def fail(name, obj):
    """Write an error json and exit non-zero."""
    save_json(name, obj)
    print(f"ERROR: wrote {name}", file=sys.stderr)
    raise SystemExit(1)

# ============================================================
# Known fmt2 requirements (from current Python reference)
# interleave_format2 expects:
#   sync_len = 43
#   addr_len = 49
#   poly_bits_dict keys: Pa..Pi
#   each poly length: 18
# ============================================================

FMT2_SYNC_LEN = 43
FMT2_ADDR_LEN = 49
FMT2_POLY_NAMES = ['Pa', 'Pb', 'Pc', 'Pd', 'Pe', 'Pf', 'Pg', 'Ph', 'Pi']
FMT2_POLY_LEN = 18

# ============================================================
# Existing vectors (keep for continuity)
# ============================================================

# ---------- Vector 1: interleave_format1 known zeros ----------
sync_bits  = np.zeros(43, dtype=int)
addr_bits  = np.zeros(49, dtype=int)
poly1_bits = np.zeros(83, dtype=int)
poly2_bits = np.zeros(83, dtype=int)

pkt1 = m.interleave_format1(sync_bits, addr_bits, poly1_bits, poly2_bits).astype(int)

save_json("fmt1_zeros.json", {
    "sync_len": int(len(sync_bits)),
    "addr_len": int(len(addr_bits)),
    "poly1_len": int(len(poly1_bits)),
    "poly2_len": int(len(poly2_bits)),
    "pkt_len": int(len(pkt1)),
    "pkt_first_64": pkt1[:64].tolist(),
    "pkt_last_64": pkt1[-64:].tolist(),
})

# ============================================================
# fmt1 golden vectors (pattern + prng) for interleaver mapping
# ============================================================

def make_fmt1_pattern_vec():
    sync = np.array([(i % 2) for i in range(43)], dtype=int)
    addr = np.array([((i + 1) % 2) for i in range(49)], dtype=int)
    p1   = np.array([((i + 2) % 2) for i in range(83)], dtype=int)
    p2   = np.array([((i + 3) % 2) for i in range(83)], dtype=int)

    pkt = m.interleave_format1(sync, addr, p1, p2).astype(int)

    save_json("fmt1_pattern.json", {
        "name": "fmt1_pattern",
        "sync_bits": to_int_list(sync),
        "addr_bits": to_int_list(addr),
        "poly1_bits": to_int_list(p1),
        "poly2_bits": to_int_list(p2),
        "packet_bits": to_int_list(pkt),
    })

def make_fmt1_prng_vec(seed=12345):
    rng = np.random.default_rng(seed)
    sync = rng.integers(0, 2, size=43, dtype=np.int64).astype(int)
    addr = rng.integers(0, 2, size=49, dtype=np.int64).astype(int)
    p1   = rng.integers(0, 2, size=83, dtype=np.int64).astype(int)
    p2   = rng.integers(0, 2, size=83, dtype=np.int64).astype(int)

    pkt = m.interleave_format1(sync, addr, p1, p2).astype(int)

    save_json(f"fmt1_prng_{seed}.json", {
        "name": f"fmt1_prng_{seed}",
        "seed": int(seed),
        "sync_bits": to_int_list(sync),
        "addr_bits": to_int_list(addr),
        "poly1_bits": to_int_list(p1),
        "poly2_bits": to_int_list(p2),
        "packet_bits": to_int_list(pkt),
    })

make_fmt1_pattern_vec()
make_fmt1_prng_vec(12345)

# ============================================================
# fmt2 key discovery (best-effort, but falls back correctly)
# ============================================================

def discover_fmt2_poly_keys():
    """
    Try to infer which keys interleave_format2 expects.
    If we can't discover, fall back to known Pa..Pi.
    """
    try:
        src = inspect.getsource(m.interleave_format2)
    except Exception:
        return []

    keys = re.findall(r"poly_bits_dict\s*\[\s*['\"]([^'\"]+)['\"]\s*\]", src)

    if not keys:
        m2 = re.search(r"poly_names\s*=\s*\[([^\]]+)\]", src)
        if m2:
            keys = re.findall(r"['\"]([^'\"]+)['\"]", m2.group(1))

    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out

fmt2_keys = []
if hasattr(m, "interleave_format2"):
    fmt2_keys = discover_fmt2_poly_keys()

if not fmt2_keys:
    fmt2_keys = list(FMT2_POLY_NAMES)

# ============================================================
# fmt2 probe + golden vectors (NO guessing grid)
# ============================================================

def make_fmt2_probe_and_goldens(poly_keys):
    if not hasattr(m, "interleave_format2"):
        return

    # ---- PROBE ----
    sync0 = np.zeros(FMT2_SYNC_LEN, dtype=int)
    addr0 = np.zeros(FMT2_ADDR_LEN, dtype=int)
    poly0 = {k: np.zeros(FMT2_POLY_LEN, dtype=int) for k in poly_keys}

    try:
        pkt0 = m.interleave_format2(sync0, addr0, poly0).astype(int)
        save_json("fmt2_probe.json", {
            "sync_len": int(len(sync0)),
            "addr_len": int(len(addr0)),
            "poly_keys": list(poly_keys),
            "poly_len": int(FMT2_POLY_LEN),
            "pkt_len": int(len(pkt0)),
            "pkt_first_64": pkt0[:64].tolist(),
            "pkt_last_64": pkt0[-64:].tolist(),
        })
    except Exception as e:
        fail("fmt2_probe_error.json", {
            "poly_keys_attempted": list(poly_keys),
            "sync_len": int(FMT2_SYNC_LEN),
            "addr_len": int(FMT2_ADDR_LEN),
            "poly_len": int(FMT2_POLY_LEN),
            "error": str(e),
            "hint": "Format2 requires poly_bits_dict keys Pa..Pi, each length 18; sync=43, addr=49."
        })

    # ---- PATTERN ----
    sync = np.array([(i % 2) for i in range(FMT2_SYNC_LEN)], dtype=int)
    addr = np.array([((i + 1) % 2) for i in range(FMT2_ADDR_LEN)], dtype=int)
    poly_dict = {
        k: np.array([((i + 2 + ki) % 2) for i in range(FMT2_POLY_LEN)], dtype=int)
        for ki, k in enumerate(poly_keys)
    }

    try:
        pkt = m.interleave_format2(sync, addr, poly_dict).astype(int)
        save_json("fmt2_pattern.json", {
            "name": "fmt2_pattern",
            "sync_bits": to_int_list(sync),
            "addr_bits": to_int_list(addr),
            "poly_keys": list(poly_keys),
            "poly_len": int(FMT2_POLY_LEN),
            "poly_bits_dict": {k: to_int_list(poly_dict[k]) for k in poly_keys},
            "packet_bits": to_int_list(pkt),
        })
    except Exception as e:
        fail("fmt2_pattern_error.json", {
            "name": "fmt2_pattern",
            "poly_keys": list(poly_keys),
            "poly_len": int(FMT2_POLY_LEN),
            "error": str(e),
        })

    # ---- PRNG ----
    seed = 12345
    rng = np.random.default_rng(seed)
    sync_r = rng.integers(0, 2, size=FMT2_SYNC_LEN, dtype=np.int64).astype(int)
    addr_r = rng.integers(0, 2, size=FMT2_ADDR_LEN, dtype=np.int64).astype(int)
    poly_r = {k: rng.integers(0, 2, size=FMT2_POLY_LEN, dtype=np.int64).astype(int) for k in poly_keys}

    try:
        pkt_r = m.interleave_format2(sync_r, addr_r, poly_r).astype(int)
        save_json(f"fmt2_prng_{seed}.json", {
            "name": f"fmt2_prng_{seed}",
            "seed": int(seed),
            "sync_bits": to_int_list(sync_r),
            "addr_bits": to_int_list(addr_r),
            "poly_keys": list(poly_keys),
            "poly_len": int(FMT2_POLY_LEN),
            "poly_bits_dict": {k: to_int_list(poly_r[k]) for k in poly_keys},
            "packet_bits": to_int_list(pkt_r),
        })
    except Exception as e:
        fail(f"fmt2_prng_{seed}_error.json", {
            "name": f"fmt2_prng_{seed}",
            "seed": int(seed),
            "poly_keys": list(poly_keys),
            "poly_len": int(FMT2_POLY_LEN),
            "error": str(e),
        })

make_fmt2_probe_and_goldens(fmt2_keys)

# ============================================================
# NEW: encoder golden vectors (fmt1 + fmt2)
# Your encoder returns a SINGLE ndarray:
#   fmt1: 166 bits (already interleaved 1/2)
#   fmt2: 162 bits (already interleaved 1/9)
# We de-interleave to the per-polynomial arrays needed by interleavers.
# ============================================================

def _find_encoder():
    """
    Find an encoder object in msk2k_complete that provides:
      - encode_format1(info_bits)-> np.ndarray (166)
      - encode_format2(info_bits)-> np.ndarray (162)
    """
    preferred = [
        "PSK2kConvolutionalEncoder",
        "ConvolutionalEncoder",
        "FECEncoder",
        "ConvEncoder",
        "MSK2KEncoder",
        "MSK2KPacketBuilder",
    ]

    def try_make(name):
        try:
            cls = getattr(m, name)
            inst = cls()
            if hasattr(inst, "encode_format1") and hasattr(inst, "encode_format2"):
                return inst
        except Exception:
            return None
        return None

    for name in preferred:
        if hasattr(m, name):
            inst = try_make(name)
            if inst is not None:
                return inst

    for _name, obj in inspect.getmembers(m):
        try:
            if inspect.isclass(obj):
                inst = obj()
                if hasattr(inst, "encode_format1") and hasattr(inst, "encode_format2"):
                    return inst
        except Exception:
            continue

    return None

def _fmt1_split_166_to_poly83(encoded166: np.ndarray):
    encoded166 = np.asarray(encoded166).astype(int)
    if len(encoded166) != 166:
        raise ValueError(f"encode_format1 returned {len(encoded166)} bits, expected 166")
    poly1 = encoded166[0::2].copy()
    poly2 = encoded166[1::2].copy()
    if len(poly1) != 83 or len(poly2) != 83:
        raise ValueError("internal split error: expected 83/83")
    return poly1, poly2

def _fmt2_split_162_to_dict18(encoded162: np.ndarray, poly_keys):
    encoded162 = np.asarray(encoded162).astype(int)
    if len(encoded162) != 162:
        raise ValueError(f"encode_format2 returned {len(encoded162)} bits, expected 162")
    if len(poly_keys) != 9:
        raise ValueError(f"need 9 poly keys, got {len(poly_keys)}: {poly_keys}")

    streams = [np.zeros(FMT2_POLY_LEN, dtype=int) for _ in range(9)]
    for i in range(FMT2_POLY_LEN):
        for j in range(9):
            streams[j][i] = int(encoded162[i*9 + j])

    out = {}
    for j, k in enumerate(poly_keys):
        out[k] = streams[j]
    return out

def make_fmt1_encode_goldens():
    enc = _find_encoder()
    if enc is None:
        fail("fmt1_encode_error.json", {"error": "No encoder found in msk2k_complete.py"})

    print("Encoder detected as:", enc.__class__.__name__)
    try:
        print("encode_format1 signature:", inspect.signature(enc.encode_format1))
        print("encode_format2 signature:", inspect.signature(enc.encode_format2))
    except Exception:
        pass

    # fmt1: 71 info bits → 166 coded
    INFO_LEN = 71

    sync = np.array([(i % 2) for i in range(43)], dtype=int)
    addr = np.array([((i + 1) % 2) for i in range(49)], dtype=int)

    # ---- pattern ----
    info = np.array([(i % 2) for i in range(INFO_LEN)], dtype=int)

    try:
        enc166 = enc.encode_format1(info)
        poly1, poly2 = _fmt1_split_166_to_poly83(enc166)
        pkt = m.interleave_format1(sync, addr, poly1, poly2).astype(int)
    except Exception as e:
        fail("fmt1_encode_pattern_error.json", {
            "error": str(e),
            "info_len": INFO_LEN,
            "hint": "encode_format1 must accept exactly 71 info bits and return 166 coded bits."
        })

    save_json("fmt1_encode_pattern.json", {
        "name": "fmt1_encode_pattern",
        "info_len": int(len(info)),
        "info_bits": to_int_list(info),
        "encoded166_bits": to_int_list(np.asarray(enc166).astype(int)),
        "poly1_bits": to_int_list(poly1),
        "poly2_bits": to_int_list(poly2),
        "sync_bits": to_int_list(sync),
        "addr_bits": to_int_list(addr),
        "packet_bits": to_int_list(pkt),
    })

    # ---- PRNG ----
    seed = 12345
    rng = np.random.default_rng(seed)
    info_r = rng.integers(0, 2, size=INFO_LEN, dtype=np.int64).astype(int)

    try:
        enc166_r = enc.encode_format1(info_r)
        poly1r, poly2r = _fmt1_split_166_to_poly83(enc166_r)
        pkt_r = m.interleave_format1(sync, addr, poly1r, poly2r).astype(int)
    except Exception as e:
        fail("fmt1_encode_prng_12345_error.json", {"error": str(e)})

    save_json("fmt1_encode_prng_12345.json", {
        "name": "fmt1_encode_prng_12345",
        "seed": int(seed),
        "info_len": int(len(info_r)),
        "info_bits": to_int_list(info_r),
        "encoded166_bits": to_int_list(np.asarray(enc166_r).astype(int)),
        "poly1_bits": to_int_list(poly1r),
        "poly2_bits": to_int_list(poly2r),
        "sync_bits": to_int_list(sync),
        "addr_bits": to_int_list(addr),
        "packet_bits": to_int_list(pkt_r),
    })

def make_fmt2_encode_goldens(poly_keys):
    enc = _find_encoder()
    if enc is None:
        fail("fmt2_encode_error.json", {"error": "No encoder found in msk2k_complete.py"})

    # fmt2: 18 info bits → 162 coded
    INFO_LEN = 18

    sync = np.array([(i % 2) for i in range(FMT2_SYNC_LEN)], dtype=int)
    addr = np.array([((i + 1) % 2) for i in range(FMT2_ADDR_LEN)], dtype=int)

    # ---- pattern ----
    info = np.array([(i % 2) for i in range(INFO_LEN)], dtype=int)

    try:
        enc162 = enc.encode_format2(info)
        poly_bits_dict = _fmt2_split_162_to_dict18(enc162, poly_keys)
        pkt = m.interleave_format2(sync, addr, poly_bits_dict).astype(int)
    except Exception as e:
        fail("fmt2_encode_pattern_error.json", {
            "error": str(e),
            "info_len": INFO_LEN,
            "hint": "encode_format2 must accept exactly 18 info bits and return 162 coded bits."
        })

    save_json("fmt2_encode_pattern.json", {
        "name": "fmt2_encode_pattern",
        "info_len": int(len(info)),
        "info_bits": to_int_list(info),
        "encoded162_bits": to_int_list(np.asarray(enc162).astype(int)),
        "poly_keys": list(poly_keys),
        "poly_len": int(FMT2_POLY_LEN),
        "poly_bits_dict": {k: to_int_list(poly_bits_dict[k]) for k in poly_keys},
        "sync_bits": to_int_list(sync),
        "addr_bits": to_int_list(addr),
        "packet_bits": to_int_list(pkt),
    })

    # ---- PRNG ----
    seed = 12345
    rng = np.random.default_rng(seed)
    info_r = rng.integers(0, 2, size=INFO_LEN, dtype=np.int64).astype(int)

    try:
        enc162_r = enc.encode_format2(info_r)
        poly_bits_dict_r = _fmt2_split_162_to_dict18(enc162_r, poly_keys)
        pkt_r = m.interleave_format2(sync, addr, poly_bits_dict_r).astype(int)
    except Exception as e:
        fail("fmt2_encode_prng_12345_error.json", {"error": str(e)})

    save_json("fmt2_encode_prng_12345.json", {
        "name": "fmt2_encode_prng_12345",
        "seed": int(seed),
        "info_len": int(len(info_r)),
        "info_bits": to_int_list(info_r),
        "encoded162_bits": to_int_list(np.asarray(enc162_r).astype(int)),
        "poly_keys": list(poly_keys),
        "poly_len": int(FMT2_POLY_LEN),
        "poly_bits_dict": {k: to_int_list(poly_bits_dict_r[k]) for k in poly_keys},
        "sync_bits": to_int_list(sync),
        "addr_bits": to_int_list(addr),
        "packet_bits": to_int_list(pkt_r),
    })

make_fmt1_encode_goldens()
make_fmt2_encode_goldens(fmt2_keys)

# ============================================================
# Vector 3: modulator waveform summary (fixed)
# ============================================================

bits = np.array(([0, 1] * 129), dtype=int)[:258]

try:
    mod = m.MSK2KModulator(sample_rate=48000)

    if hasattr(mod, "generate_packet_audio"):
        y = mod.generate_packet_audio(bits)
    elif hasattr(mod, "modulate_msk"):
        y = mod.modulate_msk(bits)
    elif hasattr(mod, "modulate"):
        y = mod.modulate(bits)
    elif hasattr(mod, "bits_to_audio"):
        y = mod.bits_to_audio(bits)
    elif hasattr(mod, "generate"):
        y = mod.generate(bits)
    else:
        raise RuntimeError(
            "MSK2KModulator has no known modulate method "
            "(generate_packet_audio/modulate_msk/modulate/bits_to_audio/generate)"
        )

    y = np.asarray(y, dtype=np.float32)

    save_npy("msk_wave.npy", y)

    save_json("msk_wave_summary.json", {
        "sr": 48000,
        "n": int(len(y)),
        "rms": float(np.sqrt(np.mean(y*y))),
        "peak": float(np.max(np.abs(y))),
        "first_16": [float(v) for v in y[:16]],
    })
except Exception as e:
    save_json("msk_wave_error.json", {"error": str(e)})

print("Wrote vectors into:", OUTDIR)

def load_json(name):
    with open(os.path.join(OUTDIR, name), "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# Step 5: RX frontend goldens (demod -> sync -> extract packet)
# ============================================================

def make_rx_frontend_goldens():
    """
    Generate golden RX intermediates from the existing Python modem.
    Output:
      - rx_baseband.npy      : soft bits from receiver._demodulate(audio)
      - rx_sync.json         : receiver._find_sync(baseband) dict
      - rx_packet_soft.npy   : receiver._extract_packet(baseband, sync) soft bits (len 258) if found
    """

    # Reference packet bits (MUST be exactly 258 for this modem)
    pkt_bits = np.array(([0, 1] * 129), dtype=int)[:258]

    # --- Generate packet audio (exactly one packet) ---
    mod = m.MSK2KModulator(sample_rate=48000)

    if hasattr(mod, "generate_packet_audio"):
        pkt_audio = mod.generate_packet_audio(pkt_bits)
    elif hasattr(mod, "modulate_msk"):
        pkt_audio = mod.modulate_msk(pkt_bits)
    elif hasattr(mod, "modulate"):
        pkt_audio = mod.modulate(pkt_bits)
    elif hasattr(mod, "bits_to_audio"):
        pkt_audio = mod.bits_to_audio(pkt_bits)
    elif hasattr(mod, "generate"):
        pkt_audio = mod.generate(pkt_bits)
    else:
        fail("rx_frontend_error.json", {"error": "No modulate method found on MSK2KModulator"})

    pkt_audio = np.asarray(pkt_audio, dtype=np.float32)

    # --- Build a longer RX stream WITHOUT changing bits ---
    sr = 48000
    lead_s = 0.25   # 250 ms silence
    gap_s  = 0.15   # 150 ms between packets
    tail_s = 0.25   # 250 ms silence

    lead = np.zeros(int(lead_s * sr), dtype=np.float32)
    gap  = np.zeros(int(gap_s  * sr), dtype=np.float32)
    tail = np.zeros(int(tail_s * sr), dtype=np.float32)

    # Repeat the same packet twice to make sync/extraction easier
    audio = np.concatenate([lead, pkt_audio, gap, pkt_audio, tail])

    save_npy("rx_audio.npy", audio)
    save_json("rx_audio_meta.json", {"sr": sr, "n": int(len(audio))})

    # --- RX frontend stages ---
    rx = m.MSK2KReceiver(sample_rate=sr)

    # 1) Demodulate to soft bits (should now be > 258)
    baseband = rx._demodulate(audio)
    baseband = np.asarray(baseband, dtype=np.float32)
    save_npy("rx_baseband.npy", baseband)

    # 2) Find sync
    sync = rx._find_sync(baseband)
    save_json("rx_sync.json", sync)

    # 3) Extract packet soft bits (258)
    packet_soft = None
    try:
        packet_soft = rx._extract_packet(baseband, sync)
    except Exception as e:
        save_json("rx_packet_soft_error.json", {"error": str(e), "sync": sync})

    if packet_soft is None:
        save_json("rx_packet_soft_none.json", {
            "error": "receiver._extract_packet returned None",
            "sync": sync,
            "baseband_len": int(len(baseband)),
            "pkt_bits_len": int(len(pkt_bits)),
            "pkt_audio_len": int(len(pkt_audio)),
            "hint": (
                "We padded audio + repeated packet. If still None, the receiver may expect a different "
                "baseband format (symbols vs bits) or _extract_packet expects extra metadata from _find_sync."
            )
        })
        print("WARN: rx_packet_soft unavailable (wrote rx_packet_soft_none.json)")
        return

    packet_soft = np.asarray(packet_soft, dtype=np.float32)
    save_npy("rx_packet_soft.npy", packet_soft)

# Call it (keep near the other make_* calls)
make_rx_frontend_goldens()
