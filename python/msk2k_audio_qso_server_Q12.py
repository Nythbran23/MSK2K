#!/usr/bin/env python3
"""
MSK2K Audio QSO Test Harness (split RX/TX devices, UI controlled)

- aiohttp server + WebSocket status/decodes
- Real audio I/O using sounddevice (PortAudio)
- Lets you run 2 instances on the same Mac with BlackHole to test sequencing.

Run:
  python3 msk2k_audio_qso_server_RUN10.py --host 127.0.0.1 --port 8088

Open:
  http://127.0.0.1:8088/
"""

from __future__ import annotations

import argparse, asyncio, json, time, math, os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set, List, Tuple
import re

import numpy as np
from aiohttp import web

# PortAudio wrapper
import sounddevice as sd

import msk2k_complete as m

HERE = Path(__file__).resolve().parent

# ---------------- utils ----------------
def utc_clock() -> str:
    t = time.gmtime()
    return f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}Z"

def period_label(slot: int) -> str:
    return "A" if slot == 0 else "B"


def utc_date_adif() -> str:
    t = time.gmtime()
    return f"{t.tm_year:04d}{t.tm_mon:02d}{t.tm_mday:02d}"

def utc_time_adif() -> str:
    t = time.gmtime()
    return f"{t.tm_hour:02d}{t.tm_min:02d}{t.tm_sec:02d}"

def _adif_field(tag: str, value: str) -> str:
    v = "" if value is None else str(value)
    return f"<{tag}:{len(v)}>{v}"

def qso_to_adif(qso: Dict[str, str]) -> str:
    # Minimal, broadly-compatible ADIF record.
    fields = [
        _adif_field("QSO_DATE", qso.get("qso_date","")),
        _adif_field("TIME_ON",  qso.get("time_on","")),
        _adif_field("TIME_OFF", qso.get("time_off","")),
        _adif_field("CALL",     qso.get("call","")),
        _adif_field("RST_SENT", qso.get("rst_sent","")),
        _adif_field("RST_RCVD", qso.get("rst_rcvd","")),
        _adif_field("MODE",     qso.get("mode","MSK2K")),
        _adif_field("SUBMODE",  qso.get("submode","MSK2K")),
        _adif_field("BAND",     qso.get("band","")),
        _adif_field("FREQ",     qso.get("freq","")),
    ]
    return "".join(fields) + "<EOR>\n"


def band_adif_from_inputs(band_sel: str, band_custom: str) -> str:
    """Map UI band selection to ADIF BAND string.

    Common VHF/UHF: 50->6M, 70->4M, 144->2M, 432->70CM, 1296->23CM.
    CUSTOM uses the provided string.
    """
    sel = (band_sel or "").strip().upper()
    custom = (band_custom or "").strip().upper()
    if sel == "CUSTOM":
        return custom
    mapping = {
        "50": "6M",
        "70": "4M",
        "144": "2M",
        "432": "70CM",
        "1296": "23CM",
    }
    return mapping.get(sel, custom or sel)

def get_adif_path() -> Path:
    # In Electron you can pass MSK2K_ADIF_PATH to place this in app.getPath('userData')
    env = os.environ.get("MSK2K_ADIF_PATH","").strip()
    if env:
        return Path(env).expanduser()
    return (Path.cwd() / "msk2k_qsos.adi")

def ensure_adif_header(path: Path) -> None:
    try:
        if path.exists() and path.stat().st_size > 0:
            return
    except Exception:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        _adif_field("ADIF_VER", "3.1.4") +
        _adif_field("PROGRAMID", "MSK2K") +
        _adif_field("PROGRAMVERSION", "0.1") +
        "<EOH>\n"
    )
    path.write_text(header, encoding="utf-8")

def append_adif(path: Path, qso: Dict[str, str]) -> None:
    ensure_adif_header(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(qso_to_adif(qso))

def read_adif_last(path: Path, limit: int = 50) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    txt = path.read_text(encoding="utf-8", errors="ignore")
    # Split on EOR markers (case-insensitive)
    parts = re.split(r"<eor>\s*", txt, flags=re.I)
    # Drop header part (before <EOH>) if present
    # We'll just parse all records and ignore ones without CALL
    qsos: List[Dict[str, str]] = []
    for rec in parts[::-1]:  # reverse for newest first
        rec = rec.strip()
        if not rec:
            continue
        q = {}
        for m2 in re.finditer(r"<([A-Z0-9_]+):(\d+)(:[^>]*)?>", rec, flags=re.I):
            tag = m2.group(1).upper()
            ln = int(m2.group(2))
            start_val = m2.end()
            val = rec[start_val:start_val+ln]
            q[tag.lower()] = val
        if q.get("call"):
            qsos.append({
                "qso_date": q.get("qso_date",""),
                "time_on": q.get("time_on",""),
                "time_off": q.get("time_off",""),
                "band": q.get("band",""),
                "freq": q.get("freq",""),
                "call": q.get("call",""),
                "rst_sent": q.get("rst_sent",""),
                "rst_rcvd": q.get("rst_rcvd",""),
                "mode": q.get("mode",""),
                "submode": q.get("submode",""),
            })
        if len(qsos) >= limit:
            break
    return qsos
# ---------------- QSO states ----------------
QSO_IDLE = "IDLE"
QSO_LISTEN = "LISTEN"
QSO_CALLING_CQ = "CALLING_CQ"
QSO_CALLING_STN = "CALLING_STN"
QSO_HAVE_CALLS = "IN_QSO"
QSO_PRIVATE = "REPORTS"
QSO_DONE = "DONE"

# ---------------- config ----------------
@dataclass
class AudioConfig:
    # Device indices as returned by /api/audio_devices
    rx_device_index: int = 0
    tx_device_index: int = 0

    # 1-based channel selection (e.g. BlackHole 2ch)
    rx_pick: int = 1
    tx_pick: int = 1

    rx_gain: float = 1.0
    tx_gain: float = 0.6

    rx_enable: bool = True
    tx_enable: bool = True


@dataclass
class Config:
    my_call: str = "GW4WND"
    their_call: str = ""
    mode: str = "LISTEN"       # LISTEN|CQ|CALL|REPLY
    period_len: int = 15
    tx_slot: str = "A"         # A|B (preferred slot; backend may override when replying)
    auto_seq: bool = True
    # Logging helpers (manual until CAT)
    band_sel: str = "144"       # UI dropdown (50/70/144/432/1296/CUSTOM)
    band_custom: str = ""       # e.g. 2M
    freq_mhz: str = ""          # e.g. 144.174

    # TX safety timeout
    tx_timeout_min: int = 120
    tx_timeout_override: bool = False
    audio: AudioConfig = field(default_factory=AudioConfig)
    initial_corr: float | None = None  # clicked decode correlation (0..1)


@dataclass
class Runtime:
    running: bool = False
    qso_state: str = QSO_IDLE
    next_tx_text: str = ""
    priv_73_count: int = 0
    observed_remote_slot: Optional[str] = None  # 'A' or 'B' based on last decoded packet time slot
    received_report: str = "27"  # Store the report we received to echo back in R<rpt>
    my_report: str = "26"  # Our report to send (computed from signal quality)
    report_locked: bool = False  # Once set for a QSO, don't change until QSO complete
    current_activity: str = "RX"  # What we're actually doing right now: "RX", "TX", or "IDLE"

    # QSO logging (UTC)
    qso_start_date: str = ""  # YYYYMMDD
    qso_start_time: str = ""  # HHMMSS
    qso_partner: str = ""

    # TX timeout tracking (epoch seconds)
    first_tx_epoch: Optional[float] = None
    
    # Rolling window for quality-based reports
    decode_history: list = None  # List of (timestamp, correlation) tuples
    
    def __post_init__(self):
        if self.decode_history is None:
            self.decode_history = []


def report_from_qpct(qpct: float) -> str:
    """Map Q% (0..100) to classic-style report codes.
    71+ -> 37, 61-70 -> 36, 51-60 -> 29, 41-50 -> 28, 21-40 -> 27, <=20 -> 26
    """
    try:
        q = int(round(float(qpct)))
    except Exception:
        q = 0
    if q >= 71:
        return "37"
    if q >= 61:
        return "36"
    if q >= 51:
        return "29"
    if q >= 41:
        return "28"
    if q >= 21:
        return "27"
    return "26"


class ReportCalculator:
    """
    Compute meaningful signal reports based on correlation quality AND hit count.
    
    This gives a "workability" metric rather than just instantaneous signal strength.
    A station with many weak pings may be more workable than one strong ping.
    
    Report mapping (3 bits, 8 symbols):
        000 = 26 (barely any/very weak)
        001 = 27
        010 = 28  
        011 = 29
        100 = 36
        101 = 37 (strong/consistent)
        110 = RRR (reserved for protocol)
        111 = 73  (reserved for protocol)
    """
    
    # Window duration for averaging (seconds)
    WINDOW_SECONDS = 30.0
    
    # Correlation scaling: map [0.40, 0.85] -> [0, 1]
    CORR_MIN = 0.40
    CORR_MAX = 0.85
    
    # Hit count scaling: 6 hits in window = maxed out
    MAX_HITS = 6
    
    # Weights for combining correlation and hit quality
    CORR_WEIGHT = 0.7
    HIT_WEIGHT = 0.3
    
    # Hysteresis threshold to prevent report flickering
    HYSTERESIS = 0.05
    
    # Q thresholds for report bins
    THRESHOLDS = [
        (0.75, "37"),  # Strong/consistent
        (0.60, "36"),
        (0.45, "29"),
        (0.30, "28"),
        (0.15, "27"),
        (0.00, "26"),  # Barely any/very weak
    ]
    
    def __init__(self):
        self.last_report = "26"
        self.last_Q = 0.0
    
    def add_decode(self, history: list, correlation: float, timestamp: float = None):
        """Add a successful decode to the rolling history."""
        if timestamp is None:
            timestamp = time.time()
        history.append((timestamp, correlation))
        
        # Prune old entries
        cutoff = timestamp - self.WINDOW_SECONDS
        while history and history[0][0] < cutoff:
            history.pop(0)
    
    def compute_quality(self, history: list) -> tuple:
        """
        Compute quality score Q from decode history.
        
        Returns: (Q, max_corr, hit_count) for debugging/logging
        """
        now = time.time()
        cutoff = now - self.WINDOW_SECONDS
        
        # Filter to window
        recent = [(t, c) for t, c in history if t >= cutoff]
        
        if not recent:
            return 0.0, 0.0, 0
        
        # Correlation quality: best correlation in window, scaled to [0,1]
        max_corr = max(c for t, c in recent)
        c = max(0.0, min(1.0, (max_corr - self.CORR_MIN) / (self.CORR_MAX - self.CORR_MIN)))
        
        # Hit quality: count of decodes, scaled to [0,1]
        hit_count = len(recent)
        h = min(1.0, hit_count / self.MAX_HITS)
        
        # Combined quality score
        Q = self.CORR_WEIGHT * c + self.HIT_WEIGHT * h
        
        return Q, max_corr, hit_count
    
    def compute_report(self, history: list) -> str:
        """
        Compute report string with hysteresis to prevent flickering.
        
        Returns: report string ("26", "27", "28", "29", "36", or "37")
        """
        Q, max_corr, hit_count = self.compute_quality(history)
        
        # Apply hysteresis: only change if Q crosses threshold by HYSTERESIS margin
        # Find what report the raw Q would give
        new_report = "26"
        for threshold, report in self.THRESHOLDS:
            if Q >= threshold:
                new_report = report
                break
        
        # Check if we should change (with hysteresis)
        if new_report != self.last_report:
            # Find the threshold for the new report
            new_threshold = 0.0
            for threshold, report in self.THRESHOLDS:
                if report == new_report:
                    new_threshold = threshold
                    break
            
            # Only change if we've crossed by hysteresis margin
            if new_report > self.last_report:
                # Moving up: Q must be above threshold + hysteresis
                if Q >= new_threshold + self.HYSTERESIS:
                    self.last_report = new_report
                    self.last_Q = Q
            else:
                # Moving down: Q must be below threshold - hysteresis  
                if Q <= new_threshold - self.HYSTERESIS:
                    self.last_report = new_report
                    self.last_Q = Q
        else:
            self.last_Q = Q
        
        return self.last_report
    
    def get_debug_info(self, history: list) -> str:
        """Get human-readable debug string."""
        Q, max_corr, hit_count = self.compute_quality(history)
        return f"Q={Q:.2f} (corr={max_corr:.2f}, hits={hit_count}/{self.MAX_HITS})"


# Global report calculator instance
_report_calc = ReportCalculator()


def compute_report_from_history(history: list, correlation: float = None) -> str:
    """
    Compute report from decode history, optionally adding a new decode first.
    
    Args:
        history: List of (timestamp, correlation) tuples (modified in place)
        correlation: If provided, add this decode to history first
    
    Returns:
        Report string ("26", "27", "28", "29", "36", or "37")
    """
    if correlation is not None:
        _report_calc.add_decode(history, correlation)
    return _report_calc.compute_report(history)


def get_report_debug_info(history: list) -> str:
    """Get debug info about current report calculation."""
    return _report_calc.get_debug_info(history)


# ---------------- audio I/O ----------------
def _device_info_safe(idx: int) -> Dict[str, Any]:
    try:
        return sd.query_devices(idx)
    except Exception:
        return {}

def list_audio_devices() -> List[Dict[str, Any]]:
    devs = []
    for i, d in enumerate(sd.query_devices()):
        devs.append({
            "index": i,
            "name": d.get("name", f"dev{i}"),
            "hostapi": d.get("hostapi", None),
            "max_input_channels": int(d.get("max_input_channels", 0) or 0),
            "max_output_channels": int(d.get("max_output_channels", 0) or 0),
            "default_samplerate": float(d.get("default_samplerate", 0.0) or 0.0),
        })
    return devs

def diagnose_device(dev_idx: int, is_input: bool, channel: int, sample_rate: int) -> str:
    """Generate diagnostic message for device configuration issues"""
    info = _device_info_safe(dev_idx)
    if not info:
        return f"Device {dev_idx} not found"
    
    name = info.get("name", f"device {dev_idx}")
    default_sr = info.get("default_samplerate", 48000)
    
    if is_input:
        max_ch = int(info.get("max_input_channels", 0) or 0)
        direction = "input"
    else:
        max_ch = int(info.get("max_output_channels", 0) or 0)
        direction = "output"
    
    issues = []
    if max_ch < 1:
        issues.append(f"no {direction} channels")
    elif channel > max_ch:
        issues.append(f"channel {channel} > max {max_ch}")
    
    if abs(sample_rate - default_sr) > 100:
        issues.append(f"rate {sample_rate}Hz != default {default_sr}Hz")
    
    if issues:
        return f"'{name}': {', '.join(issues)}"
    return f"'{name}': OK"

class AudioIO:
    """
    RX capture and TX playback.
    - RX records 2ch if available, then selects channel.
    - TX outputs stereo if available, routes mono packet to both/left/right.
    """
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.cfg = AudioConfig()
        self._tx_stream = None  # Track active TX stream for cleanup
        self.last_rx_rms_dbfs = None
        self.last_rx_peak_dbfs = None
        self.last_rx_clip = False
        self.last_tx_rms_dbfs = None
        self.last_tx_peak_dbfs = None
        self.last_tx_clip = False

        # --- Optional slow RX auto-level control (ALC) ---
        # Intentionally slow to avoid "pumping" on weak-signal decodes.
        self.auto_rx_enable: bool = False
        self.auto_rx_target_dbfs: float = -20.0   # target RMS level (dBFS)
        self.auto_rx_min_gain: float = 0.05
        self.auto_rx_max_gain: float = 2.00
        self.auto_rx_max_step_db: float = 0.6     # max change per update (dB)
        self._auto_rx_last_t: float = 0.0

    def apply_config(self, cfg: AudioConfig) -> None:
        self.cfg = cfg

    def set_gains(self, rx_gain: Optional[float] = None, tx_gain: Optional[float] = None) -> None:
        """Update gains without touching device/channel selection."""
        if rx_gain is not None:
            self.cfg.rx_gain = float(max(0.05, min(2.0, rx_gain)))
        if tx_gain is not None:
            self.cfg.tx_gain = float(max(0.05, min(1.0, tx_gain)))

    def set_auto_rx(self, enable: bool, target_dbfs: Optional[float] = None) -> None:
        self.auto_rx_enable = bool(enable)
        if target_dbfs is not None:
            try:
                self.auto_rx_target_dbfs = float(target_dbfs)
            except Exception:
                pass
    
    def stop_audio(self) -> None:
        """Stop any in-progress audio transmission immediately"""
        import sounddevice as sd
        try:
            # Stop all sounddevice playback
            sd.stop()
            if self._tx_stream is not None:
                self._tx_stream = None
        except Exception:
            pass

    def get_config_diagnostics(self) -> Dict[str, str]:
        """Get diagnostic information about current audio configuration"""
        rx_diag = diagnose_device(
            self.cfg.rx_device_index, 
            True, 
            self.cfg.rx_pick, 
            self.sample_rate
        )
        tx_diag = diagnose_device(
            self.cfg.tx_device_index, 
            False, 
            self.cfg.tx_pick, 
            self.sample_rate
        )
        return {
            "rx": rx_diag,
            "tx": tx_diag,
            "sample_rate": f"{self.sample_rate}Hz"
        }

    def _pick_in_channels(self) -> int:
        # Device selection uses numeric indices for sounddevice.
        info = _device_info_safe(self.cfg.rx_device_index)
        mic = int(info.get("max_input_channels", 1) or 1)
        # Clamp to what the device actually supports
        if mic >= 2:
            return 2
        return 1

    def _pick_out_channels(self) -> int:
        info = _device_info_safe(self.cfg.tx_device_index)
        moc = int(info.get("max_output_channels", 1) or 1)
        # Clamp to what the device actually supports
        if moc >= 2:
            return 2
        return 1

    def _validate_rx_config(self) -> bool:
        """Validate RX device configuration before use"""
        info = _device_info_safe(self.cfg.rx_device_index)
        if not info:
            return False
        
        max_in = int(info.get("max_input_channels", 0) or 0)
        if max_in < 1:
            return False
        
        # Check if requested channel is valid
        if self.cfg.rx_pick > max_in:
            return False
        
        return True

    def _validate_tx_config(self) -> bool:
        """Validate TX device configuration before use"""
        info = _device_info_safe(self.cfg.tx_device_index)
        if not info:
            return False
        
        max_out = int(info.get("max_output_channels", 0) or 0)
        if max_out < 1:
            return False
        
        # Check if requested channel is valid
        if self.cfg.tx_pick > max_out:
            return False
        
        return True

    def _rx_channels(self) -> int:
        """Return the number of channels to capture from RX device"""
        return self._pick_in_channels()

    def _route_tx(self, mono: np.ndarray) -> np.ndarray:
        """Route mono signal to stereo output if needed"""
        out_ch = self._pick_out_channels()
        if out_ch == 1:
            return mono.reshape(-1, 1)
        
        # Stereo output - route to selected channel
        stereo = np.zeros((len(mono), 2), dtype=np.float32)
        ch = int(self.cfg.tx_pick)
        ch = 1 if ch < 1 else min(ch, 2)
        stereo[:, ch - 1] = mono
        return stereo

    async def rx_capture_window(self, seconds: float) -> Optional[np.ndarray]:
        if not self.cfg.rx_enable:
            return None

        # Validate configuration first
        if not self._validate_rx_config():
            raise RuntimeError(
                f"Invalid RX config: device={self.cfg.rx_device_index}, "
                f"channel={self.cfg.rx_pick}. Check device has input channels."
            )

        n = int(self.sample_rate * float(seconds))
        if n <= 0:
            return None

        import sounddevice as sd

        # Get device info for better error messages
        info = _device_info_safe(self.cfg.rx_device_index)
        dev_name = info.get("name", f"device {self.cfg.rx_device_index}")
        
        # Record from selected device, then pick requested channel to mono float32.
        try:
            def _rec_blocking():
                data = sd.rec(
                    frames=n,
                    samplerate=self.sample_rate,
                    channels=self._rx_channels(),
                    dtype="float32",
                    device=self.cfg.rx_device_index,
                    blocking=True,
                )
                return data

            raw = await asyncio.to_thread(_rec_blocking)
        except Exception as e:
            err_msg = str(e)
            if "err='-50'" in err_msg or "-50" in err_msg:
                raise RuntimeError(
                    f"CoreAudio error -50 on '{dev_name}': "
                    f"Device likely doesn't support {self.sample_rate}Hz or "
                    f"{self._rx_channels()}ch config. Try different device/settings."
                )
            raise RuntimeError(f"RX capture failed on '{dev_name}': {e}")

        if raw is None:
            return None

        # raw shape: (n, ch)
        ch = int(self.cfg.rx_pick)
        ch = 1 if ch < 1 else ch
        ch = min(ch, raw.shape[1])
        mono = raw[:, ch - 1].astype(np.float32, copy=False)

        # Optional slow ALC: adjust rx_gain toward a target RMS level.
        # Uses PRE-gain RMS so the control behaves predictably.
        if self.auto_rx_enable and mono.size:
            try:
                now = time.monotonic()
                # limit update rate (~4 Hz)
                if (now - self._auto_rx_last_t) >= 0.25:
                    self._auto_rx_last_t = now
                    rms_pre = float(np.sqrt(np.mean(np.square(mono))))
                    eps = 1e-12
                    rms_pre_dbfs = 20.0 * math.log10(max(rms_pre, eps))
                    err_db = (self.auto_rx_target_dbfs - (rms_pre_dbfs + 20.0 * math.log10(max(float(self.cfg.rx_gain), eps))))
                    # clamp step
                    step_db = max(-self.auto_rx_max_step_db, min(self.auto_rx_max_step_db, err_db))
                    new_gain = float(self.cfg.rx_gain) * (10.0 ** (step_db / 20.0))
                    new_gain = max(self.auto_rx_min_gain, min(self.auto_rx_max_gain, new_gain))
                    self.cfg.rx_gain = float(new_gain)
            except Exception:
                pass

        # simple gain
        mono = (mono * float(self.cfg.rx_gain)).astype(np.float32, copy=False)
        
        # Level stats (post-gain), for UI metering
        try:
            peak = float(np.max(np.abs(mono))) if mono.size else 0.0
            rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
            eps = 1e-12
            self.last_rx_peak_dbfs = 20.0 * math.log10(max(peak, eps))
            self.last_rx_rms_dbfs = 20.0 * math.log10(max(rms, eps))
            self.last_rx_clip = bool(peak >= 0.999)
        except Exception:
            pass

        return mono

    def _repeat_to_length(self, wav: np.ndarray, n: int) -> np.ndarray:
        if len(wav) >= n:
            return wav[:n].copy()
        reps = int(math.ceil(n / len(wav)))
        return np.tile(wav, reps)[:n].copy()

    async def tx_play_period(self, wav: np.ndarray, seconds: float) -> None:
        # Repeat the packet waveform to fill the whole slot, then play on selected device/channel.
        if not self.cfg.tx_enable:
            return

        # Validate configuration first
        if not self._validate_tx_config():
            raise RuntimeError(
                f"Invalid TX config: device={self.cfg.tx_device_index}, "
                f"channel={self.cfg.tx_pick}. Check device has output channels."
            )

        n = int(self.sample_rate * float(seconds))
        if n <= 0:
            return
        sig = self._repeat_to_length(wav.astype(np.float32), n)

        out = self._route_tx(sig) * float(self.cfg.tx_gain)

        # Level stats (post-gain), for UI metering
        try:
            peak = float(np.max(np.abs(out))) if out.size else 0.0
            rms = float(np.sqrt(np.mean(np.square(out)))) if out.size else 0.0
            eps = 1e-12
            self.last_tx_peak_dbfs = 20.0 * math.log10(max(peak, eps))
            self.last_tx_rms_dbfs = 20.0 * math.log10(max(rms, eps))
            self.last_tx_clip = bool(peak >= 0.999)
        except Exception:
            pass


        import sounddevice as sd

        # Get device info for better error messages
        info = _device_info_safe(self.cfg.tx_device_index)
        dev_name = info.get("name", f"device {self.cfg.tx_device_index}")

        try:
            # Store stream reference before playing
            self._tx_stream = "active"
            try:
                await asyncio.to_thread(
                    sd.play,
                    out,
                    samplerate=self.sample_rate,
                    blocking=True,
                    device=self.cfg.tx_device_index,
                )
            except asyncio.CancelledError:
                # Task was cancelled - stop audio immediately
                sd.stop()
                raise
            finally:
                self._tx_stream = None
        except asyncio.CancelledError:
            # Re-raise cancellation
            raise
        except Exception as e:
            err_msg = str(e)
            if "err='-50'" in err_msg or "-50" in err_msg:
                raise RuntimeError(
                    f"CoreAudio error -50 on '{dev_name}': "
                    f"Device likely doesn't support {self.sample_rate}Hz or "
                    f"{self._pick_out_channels()}ch config. Try different device/settings."
                )
            raise RuntimeError(f"TX play failed on '{dev_name}': {e}")
class MSK2KBackend:
    def __init__(self):
        self.sample_rate = 48000
        self.tx = m.MSK2KTransmitter(sample_rate=self.sample_rate)
        self.rx = m.MSK2KReceiver(sample_rate=self.sample_rate)
        self.cfg = Config()
        self.rt = Runtime()
        self.audio = AudioIO(sample_rate=self.sample_rate)

        self._ws_clients: Set[web.WebSocketResponse] = set()
        self._task: Optional[asyncio.Task] = None
        self.adif_path: Path = get_adif_path()


    def _mark_qso_start(self, partner: str) -> None:
        partner = (partner or "").strip().upper()
        if not partner:
            return
        if self.rt.qso_partner != partner or not self.rt.qso_start_date or not self.rt.qso_start_time:
            self.rt.qso_partner = partner
            self.rt.qso_start_date = utc_date_adif()
            self.rt.qso_start_time = utc_time_adif()
    async def _complete_qso(self, reason: str = "early_complete") -> None:
        """Force QSO completion immediately and stop any further TX."""
        # Prevent any further TX immediately
        self.rt.qso_state = QSO_DONE
        self.rt.next_tx_text = "(done)"
        self.rt.priv_73_count = 0

        # Persist + broadcast completion
        try:
            qso = await self._finalize_qso_and_log()
            await self.broadcast({"type": "qso_complete", "ts": utc_clock(), "partner": self.cfg.their_call, "qso": qso, "reason": reason})
        except Exception as e:
            await self.info(f"QSO complete (no log due to error): {e}")
        finally:
            # Reset markers and unlock for next QSO
            self._reset_qso_markers()
            self.cfg.their_call = ""
            self.rt.observed_remote_slot = None
            self.rt.report_locked = False



    def _reset_qso_markers(self) -> None:
        self.rt.qso_start_date = ""
        self.rt.qso_start_time = ""
        self.rt.qso_partner = ""

    def _build_qso_record(self, partner: str) -> Dict[str, str]:
        partner = (partner or self.cfg.their_call or "").strip().upper()
        start_date = self.rt.qso_start_date or utc_date_adif()
        start_time = self.rt.qso_start_time or utc_time_adif()
        end_date = utc_date_adif()
        end_time = utc_time_adif()
        band = band_adif_from_inputs(self.cfg.band_sel, self.cfg.band_custom)
        freq = (self.cfg.freq_mhz or '').strip()
        return {
            "qso_date": start_date,
            "time_on": start_time,
            "time_off": end_time,
            "call": partner,
            "rst_sent": str(self.rt.my_report or ""),
            "rst_rcvd": str(self.rt.received_report or ""),
            "band": band,
            "freq": freq,
            "mode": "MSK2K",
            "submode": "MSK2K",
        }

    async def _finalize_qso_and_log(self) -> Dict[str, str]:
        partner = (self.cfg.their_call or "").strip().upper()
        qso = self._build_qso_record(partner)
        try:
            append_adif(self.adif_path, qso)
        except Exception as e:
            # Don't crash the QSO flow if file writing fails
            await self.info(f"ADIF log write failed: {e}")
        return qso

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        dead = []
        for ws in list(self._ws_clients):
            try:
                await ws.send_str(json.dumps(payload))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._ws_clients.discard(ws)

    async def info(self, msg: str) -> None:
        await self.broadcast({"type": "info", "message": msg})

    def _compute_next_tx_text(self) -> str:
        my = self.cfg.my_call
        their = self.cfg.their_call

        # CALLING_CQ: Send CQ
        if self.rt.qso_state == QSO_CALLING_CQ:
            return f"CQ de {my}"
        
        # CALLING_STN: Send call WITHOUT report (cold call - they haven't heard us yet)
        if self.rt.qso_state == QSO_CALLING_STN:
            return f"{their} de {my}" if their else f"CQ de {my}"

        # HAVE_CALLS: We answered a CQ, send our call with report
        if self.rt.qso_state == QSO_HAVE_CALLS:
            return f"{their} de {my} {self.rt.my_report}" if their else f"CQ de {my}"

        # PRIVATE: Multi-step exchange
        if self.rt.qso_state == QSO_PRIVATE:
            # Count 0: Send R<rpt> - acknowledge their report using the actual report we received
            if self.rt.priv_73_count == 0:
                return f"R{self.rt.my_report}"
            
            # Count 1: Send RR
            if self.rt.priv_73_count == 1:
                return "RR"
            
            # Count 2+: Send 73
            return "73"

        # DONE
        if self.rt.qso_state == QSO_DONE:
            return "(done)"

        return ""  # LISTEN/IDLE

    def _make_tx_audio(self, msg: str) -> np.ndarray:
        my = self.cfg.my_call
        their = self.cfg.their_call

        if not msg:
            return self.tx.generate_cq(my)  # fallback

        # CQ messages
        if msg.startswith("CQ"):
            return self.tx.generate_cq(my)

        # Format 2 short messages (R26-R29, R36-R37, RR, RRR, 73)
        if msg in ("R26", "R27", "R28", "R29", "R36", "R37", "RR", "RRR", "73"):
            if their:
                return self.tx.generate_format2_message(my, their, msg)
            else:
                # No target callsign, fall back to CQ
                return self.tx.generate_cq(my)

        # DEBUG: Log what we're processing
        import asyncio
        import math
        asyncio.create_task(self.info(f"DEBUG _make_tx_audio: msg='{msg}', my='{my}', their='{their}', has_de={' de ' in msg}"))

        # Format 1 messages with " de "
        if " de " in msg and their:
            # Check if message contains a report number
            parts = msg.split()
            report = None
            for part in parts:
                if part.isdigit() and len(part) == 2 and int(part) in (26, 27, 28, 29, 36, 37):
                    report = part
                    break
            
            if report:
                # Has report - use generate_call_with_report
                return self.tx.generate_call_with_report(my, their, report)
            else:
                # No report - use generate_cold_call (before hearing them)
                return self.tx.generate_cold_call(my, their)

        # Generic call with report
        if their:
            return self.tx.generate_call_with_report(my, their, self.rt.my_report)
        
        # Fallback
        return self.tx.generate_cq(my)

    def _effective_tx_slot(self) -> int:
        """If replying to an observed remote slot, choose the opposite."""
        base = 0 if self.cfg.tx_slot == "A" else 1
        if self.rt.observed_remote_slot in ("A", "B"):
            remote = 0 if self.rt.observed_remote_slot == "A" else 1
            return 1 - remote
        return base

    async def _on_tx_slot_start(self) -> None:
        # Check if we're still running before any TX attempt
        if not self.rt.running:
            return
            
        # only TX if state expects it
        if self.rt.qso_state not in (QSO_CALLING_CQ, QSO_CALLING_STN, QSO_HAVE_CALLS, QSO_PRIVATE):
            self.rt.current_activity = "RX"
            return

        self.rt.next_tx_text = self._compute_next_tx_text()

        # Stamp QSO start time on the FIRST *actual transmission* to a specific station.
        # (Not when we first decode them, and not when the QSO completes.)
        if self.rt.qso_state in (QSO_CALLING_STN, QSO_HAVE_CALLS, QSO_PRIVATE):
            partner = (self.cfg.their_call or '').strip().upper()
            # Only mark once, and only if we are about to TX something meaningful
            if partner and not self.rt.qso_start_time and self.rt.next_tx_text:
                self.rt.qso_partner = partner
                self.rt.qso_start_date = utc_date_adif()
                self.rt.qso_start_time = utc_time_adif()

        msg = self.rt.next_tx_text

        # --- TX timeout safety (minutes, counted from first *actual* TX) ---
        if msg:
            if self.rt.first_tx_epoch is None:
                self.rt.first_tx_epoch = time.time()
            if (not self.cfg.tx_timeout_override) and int(self.cfg.tx_timeout_min or 0) > 0:
                limit_sec = int(self.cfg.tx_timeout_min) * 60
                elapsed = time.time() - float(self.rt.first_tx_epoch or 0.0)
                if elapsed >= limit_sec:
                    await self.info(f"TX timeout: reached {self.cfg.tx_timeout_min} min limit; forcing LISTEN")
                    await self.broadcast({"type":"info","message": f"⚠️ TX timeout reached ({self.cfg.tx_timeout_min} min) - forced LISTEN"})
                    # Return to receive-only without killing the scheduler
                    self.cfg.mode = "LISTEN"
                    self.rt.qso_state = QSO_LISTEN
                    self.cfg.their_call = ""
                    self.rt.observed_remote_slot = None
                    self.rt.report_locked = False
                    self.rt.priv_73_count = 0
                    self.rt.next_tx_text = ""
                    self._reset_qso_markers()
                    self.rt.first_tx_epoch = None
                    self.rt.current_activity = "RX"
                    return
        
        # Determine message type for logging
        msg_type = "CQ"
        if msg in ("R26", "R27", "R28", "R29", "R36", "R37", "RR", "RRR", "73"):
            msg_type = "Format2"
        elif " de " in msg and any(ch.isdigit() for ch in msg):
            msg_type = "Format1+report"
        
        try:
            self.rt.current_activity = "TX"
            audio = self._make_tx_audio(msg)
            await self.audio.tx_play_period(audio, float(self.cfg.period_len))
            await self.broadcast({"type": "tx", "ts": utc_clock(), "text": msg, "msg_type": msg_type})
            await self.info(f"TX [{msg_type}]: {msg}")
        except asyncio.CancelledError:
            # Task cancelled - let it propagate to stop the scheduler
            raise
        except Exception as e:
            await self.info(str(e))
        finally:
            self.rt.current_activity = "RX"

        # Handle 73 counting and QSO completion
        if self.rt.qso_state == QSO_PRIVATE and msg == "73":
            self.rt.priv_73_count += 1
            if self.rt.priv_73_count >= 7:  # 5 periods of 73 (count starts at 2, so 2+5=7)
                await self.info("Auto QSO: Completed 5x 73 timeout, QSO done")
                
                # IMMEDIATELY set to DONE to prevent any further TX
                self.rt.qso_state = QSO_DONE
                self.rt.priv_73_count = 0
                
                # Persist and broadcast QSO complete so UI can clear and log
                qso = await self._finalize_qso_and_log()
                await self.broadcast({"type": "qso_complete", "ts": utc_clock(), "partner": self.cfg.their_call, "qso": qso})
                self._reset_qso_markers()
                
                # Clear partner and unlock report for next QSO
                self.cfg.their_call = ""
                self.rt.observed_remote_slot = None
                self.rt.report_locked = False  # Unlock for next QSO
                self.rt.next_tx_text = ""  # Clear pending TX
                
                # After a short delay, return to original mode
                if self.cfg.mode == "CQ":
                    self.rt.qso_state = QSO_CALLING_CQ
                else:
                    self.rt.qso_state = QSO_LISTEN
                await self.info(f"Returned to {self.rt.qso_state} mode")
                return

    async def _handle_decode(self, res: Dict[str, Any]) -> None:
        text = (res.get("text") or "").strip()
        # GLOBAL_RPT_LATCH: latch received report from any "Rxx" token immediately for logging
        try:
            m_r = re.search(r"\bR(\d{2})\b", text)
            if m_r:
                self.rt.received_report = m_r.group(1)
                await self.info(f"Latched received report: {self.rt.received_report} from '{m_r.group(0)}'")
        except Exception:
            pass
        if not text:
            return

        # Track ALL successful decodes in history for quality-based reports
        corr = res.get("sync_correlation", 0.0)
        if corr > 0:
            _report_calc.add_decode(self.rt.decode_history, corr)

        # Slot observation: the slot we are *currently* in when decoded is RX slot for us,
        # so remote is transmitting in that slot. Record it so we transmit opposite.
        if res.get("slot") in ("A", "B"):
            self.rt.observed_remote_slot = res["slot"]

        # Basic QSO state triggers
        my = self.cfg.my_call
        their = self.cfg.their_call
        
        # Handle CQ messages - always show as public
        if text.startswith("CQ") and " de " in text:
            parts = text.split()
            try:
                de_i = parts.index("de")
                caller = parts[de_i + 1].strip().upper()

            except Exception:
                caller = ""

            # EARLY_COMPLETE_73_CQ: if partner shows up again (CQ) during 73 loop after at least one 73 TX, end QSO early
            try:
                if self.rt.qso_state == QSO_PRIVATE and getattr(self.rt, "qso_partner", ""):
                    if getattr(self.rt, "priv_73_count", 0) >= 3:
                        if caller and caller.upper() == self.rt.qso_partner.upper():
                            await self.info(f"Early complete: partner {caller} CQ during 73 loop -> complete now")
                            await self._complete_qso("partner_seen_again")
                            return
            except Exception:
                pass
            await self.broadcast({"type": "rx", "ts": utc_clock(), "text": text, "visibility": "public",
                                  "snr": res.get("sync_correlation"), "method": res.get("method"), "from": caller})
            return

        # Determine visibility - private if addressed to us or Format 2
        visibility = "public"
        is_format2 = res.get("format") in (2, "format2")
        if my in text or is_format2:
            visibility = "private"

        await self.broadcast({"type": "rx", "ts": utc_clock(), "text": text, "visibility": visibility,
                              "snr": res.get("sync_correlation"), "method": res.get("method"), "format": 2 if is_format2 else 1})
        # EARLY_COMPLETE_73_FMT1: if partner appears again in any Format1 message during 73 loop after >=1 TX 73, end QSO early
        try:
            if self.rt.qso_state == QSO_PRIVATE and getattr(self.rt, "qso_partner", ""):
                if getattr(self.rt, "priv_73_count", 0) >= 3 and not is_format2:
                    partner = self.rt.qso_partner.upper()
                    if partner and partner in text.upper():
                        await self.info(f"Early complete: partner {partner} seen again in FMT1 during 73 loop -> complete now")
                        await self._complete_qso("partner_seen_again")
                        return
        except Exception:
            pass



        # Auto-sequencing state machine
        if not self.cfg.auto_seq:
            return
        
        # Extract report number if present (26-29 or 36-37)
        def extract_report(txt):
            valid_reports = {'26', '27', '28', '29', '36', '37'}
            for word in txt.split():
                # Check for R-report format first (R27, etc) before other checks
                if word.startswith('R') and len(word) == 3 and word[1:] in valid_reports:
                    return word[1:]  # R27 -> 27
                # Skip words that contain '/' (callsign suffixes like SA3WDB/7)
                if '/' in word:
                    continue
                # Skip words with letters (callsigns) - but we already handled R-reports above
                if any(c.isalpha() for c in word):
                    continue
                # Check for exact report match (pure numbers)
                if word in valid_reports:
                    return word
            return None
        
        # SCENARIO 3: Someone calls ME directly when I'm listening
        # RX: "GW4WND de DJ5HG" (no report) → I reply with MY report
        # OR: "GW4WND de DJ5HG 26" (with report) → I reply with MY report
        if self.rt.qso_state in (QSO_LISTEN, QSO_IDLE):
            if my in text and " de " in text:
                toks = text.split()
                try:
                    my_idx = next((i for i, t in enumerate(toks) if t == my), -1)
                    de_idx = next((i for i, t in enumerate(toks) if t == "de"), -1)
                    # Format: "MY de THEIR" or "MY de THEIR <rpt>"
                    if my_idx != -1 and de_idx != -1 and my_idx < de_idx and de_idx + 1 < len(toks):
                        their_call = toks[de_idx + 1].strip().upper()
                        if their_call:
                            # They called me (with or without report)
                            # Compute and LOCK MY report for this QSO
                            if not self.rt.report_locked:
                                # Latch report from the first Q% we derive from the remote station during this QSO
                                remote_qpct = 0
                                try:
                                    remote_qpct = int(round(max(0.0, min(1.0, corr)) * 100))
                                except Exception:
                                    remote_qpct = 0
                                self.rt.my_report = report_from_qpct(remote_qpct)
                                self.rt.report_locked = True
                                debug_info = f"Q={remote_qpct}% -> RPT {self.rt.my_report}"
                            else:
                                debug_info = ""
                            # I need to reply with MY report
                            self.cfg.their_call = their_call
                            rpt = extract_report(text)
                            if rpt:
                                # They sent a report - both calls + both reports exchanged
                                # Go directly to Format 2 exchange
                                self.rt.received_report = rpt  # Store the report they sent
                                self.rt.qso_state = QSO_PRIVATE  # Will send R<my_report> next
                                self.rt.priv_73_count = 0
                                await self.info(f"Auto QSO: {their_call} called me with report {rpt} [{debug_info}], will send R{self.rt.my_report}")
                            else:
                                # They sent NO report (cold call) - need to exchange reports in Format 1
                                # Go to HAVE_CALLS to send Format 1 with MY report
                                self.rt.qso_state = QSO_HAVE_CALLS  # Will send "THEIR de MY <my_report>" next
                                self.rt.priv_73_count = 0
                                await self.info(f"Auto QSO: {their_call} called me without report [{debug_info}], will reply with {self.rt.my_report}")
                            return
                except Exception:
                    pass
            
        # STATE: CALLING_STN - I called someone with report, waiting for their response  
        # RX: "GW4WND de DJ5HG 27" (Format 1) → I send R27 (Format 2)
        if self.rt.qso_state == QSO_CALLING_STN and their:
            if my in text and " de " in text and their in text:
                rpt = extract_report(text)
                if rpt:
                    # Compute and LOCK MY report for this QSO (first remote Q%)
                    if not self.rt.report_locked:
                        remote_qpct = 0
                        try:
                            remote_qpct = int(round(max(0.0, min(1.0, corr)) * 100))
                        except Exception:
                            remote_qpct = 0
                        self.rt.my_report = report_from_qpct(remote_qpct)
                        self.rt.report_locked = True
                    # They sent Format 1 with their report
                    # Both calls + both reports now exchanged → switch to Format 2
                    self.rt.received_report = rpt
                    self.rt.qso_state = QSO_PRIVATE
                    self.rt.priv_73_count = 0  # Next: send R<rpt>
                    await self.info(f"Auto QSO: Received report {rpt} from {their} [Q={remote_qpct}% -> RPT {self.rt.my_report}], will send R{self.rt.my_report}")
                    return
        
        # STATE: CALLING_CQ - SCENARIO 1: Someone answered my CQ
        # RX: "GW4WND de DJ5HG 26" → I send R26 immediately (Format 2)
        if self.rt.qso_state == QSO_CALLING_CQ:
            if my in text and " de " in text:
                toks = text.split()
                try:
                    my_idx = next((i for i, t in enumerate(toks) if t == my), -1)
                    de_idx = next((i for i, t in enumerate(toks) if t == "de"), -1)
                    if my_idx != -1 and de_idx != -1 and my_idx < de_idx and de_idx + 1 < len(toks):
                        their_call = toks[de_idx + 1].strip().upper()
                        rpt = extract_report(text)
                        if their_call and rpt:
                            # Someone answered CQ with report
                            # Compute and LOCK MY report for this QSO
                            if not self.rt.report_locked:
                                remote_qpct = 0
                                try:
                                    remote_qpct = int(round(max(0.0, min(1.0, corr)) * 100))
                                except Exception:
                                    remote_qpct = 0
                                self.rt.my_report = report_from_qpct(remote_qpct)
                                self.rt.report_locked = True
                                debug_info = f"Q={remote_qpct}% -> RPT {self.rt.my_report}"
                            else:
                                debug_info = ""
                            # Both calls + report now exchanged → go to Format 2
                            self.cfg.their_call = their_call
                            self.rt.received_report = rpt
                            self.rt.qso_state = QSO_PRIVATE
                            self.rt.priv_73_count = 0  # Next: send R<rpt>
                            await self.info(f"Auto QSO: Received report {rpt} from {their_call} [{debug_info}], will send R{self.rt.my_report}")
                            return
                except Exception:
                    pass
        
        # STATE: HAVE_CALLS - I sent "THEIR de MY 26" (both calls + my report)
        # They should acknowledge with R26 (Format 2)
        # RX: "R26" → I send RR
        if self.rt.qso_state == QSO_HAVE_CALLS and their:
            # Looking for Format 2: "R<rpt>"
            is_format2 = res.get("format") in (2, "format2")
            if is_format2 and text.startswith("R") and len(text) == 3:
                # They acknowledged my report with R<rpt>
                self.rt.qso_state = QSO_PRIVATE
                self.rt.priv_73_count = 1  # Next: send RR
                await self.info(f"Auto QSO: Received {text}, will send RR")
                return
        
        # STATE: PRIVATE - Exchange R<rpt>, RR, and 73s
        if self.rt.qso_state == QSO_PRIVATE:
            # If we're at count 0, we sent R<rpt> and are waiting for RR
            if self.rt.priv_73_count == 0:
                await self.info(f"[DEBUG] PRIVATE count=0, checking text='{text}', format={res.get('format')}, is_format2={is_format2}")
                if text.startswith("R") and len(text) == 3 and text[1:].isdigit():
                    # Store the report we received (R37 -> 37) for logging
                    self.rt.received_report = text[1:]
                    self.rt.priv_73_count = 1  # Next we will send RR
                    await self.info(f"Auto QSO: Received {text}, will send RR")
                    return

                if "RR" in text:
                    self.rt.priv_73_count = 2  # Move directly to sending 73
                    await self.info("Auto QSO: Received RR, will send 73")
                    return
            
            # If we're at count 1, we're sending RR and waiting for their 73
            elif self.rt.priv_73_count == 1:
                if "73" in text:
                    self.rt.priv_73_count = 2  # Move to sending 73
                    await self.info("Auto QSO: Received 73, will send 73")
                    return
            
            # If we're at count 2+, we're exchanging 73s
            elif self.rt.priv_73_count >= 2:
                # If we're at count 2+, we're exchanging 73s.
                # EARLY COMPLETE (after we've TX'd at least one 73):
                # If we see the partner callsign appear again in any Format-1 style line,
                # it strongly suggests they've moved on (e.g. "CQ ..." or "<NEW> ..."),
                # so we can cancel remaining 73 repeats and complete the QSO immediately.
                #
                # NOTE: In MSK2K we may *display* "de" locally, but it is not necessarily sent
                # over the air; therefore we do NOT key off the presence of "de" here.
                #
                # priv_73_count semantics:
                #   - set to 2 when we enter 73 exchange (before first 73 TX)
                #   - incremented on each TX of "73"
                #   - therefore, priv_73_count >= 3 means we've TX'd at least one 73.
                if self.rt.priv_73_count >= 3 and their:
                    try:
                        partner_u = their.upper()
                        text_u = text.upper()
                        # Match partner as a standalone token (callsigns can include /)
                        if re.search(rf"\b{re.escape(partner_u)}\b", text_u):
                            await self.info(
                                f"Auto QSO: Partner {their} appeared again during 73 loop "
                                f"(after 1+ 73 TX) — EARLY COMPLETE! ✓"
                            )

                            # IMMEDIATELY set to DONE to prevent any further TX
                            self.rt.qso_state = QSO_DONE
                            self.rt.priv_73_count = 0

                            # Persist and broadcast QSO complete so UI can clear and log
                            qso = await self._finalize_qso_and_log()
                            await self.broadcast({"type": "qso_complete", "ts": utc_clock(), "partner": self.cfg.their_call, "qso": qso})
                            self._reset_qso_markers()

                            # Clear partner and unlock report for next QSO
                            partner = self.cfg.their_call
                            self.cfg.their_call = ""
                            self.rt.observed_remote_slot = None
                            self.rt.report_locked = False  # Unlock for next QSO
                            self.rt.next_tx_text = ""  # Clear pending TX

                            # After completing, return to appropriate mode
                            if self.cfg.mode == "CQ":
                                self.rt.qso_state = QSO_CALLING_CQ
                                await self.info("Returning to CQ mode")
                            else:
                                self.rt.qso_state = QSO_LISTEN
                                await self.info("Returning to LISTEN mode")

                            return
                    except Exception:
                        pass

                if "73" in text:
                    # We've sent 73 and received 73 - QSO is complete!
                    await self.info("Auto QSO: Received 73 - QSO COMPLETE! ✓")
                    
                    # IMMEDIATELY set to DONE to prevent any further TX
                    self.rt.qso_state = QSO_DONE
                    self.rt.priv_73_count = 0
                    
                    # Persist and broadcast QSO complete so UI can clear and log
                    qso = await self._finalize_qso_and_log()
                    await self.broadcast({"type": "qso_complete", "ts": utc_clock(), "partner": self.cfg.their_call, "qso": qso})
                    self._reset_qso_markers()
                    
                    # Clear partner and unlock report for next QSO
                    partner = self.cfg.their_call
                    self.cfg.their_call = ""
                    self.rt.observed_remote_slot = None
                    self.rt.report_locked = False  # Unlock for next QSO
                    self.rt.next_tx_text = ""  # Clear pending TX
                    
                    # After completing, return to appropriate mode
                    if self.cfg.mode == "CQ":
                        self.rt.qso_state = QSO_CALLING_CQ
                        await self.info("Returning to CQ mode")
                    else:
                        self.rt.qso_state = QSO_LISTEN
                        await self.info("Returning to LISTEN mode")
                    
                    return

    async def _on_rx_slot_start(self) -> None:
        if not self.cfg.audio.rx_enable:
            return
        
        # CRITICAL: Capture the slot NOW, at the START of RX period
        # NOT after decode completes (which may be in the next period!)
        capture_time = time.time()
        capture_slot = period_label(self._current_slot_for_time(capture_time, self.cfg.period_len))
        
        try:
            audio = await self.audio.rx_capture_window(float(self.cfg.period_len))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self.info(str(e))
            return
        
        # Track timing for diagnostics
        audio_captured_time = time.time()
        audio_capture_duration = audio_captured_time - capture_time

        try:
            # decode() signature: decode(audio, my_callsign=..., partner_callsign=...)
            decode_start = time.time()
            res = self.rx.decode(audio, my_callsign=self.cfg.my_call, partner_callsign=self.cfg.their_call or None)
            decode_end = time.time()
            decode_duration = decode_end - decode_start
            total_lag = decode_end - capture_time
            
            if res.get("success"):
                # Use the slot captured at RX start, not current time
                res["slot"] = capture_slot
                # Log timing info
                await self.info(f"⏱ RX timing: capture={audio_capture_duration:.1f}s decode={decode_duration:.1f}s total_lag={total_lag:.1f}s slot={capture_slot}")
                await self._handle_decode(res)
            else:
                # Log decode failures for debugging
                error_msg = res.get("error", "Unknown decode error")
                await self.info(f"Decode failed: {error_msg} (decode took {decode_duration:.1f}s)")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self.info(f"RX decode error: {e}")

    def _current_slot_for_time(self, now: float, period_len: int) -> int:
        slot_num = int(now // period_len)
        return slot_num % 2

    async def run(self) -> None:
        await self.info("Scheduler started")
        last_slot_num: Optional[int] = None

        while self.rt.running:
            now = time.time()
            slot_num = int(now // self.cfg.period_len)
            slot = slot_num % 2  # 0=A,1=B
            tx_slot = self._effective_tx_slot()

            self.rt.next_tx_text = self._compute_next_tx_text() if self.cfg.auto_seq else self.rt.next_tx_text

            # Show TX if we're in a TX-capable state AND it's our TX slot
            is_tx_state = self.rt.qso_state in (QSO_CALLING_CQ, QSO_CALLING_STN, QSO_HAVE_CALLS, QSO_PRIVATE)
            is_tx_slot = (slot == tx_slot)
            rx_tx = "TX" if (is_tx_state and is_tx_slot) else "RX"

            await self.broadcast({
                "type": "tick",
                "clock": utc_clock(),
                "period_label": f"P{period_label(slot)}",
                "rx_tx": rx_tx,
                "qso_state": self.rt.qso_state,
                "next_tx": self.rt.next_tx_text or "(none)",
                "their_call": self.cfg.their_call,
                "tx_slot": "A" if tx_slot == 0 else "B",
                "remote_slot": self.rt.observed_remote_slot or "",
            })

            if last_slot_num is None or slot_num != last_slot_num:
                last_slot_num = slot_num
                if slot == tx_slot:
                    await self._on_tx_slot_start()
                else:
                    await self._on_rx_slot_start()

            await asyncio.sleep(0.25)

        await self.info("Scheduler stopped")

    async def start(self, cfg: Config) -> None:
        self.cfg = cfg
        self.audio.apply_config(cfg.audio)

        # initial state based on mode
        st = QSO_LISTEN
        if cfg.mode == "CQ":
            st = QSO_CALLING_CQ
        elif cfg.mode == "CALL":
            st = QSO_CALLING_STN
        elif cfg.mode == "REPLY":
            st = QSO_HAVE_CALLS if cfg.their_call else QSO_LISTEN

        self.rt = Runtime(running=True, qso_state=st, next_tx_text="", priv_73_count=0, observed_remote_slot=None, received_report="27", my_report="26", report_locked=False, decode_history=[])
        
        # Latch outgoing report immediately when replying from a clicked decode
        if cfg.mode == "REPLY" and cfg.their_call and cfg.initial_corr is not None:
            try:
                c = float(cfg.initial_corr)
                c = max(0.0, min(1.0, c))
                qpct = int(round(c * 100))
                self.rt.my_report = report_from_qpct(qpct)
                self.rt.report_locked = True
                await self.info(f"Report latched from click: Q={qpct}% -> {self.rt.my_report}")
            except Exception:
                pass

        self.rt.next_tx_text = self._compute_next_tx_text()

        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.create_task(self.run())

        await self.info(f"Started: mode={cfg.mode} my={cfg.my_call} their={cfg.their_call or '-'} "
                        f"period={cfg.period_len}s pref_tx={cfg.tx_slot} eff_tx={('A' if self._effective_tx_slot()==0 else 'B')} "
                        f"auto={cfg.auto_seq} rx_dev={cfg.audio.rx_device_index} tx_dev={cfg.audio.tx_device_index} "
                        f"rx_pick={cfg.audio.rx_pick+1} tx_pick={cfg.audio.tx_pick} tx_gain={cfg.audio.tx_gain}")

    async def stop(self) -> None:
        self.rt.running = False
        # Stop any in-progress audio transmission
        self.audio.stop_audio()
        self._reset_qso_markers()
        if self._task and not self._task.done():
            self._task.cancel()
        self.rt = Runtime(running=False, qso_state=QSO_IDLE)

backend = MSK2KBackend()

# ---------------- aiohttp server ----------------
async def handle_index(_: web.Request) -> web.Response:
    ui = HERE / "msk2k_audio_qso_ui_Q12.html"
    if not ui.exists():
        ui = HERE / "msk2k_audio_qso_ui_WITH_QSOLOG.html"
    return web.FileResponse(ui)

async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    backend._ws_clients.add(ws)
    await backend.info("UI connected")
    try:
        async for _ in ws:
            pass
    finally:
        backend._ws_clients.discard(ws)
    return ws

async def api_devices(_: web.Request) -> web.Response:
    return web.json_response({"devices": list_audio_devices(), "sample_rate": backend.sample_rate})

async def api_audio_set(request: web.Request) -> web.Response:
    data = await request.json()
    def _to_int(x):
        try: return int(x)
        except Exception: return None

    rx_dev = _to_int(data.get("rx_device"))
    tx_dev = _to_int(data.get("tx_device"))
    rx_pick = _to_int(data.get("rx_pick"))
    if rx_pick is None:
        rx_pick = 0
    tx_pick = _to_int(data.get("tx_pick"))
    if tx_pick is None:
        tx_pick = 0
    rx_en = bool(data.get("rx_enable", True))
    tx_en = bool(data.get("tx_enable", True))
    tx_gain = float(data.get("tx_gain", 0.60))
    tx_gain = max(0.05, min(1.0, tx_gain))
    rx_gain = float(data.get("rx_gain", 1.00))
    rx_gain = max(0.05, min(2.0, rx_gain))


    backend.cfg.audio = AudioConfig(
        rx_device_index=rx_dev,
        tx_device_index=tx_dev,
        rx_pick=rx_pick + 1,  # Convert from 0-based to 1-based
        tx_pick=tx_pick + 1,  # Convert from 0-based to 1-based
        rx_enable=rx_en,
        tx_enable=tx_en,
        rx_gain=rx_gain,
        tx_gain=tx_gain,
    )
    backend.audio.apply_config(backend.cfg.audio)
    
    # Return diagnostics to help user identify issues
    diag = backend.audio.get_config_diagnostics()

    return web.json_response({
        "ok": True,
        "diagnostics": diag
    })


async def api_gain_set(request: web.Request) -> web.Response:
    """Update RX/TX gains (and optional slow RX ALC) without reconfiguring devices."""
    data = await request.json()
    rx_gain = data.get("rx_gain", None)
    tx_gain = data.get("tx_gain", None)
    auto_rx = data.get("auto_rx", None)
    auto_target = data.get("auto_target_dbfs", None)

    try:
        if rx_gain is not None:
            rx_gain = float(rx_gain)
        if tx_gain is not None:
            tx_gain = float(tx_gain)
    except Exception:
        return web.json_response({"ok": False, "error": "invalid gain"}, status=400)

    backend.audio.set_gains(rx_gain=rx_gain, tx_gain=tx_gain)
    # Keep config object in sync
    backend.cfg.audio.rx_gain = backend.audio.cfg.rx_gain
    backend.cfg.audio.tx_gain = backend.audio.cfg.tx_gain

    if auto_rx is not None:
        backend.audio.set_auto_rx(bool(auto_rx), target_dbfs=auto_target)

    return web.json_response({
        "ok": True,
        "rx_gain": float(backend.audio.cfg.rx_gain),
        "tx_gain": float(backend.audio.cfg.tx_gain),
        "auto_rx": bool(backend.audio.auto_rx_enable),
        "auto_target_dbfs": float(backend.audio.auto_rx_target_dbfs),
    })

async def api_diagnostics(_: web.Request) -> web.Response:
    """Return current audio configuration diagnostics"""
    diag = backend.audio.get_config_diagnostics()
    return web.json_response({"diagnostics": diag})



async def api_levels(request: web.Request) -> web.Response:
    """Return latest RX/TX audio level stats for UI meters."""
    a = backend.audio
    return web.json_response({
        "ok": True,
        "rx": {
            "rms_dbfs": a.last_rx_rms_dbfs,
            "peak_dbfs": a.last_rx_peak_dbfs,
            "clip": bool(a.last_rx_clip),
            "gain": float(backend.cfg.audio.rx_gain),
            "auto": bool(a.auto_rx_enable),
            "auto_target_dbfs": float(a.auto_rx_target_dbfs),
        },
        "tx": {
            "rms_dbfs": a.last_tx_rms_dbfs,
            "peak_dbfs": a.last_tx_peak_dbfs,
            "clip": bool(a.last_tx_clip),
            "gain": float(backend.cfg.audio.tx_gain),
        }
    })


async def api_tx_timeout(request: web.Request) -> web.Response:
    """Update TX timeout settings without restarting the backend.

    Body: {minutes:int, override:bool}
    """
    data = await request.json()

    def clamp_int(v, lo, hi, default):
        try:
            v = int(v)
        except Exception:
            v = default
        return max(lo, min(hi, v))

    minutes = clamp_int(data.get("minutes", backend.cfg.tx_timeout_min), 0, 720, 120)
    override = bool(data.get("override", backend.cfg.tx_timeout_override))

    backend.cfg.tx_timeout_min = minutes
    backend.cfg.tx_timeout_override = override
    # Re-arm from next actual TX
    backend.rt.first_tx_epoch = None
    await backend.info(f"TX timeout set: {minutes} min, override={'ON' if override else 'OFF'}")
    return web.json_response({"ok": True, "minutes": minutes, "override": override})


async def api_start(request: web.Request) -> web.Response:
    data = await request.json()

    def clamp_int(v, lo, hi, default):
        try:
            v = int(v)
        except Exception:
            v = default
        return max(lo, min(hi, v))

    mode = str(data.get("mode", "LISTEN")).upper().strip()
    if mode not in ("LISTEN","CQ","CALL","REPLY"):
        mode = "LISTEN"

    cfg = Config(
        my_call=str(data.get("my_call","GW4WND")).upper().strip(),
        their_call=str(data.get("their_call","")).upper().strip(),
        mode=mode,
        period_len=clamp_int(data.get("period_len", 15), 5, 60, 15),
        tx_slot=str(data.get("tx_slot","A")).upper().strip() if str(data.get("tx_slot","A")).upper().strip() in ("A","B") else "A",
        auto_seq=bool(data.get("auto_seq", True)),
        band_sel=str(data.get("band_sel","144")).upper().strip(),
        band_custom=str(data.get("band_custom",""))[:16],
        freq_mhz=str(data.get("freq_mhz",""))[:16],
        tx_timeout_min=clamp_int(data.get("tx_timeout_min", 120), 0, 720, 120),
        tx_timeout_override=bool(data.get("tx_timeout_override", False)),
        audio=backend.cfg.audio,  # already set via /api/audio
    )
    await backend.start(cfg)
    # If UI started a REPLY QSO by clicking a decode, it can pass the clicked decode's correlation
    # so we latch the report *immediately* for the very first transmission.
    init_corr = data.get("initial_corr", None)
    if init_corr is not None and mode == "REPLY" and cfg.their_call:
        try:
            c = float(init_corr)
        except Exception:
            c = None
        if c is not None:
            qpct = int(round(max(0.0, min(1.0, c)) * 100))
            backend.rt.my_report = report_from_qpct(qpct)
            backend.rt.report_locked = True
            backend.rt.next_tx_text = backend._compute_next_tx_text()
            await backend.info(f"Initial report latched from click: Q={qpct}% -> {backend.rt.my_report}")
    return web.json_response({"ok": True})

async def api_stop(_: web.Request) -> web.Response:
    await backend.stop()
    await backend.info("Stopped")
    return web.json_response({"ok": True})

async def api_set_target(request: web.Request) -> web.Response:
    data = await request.json()
    their = str(data.get("their_call","")).upper().strip()
    if their:
        backend.cfg.their_call = their
        backend.rt.qso_state = QSO_CALLING_STN
        backend.rt.observed_remote_slot = data.get("remote_slot") or backend.rt.observed_remote_slot
        await backend.info(f"Target set: {their} (will TX in opposite slot if remote_slot known)")
    return web.json_response({"ok": True})

async def api_cq(_: web.Request) -> web.Response:
    backend.cfg.mode = "CQ"
    backend.rt.qso_state = QSO_CALLING_CQ
    backend.rt.observed_remote_slot = None
    await backend.info("CQ mode engaged")
    return web.json_response({"ok": True})

async def api_listen(_: web.Request) -> web.Response:
    backend.cfg.mode = "LISTEN"
    backend.rt.qso_state = QSO_LISTEN
    await backend.info("Listen mode engaged")
    return web.json_response({"ok": True})


async def api_qso_log(request: web.Request) -> web.Response:
    try:
        limit = int(request.query.get("limit","200"))
    except Exception:
        limit = 200
    limit = max(1, min(limit, 5000))
    try:
        qsos = read_adif_last(backend.adif_path, limit=limit)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e), "qsos": []})
    return web.json_response({"ok": True, "qsos": qsos, "adif_path": str(backend.adif_path)})


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/api/devices", api_devices)
    app.router.add_get("/api/diagnostics", api_diagnostics)
    app.router.add_get("/api/qso_log", api_qso_log)
    app.router.add_get("/api/levels", api_levels)
    app.router.add_post("/api/audio", api_audio_set)
    app.router.add_post("/api/gain", api_gain_set)
    app.router.add_post("/api/tx_timeout", api_tx_timeout)
    app.router.add_post("/api/start", api_start)
    app.router.add_post("/api/stop", api_stop)
    app.router.add_post("/api/set_target", api_set_target)
    app.router.add_post("/api/cq", api_cq)
    app.router.add_post("/api/listen", api_listen)
    return app

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8088)
    args = p.parse_args()
    web.run_app(make_app(), host=args.host, port=args.port)

if __name__ == "__main__":
    main()