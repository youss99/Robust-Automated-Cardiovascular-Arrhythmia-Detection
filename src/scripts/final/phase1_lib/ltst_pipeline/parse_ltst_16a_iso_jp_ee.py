#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb


DEFAULT_RECORDS = [
    "s20011","s20051","s20061","s20071",
    "s20081","s20091","s20101","s20111",
    "s20121","s20131","s20141","s20201",
    "s20211","s20221","s20231","s20241",
]


def record_path(base_path: str, rec: str) -> str:
    return f"{base_path}/{rec}/{rec}"


def get_lead_names(sig) -> List[str]:
    try:
        names = list(sig.sig_name)
        if names and all(isinstance(n, str) for n in names):
            return names
    except Exception:
        pass
    return [f"lead{i}" for i in range(getattr(sig, "n_sig", 2))]


def to_mV(values: np.ndarray, unit: str) -> Tuple[np.ndarray, str]:
    """
    Convert a vector to mV based on WFDB unit string. Returns (converted_values, normalized_unit_str).
    If unit is unknown, returns original values and unit.
    """
    if unit is None:
        return values, "unknown"
    u = str(unit).strip().lower()

    # common variants
    if u in ("mv", "millivolt", "millivolts"):
        return values, "mV"
    if u in ("uv", "µv", "microvolt", "microvolts"):
        return values / 1000.0, "mV"
    if u in ("v", "volt", "volts"):
        return values * 1000.0, "mV"

    # unknown: keep as-is
    return values, unit


def mean_window(x: np.ndarray, idx: int, half: int = 2) -> float:
    """
    Mean of x[idx-half : idx+half+1] with bounds checking.
    half=2 -> 5 samples (~20ms @ 250Hz)
    """
    n = x.size
    if idx < 0 or idx >= n:
        return np.nan
    lo = max(0, idx - half)
    hi = min(n, idx + half + 1)
    seg = x[lo:hi]
    return float(np.mean(seg)) if seg.size else np.nan


_ST_RE = re.compile(r"^(?P<dmy>DMY)?\s*(?P<tag>ST(?P<lead>[01]))(?P<body>.*)$", re.IGNORECASE)


def parse_aux(aux: str) -> Optional[Dict[str, Any]]:
    """
    Parse aux note like:
      ST0-11,-4,-53,-57,-40,-11,+41,+112,72,40,120,0,9
    or with DMY prefix:
      DMYST1+...,72,40,120,0,9

    Returns dict with:
      lead (0/1), dmy (bool),
      aa..hh (ints uV), ISO_ms, JP_ms, J80or60_ms, NL, NR
    """
    if aux is None:
        return None
    s = aux.strip()
    if not s:
        return None

    m = _ST_RE.match(s)
    if not m:
        return None

    dmy = bool(m.group("dmy"))
    lead = int(m.group("lead"))
    body = m.group("body").strip()

    # body begins with the first number, e.g. "-11,-4,..."
    if body.startswith(","):
        body = body[1:].strip()

    parts = [p.strip() for p in body.split(",") if p.strip() != ""]
    # expected: 8 ST values + ISO + JP + J80(60) + NL + NR = 13 integers
    if len(parts) < 13:
        return None

    try:
        vals = [int(p) for p in parts[:13]]
    except Exception:
        return None

    aa, bb, cc, dd, ee, ff, gg, hh, ISO_ms, JP_ms, J8060_ms, NL, NR = vals

    return {
        "lead_idx": lead,
        "DMY": dmy,
        "aa_uV": aa, "bb_uV": bb, "cc_uV": cc, "dd_uV": dd,
        "ee_uV": ee, "ff_uV": ff, "gg_uV": gg, "hh_uV": hh,
        "ISO_ms": ISO_ms,
        "JP_ms": JP_ms,
        "J80or60_ms": J8060_ms,
        "NL": NL,
        "NR": NR,
    }


def compute_locations(sample_16a: int, fs: float, ISO_ms: int, JP_ms: int) -> Dict[str, int]:
    """
    LTSTDB: .16a annotation time = Aristotle fiducial + 16 ms.
    So fiducial_sample ≈ sample_16a - round(0.016*fs).

    ISO_ms: (typically) milliseconds BEFORE the fiducial (stored as positive)
    JP_ms:  milliseconds AFTER  the fiducial (stored as positive)

    We implement:
      fid = sample - round(16ms)
      iso = fid - round(ISO_ms)
      jp  = fid + round(JP_ms)
    with sign-safe fallbacks if ISO_ms/JP_ms show up negative.
    """
    fid = int(sample_16a - round(0.016 * fs))

    # ISO: usually a positive number meaning "before"
    if ISO_ms >= 0:
        iso = int(fid - round((ISO_ms / 1000.0) * fs))
    else:
        iso = int(fid + round((abs(ISO_ms) / 1000.0) * fs))

    # JP: usually positive meaning "after"
    if JP_ms >= 0:
        jp = int(fid + round((JP_ms / 1000.0) * fs))
    else:
        jp = int(fid - round((abs(JP_ms) / 1000.0) * fs))

    return {"fid_sample": fid, "ISO_sample": iso, "JP_sample": jp}


def process_record(base_path: str, rec: str) -> pd.DataFrame:
    rp = record_path(base_path, rec)

    sig = wfdb.rdrecord(rp)
    ann = wfdb.rdann(rp, "16a")

    fs = float(sig.fs)
    X = np.asarray(sig.p_signal, dtype=float)  # physical units per sig.units
    n_samples = X.shape[0]
    n_leads = X.shape[1] if X.ndim == 2 else 1
    lead_names = get_lead_names(sig)

    # per-lead unit handling
    units = getattr(sig, "units", None)
    if units is None or len(units) < n_leads:
        units = ["unknown"] * n_leads

    # annotation fields
    samp = np.asarray(ann.sample, dtype=int)
    sym = np.asarray(getattr(ann, "symbol", [""] * len(samp)), dtype=object)

    if hasattr(ann, "chan") and ann.chan is not None:
        chan = np.asarray(ann.chan, dtype=int)
    else:
        chan = np.zeros_like(samp)

    aux_raw = getattr(ann, "aux_note", [""] * len(samp))
    aux = [a.decode() if isinstance(a, (bytes, bytearray)) else str(a) for a in aux_raw]

    rows: List[Dict[str, Any]] = []

    for i in range(len(samp)):
        if str(sym[i]) != "s":  # ST measurement entries
            continue

        parsed = parse_aux(aux[i])
        if parsed is None:
            continue

        lead_idx = int(chan[i]) if 0 <= int(chan[i]) < n_leads else parsed["lead_idx"]
        if not (0 <= lead_idx < n_leads):
            continue

        # compute absolute locations
        loc = compute_locations(int(samp[i]), fs, int(parsed["ISO_ms"]), int(parsed["JP_ms"]))
        fid = loc["fid_sample"]
        iso_s = loc["ISO_sample"]
        jp_s = loc["JP_sample"]
        j60_s = jp_s + int(round(0.060 * fs))

        # bounds check
        if not (0 <= fid < n_samples):
            continue

        # get signal trace in mV for this lead
        x = X[:, lead_idx] if X.ndim == 2 else X
        x_mV, unit_norm = to_mV(x, units[lead_idx])

        # ISO & JP voltages from signal (5-sample mean)
        iso_v_mV = mean_window(x_mV, iso_s, half=2) if 0 <= iso_s < n_samples else np.nan
        jp_v_mV = mean_window(x_mV, jp_s, half=2) if 0 <= jp_s < n_samples else np.nan

        # ST at J+60 from signal (5-sample mean) — optional but useful
        st_j60_v_mV = mean_window(x_mV, j60_s, half=2) if 0 <= j60_s < n_samples else np.nan

        # ee from .16a is in uV -> convert to mV
        ee_mV = float(parsed["ee_uV"]) / 1000.0
        aa_mV = float(parsed["aa_uV"]) / 1000.0

        rows.append({
            "record": rec,
            "fs": fs,
            "lead_idx": lead_idx,
            "lead_name": lead_names[lead_idx] if lead_idx < len(lead_names) else f"lead{lead_idx}",
            "signal_unit_after_conv": unit_norm,

            "ann_sample": int(samp[i]),
            "ann_time_s": float(samp[i]) / fs,
            "DMY": bool(parsed["DMY"]),
            "NL": int(parsed["NL"]),
            "NR": int(parsed["NR"]),

            # fiducial & locations
            "fid_sample": fid,
            "fid_time_s": fid / fs,

            "ISO_ms": int(parsed["ISO_ms"]),
            "ISO_sample": iso_s,
            "ISO_time_s": (iso_s / fs) if 0 <= iso_s < n_samples else np.nan,
            "ISO_voltage_mV": iso_v_mV,

            "JP_ms": int(parsed["JP_ms"]),
            "JP_sample": jp_s,
            "JP_time_s": (jp_s / fs) if 0 <= jp_s < n_samples else np.nan,
            "JP_voltage_mV": jp_v_mV,

            # ee (from file) and helpful comparisons
            "ee_uV": int(parsed["ee_uV"]),
            "ee_mV": ee_mV,

            "J_plus_60_sample": j60_s,
            "J_plus_60_time_s": (j60_s / fs) if 0 <= j60_s < n_samples else np.nan,
            "J_plus_60_voltage_mV_from_signal": st_j60_v_mV,

            # aa is HR-dependent point; sometimes equals ee (if aa uses 60ms)
            "aa_uV": int(parsed["aa_uV"]),
            "aa_mV": aa_mV,

            # full set (optional, but handy to keep)
            "bb_mV": float(parsed["bb_uV"]) / 1000.0,
            "cc_mV": float(parsed["cc_uV"]) / 1000.0,
            "dd_mV": float(parsed["dd_uV"]) / 1000.0,
            "ff_mV": float(parsed["ff_uV"]) / 1000.0,
            "gg_mV": float(parsed["gg_uV"]) / 1000.0,
            "hh_mV": float(parsed["hh_uV"]) / 1000.0,
            "J80or60_ms": int(parsed["J80or60_ms"]),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Parse LTSTDB .16a into ISO/JP locations+voltages and ee (mV).")
    ap.add_argument("--base_path", default="ltst_records", help="Root folder containing sXXXXX/sXXXXX.*")
    ap.add_argument("--records", nargs="*", default=None, help="Record IDs (default: common LTST list)")
    ap.add_argument("--out_dir", default="exports_16a_parsed", help="Output directory")
    args = ap.parse_args()

    records = args.records or DEFAULT_RECORDS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_df = []
    for rec in records:
        print(f"[load] {rec}")
        try:
            df = process_record(args.base_path, rec)
        except Exception as e:
            print(f"[warn] failed {rec}: {e}")
            continue

        if df.empty:
            print(f"[warn] {rec}: no parsable .16a ST rows found")
            continue

        # write per-record, per-lead
        for lead_idx in sorted(df["lead_idx"].unique()):
            sub = df[df["lead_idx"] == lead_idx].copy()
            rec_dir = out_dir / rec
            rec_dir.mkdir(parents=True, exist_ok=True)
            out_path = rec_dir / f"{rec}_lead{lead_idx}_16a_iso_jp_ee.csv"
            sub.to_csv(out_path, index=False)
            print(f"[write] {out_path} ({len(sub)} rows)")

        all_df.append(df)

    if all_df:
        big = pd.concat(all_df, axis=0, ignore_index=True)
        big_path = out_dir / "ALL_records_ALL_leads_16a_iso_jp_ee.csv"
        big.to_csv(big_path, index=False)
        print(f"\n[write] {big_path} ({len(big)} rows)")
    else:
        print("[done] nothing exported.")


if __name__ == "__main__":
    main()
