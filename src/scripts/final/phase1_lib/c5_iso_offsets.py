"""C3/C4 follow-up: fixed-offset isoelectric level. Clean, raw, no filter (delin_raw_nofilter_neurokit).

Per beat:
  ST   st3med : median of x[J+60-1 : J+60+2]  (stored measurement)
       st5med : median of x[J+60-2 : J+60+3]
       st5mean: mean   of x[J+60-2 : J+60+3]  (LTST convention: 5 samples = 20 ms)
  ISO  (5-sample mean around a centre sample c, LTST convention)
       qon{k}  : c = R_onset - k ms                          (PQ junction; needs R_onset, R_onset < R-peak)
       rpk{k}  : c = R-peak  - k ms                          (no delineation needed)
       flat    : flattest 5-sample window (min max-min), c in [R_onset - 60 ms, R_onset - 8 ms]
       prmed   : median of PR segment [P_off, R_on), gate P_off < R_on < R-peak < J+60 (stored; reference)
  ST deviation = ST - ISO; NaN where either is unavailable.

Writes st_beats/<st>_<iso>/<rec>/lead_<k>/waves_<rec>_lead<k>_<st>_<iso>.csv (columns as the pipeline expects).
Usage: python3 c5_iso_offsets.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

ROOT = Path("/home/student/Desktop/Research Data/DATASET NK")
DELIN = ROOT / "Delineation Data" / "delin_raw_nofilter_neurokit"
STORED = ROOT / "ST features Data" / "st_raw_pr_only_nofilter"
OUT = ROOT / "C_neurokit" / "c5_iso" / "st_beats"
FS = 250
QON_MS = [0, 8, 12, 20]
RPK_MS = [48, 64, 80]
FLAT_MS = (60, 8)
COLS = ["r_peak", "ECG_P_Offsets", "ECG_R_Onsets", "J_point", "J_plus_60ms"]


def ms(v):
    return int(round(v / 1000.0 * FS))


def window_stat(x, centre, half, fn):
    """fn over x[c-half : c+half+1] for each centre; NaN where centre is NaN or out of bounds."""
    out = np.full(len(centre), np.nan)
    ok = ~np.isnan(centre)
    c = np.where(ok, centre, 0).astype(np.int64)
    ok &= (c - half >= 0) & (c + half < len(x))
    idx = c[ok, None] + np.arange(-half, half + 1)
    out[ok] = fn(x[idx], axis=1)
    return out


def flat_iso(x, r_on):
    lo, hi = ms(FLAT_MS[0]), ms(FLAT_MS[1])
    offs = np.arange(-lo, -hi + 1)
    out = np.full(len(r_on), np.nan)
    ok = ~np.isnan(r_on)
    c = np.where(ok, r_on, 0).astype(np.int64)
    ok &= (c - lo - 2 >= 0) & (c - hi + 2 < len(x))
    cand = c[ok, None] + offs                                   # beats x candidates
    win = x[cand[..., None] + np.arange(-2, 3)]                  # beats x candidates x 5
    best = np.argmin(win.max(axis=2) - win.min(axis=2), axis=1)
    out[ok] = win[np.arange(len(best)), best].mean(axis=1)
    return out


def main():
    records = sorted(p.stem.split("_")[1] for p in (DELIN / "lead_0").glob("waves_*.csv"))
    for rec in records:
        sig = wfdb.rdrecord(str(ROOT / "ltst_records" / rec / rec)).p_signal
        for lead in (0, 1):
            x = sig[:, lead]
            n = len(x)
            df = pd.read_csv(DELIN / f"lead_{lead}" / f"waves_{rec}_lead{lead}.csv", usecols=COLS)
            rp, p_off, r_on, j60 = (df[c].to_numpy(dtype=float) for c in ["r_peak", "ECG_P_Offsets", "ECG_R_Onsets", "J_plus_60ms"])
            j_ok = ~np.isnan(j60) & (rp < j60)
            jc = np.where(j_ok, j60, np.nan)
            st = {"st3med": window_stat(x, jc, 1, np.median), "st5med": window_stat(x, jc, 2, np.median),
                  "st5mean": window_stat(x, jc, 2, np.mean)}

            qrs_ok = ~np.isnan(r_on) & (r_on < rp)
            ron = np.where(qrs_ok, r_on, np.nan)
            iso = {f"qon{k}": window_stat(x, ron - ms(k), 2, np.mean) for k in QON_MS}
            iso.update({f"rpk{k}": window_stat(x, rp - ms(k), 2, np.mean) for k in RPK_MS})
            iso["flat"] = flat_iso(x, ron)
            pr_ok = (~np.isnan(p_off) & ~np.isnan(r_on) & j_ok & (p_off >= 0) & (r_on < n)
                     & (p_off < r_on) & (r_on < rp))
            prmed = np.full(len(df), np.nan)
            for i in np.where(pr_ok)[0]:
                prmed[i] = np.median(x[int(p_off[i]):int(r_on[i])])
            iso["prmed"] = prmed

            for sn, sv in st.items():
                for iname, iv in iso.items():
                    cfg = f"{sn}_{iname}"
                    out = df[["r_peak", "J_point", "J_plus_60ms"]].copy()
                    out["st_value_mV"] = sv
                    out["iso_baseline_mV"] = iv
                    out["st_deviation_mV"] = np.round(sv - iv, 6)
                    d = OUT / cfg / rec / f"lead_{lead}"
                    d.mkdir(parents=True, exist_ok=True)
                    out.to_csv(d / f"waves_{rec}_lead{lead}_{cfg}.csv", index=False)

            ref = pd.read_csv(next((STORED / rec / f"lead_{lead}").glob("*.csv")), usecols=["st_deviation_mV"]).st_deviation_mV.to_numpy()
            mine = st["st3med"] - iso["prmed"]
            same = np.allclose(np.nan_to_num(ref, nan=9e9), np.nan_to_num(mine, nan=9e9), atol=1e-9)
            print(rec, lead, "st3med_prmed == stored:", same, flush=True)
    (OUT / "BUILT").touch()


if __name__ == "__main__":
    main()
