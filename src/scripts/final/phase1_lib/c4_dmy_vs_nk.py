"""C4: NeuroKit vs DMY beats (LTST .16a, manually placed ISO/J on the 16 s average beat). Clean, raw, no filter.

For each DMY beat (record, lead):
  window   = the NL .16a beats before it + itself + the NR .16a beats after it (the beats LTST averaged)
  matching = each window beat -> nearest NeuroKit R-peak within +/- TOL samples of its 16a sample
             (no offset correction: on MV2 NK's R-peak flips between two QRS deflections ~60 ms apart)
  NK value = median (and mean) of per-beat NK ST deviation over matched, usable window beats, per C3 config
  DMY value = ee (ST at J+60 minus ISO on the average beat), uV -> mV
Positions, relative to each beat's fiducial (fid = 16a sample - 16 ms):
  DMY J = fid + JP_ms, DMY ISO point = fid - ISO_ms  (average beat)
  NK  J = R_offset - fid, NK PR segment = [P_offset, R_onset) - fid  (median over window beats)

Usage: python3 c4_dmy_vs_nk.py   -> c4_dmy/c4_dmy_per_beat.csv
       python3 c4_dmy_vs_nk.py --st_root <dir with config subdirs> --configs a b c --out_csv <path>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/Neurokit2_Pipeline")
from ltst_pipeline.parse_ltst_16a_iso_jp_ee import parse_aux  # noqa: E402

ROOT = Path("/home/student/Desktop/Research Data/DATASET NK")
DELIN = ROOT / "Delineation Data" / "delin_raw_nofilter_neurokit"
C3 = ROOT / "C_neurokit" / "c3_st_configs" / "st_beats"
STORED = ROOT / "ST features Data" / "st_raw_pr_only_nofilter"
OUT = ROOT / "C_neurokit" / "c4_dmy"
FS = 250
TOL = 25  # 100 ms; RR >= 300 ms so no ambiguity
FID_SHIFT = round(0.016 * FS)
CONFIGS = ["ref_stored", "pr_ge1", "pr_ge2", "pr_ge3", "pr_ge4", "pr_ge5"]


def load_16a(rec):
    ann = wfdb.rdann(str(ROOT / "ltst_records" / rec / rec), "16a")
    rows = []
    for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
        if sym != "s":
            continue
        p = parse_aux(aux)
        if p is not None:
            rows.append({"sample": int(s), **p})
    return pd.DataFrame(rows)


def st_file(cfg, rec, lead, st_root=C3):
    d = (STORED if cfg == "ref_stored" else st_root / cfg) / rec / f"lead_{lead}"
    return next(d.glob(f"waves_{rec}_lead{lead}_*.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--st_root", type=Path, default=C3)
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--out_csv", type=Path, default=OUT / "c4_dmy_per_beat.csv")
    a = ap.parse_args()
    configs = a.configs
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    records = sorted(p.stem.split("_")[1] for p in (DELIN / "lead_0").glob("waves_*.csv"))
    out = []
    for rec in records:
        a16 = load_16a(rec)
        for lead in (0, 1):
            L = a16[a16.lead_idx == lead].sort_values("sample").reset_index(drop=True)
            if not L.DMY.any():
                continue
            wv = pd.read_csv(DELIN / f"lead_{lead}" / f"waves_{rec}_lead{lead}.csv",
                             usecols=["r_peak", "ECG_P_Offsets", "ECG_R_Onsets", "ECG_R_Offsets"])
            R = wv.r_peak.to_numpy()
            st = {c: pd.read_csv(st_file(c, rec, lead, a.st_root), usecols=["r_peak", "st_deviation_mV"]) for c in configs}
            for c in configs:
                assert np.array_equal(st[c].r_peak.to_numpy(), R), (c, rec, lead)

            s16 = L["sample"].to_numpy()
            i = np.clip(np.searchsorted(R, s16), 1, len(R) - 1)
            nearest = np.where(np.abs(R[i] - s16) < np.abs(R[i - 1] - s16), i, i - 1)
            off_all = s16 - R[nearest]
            matched = np.abs(off_all) <= TOL
            offset = int(np.median(off_all[matched]))
            nk_idx = np.where(matched, nearest, -1)

            fid = s16 - FID_SHIFT
            for k in np.where(L.DMY.to_numpy())[0]:
                r = L.iloc[k]
                lo, hi = max(0, k - int(r.NL)), min(len(L) - 1, k + int(r.NR))
                win = np.arange(lo, hi + 1)
                idx = nk_idx[win]
                ok = idx >= 0
                row = {"record": rec, "lead": "ML2" if lead == 0 else "MV2", "sample_16a": int(r["sample"]),
                       "NL": int(r.NL), "NR": int(r.NR), "n_window": int(r.NL + r.NR + 1),
                       "n_window_in_16a": len(win), "n_nk_matched": int(ok.sum()),
                       "center_matched": bool(nk_idx[k] >= 0), "offset_16a_minus_rpeak": offset,
                       "window_offset_iqr": float(np.subtract(*np.percentile(off_all[win][ok], [75, 25]))) if ok.any() else np.nan,
                       "dmy_ee_mV": r.ee_uV / 1000.0, "dmy_aa_mV": r.aa_uV / 1000.0,
                       "dmy_J_ms": r.JP_ms, "dmy_ISO_ms": r.ISO_ms}
                for c in configs:
                    v = st[c].st_deviation_mV.to_numpy()[idx[ok]]
                    v = v[~np.isnan(v)]
                    row[f"{c}_n_usable"] = len(v)
                    row[f"{c}_median_mV"] = float(np.median(v)) if len(v) else np.nan
                    row[f"{c}_mean_mV"] = float(np.mean(v)) if len(v) else np.nan
                # positions (ms, relative to each beat's own fiducial)
                f = fid[win][ok]
                w = wv.iloc[idx[ok]]
                j = (w.ECG_R_Offsets.to_numpy() - f) * 4.0
                pr_lo = (w.ECG_P_Offsets.to_numpy() - f) * 4.0
                pr_hi = (w.ECG_R_Onsets.to_numpy() - f) * 4.0
                valid_pr = ~np.isnan(pr_lo) & ~np.isnan(pr_hi) & (pr_hi > pr_lo)
                row["nk_J_ms_median"] = float(np.nanmedian(j)) if np.isfinite(j).any() else np.nan
                row["nk_PRmid_ms_median"] = float(np.median((pr_lo + pr_hi)[valid_pr] / 2)) if valid_pr.any() else np.nan
                iso = -float(r.ISO_ms)
                row["dmy_ISO_inside_nk_PR_frac"] = float(((pr_lo <= iso) & (iso < pr_hi))[valid_pr].mean()) if valid_pr.any() else np.nan
                row["n_valid_PR"] = int(valid_pr.sum())
                out.append(row)
        print(rec, "done", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(a.out_csv, index=False)
    print("wrote", a.out_csv, len(df), "DMY beats")


if __name__ == "__main__":
    main()
