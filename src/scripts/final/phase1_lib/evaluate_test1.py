#!/usr/bin/env python3
"""
Test 1 evaluation: beat-centered regression models vs the published cv5_optB models.

Two model sets  x  two inference schemes, on the clean signal:
  model set  published : ltstdb_median_cv5_optB_*        (grid windows, median label)
             test1     : ltstdb_beatcenter_test1_cv5_optB_* (beat-centered windows, center-beat label)
  scheme     grid      : non-overlapping 10 s windows from sample 0, FSM anchor = window end
                         (exactly new_fsm_ml.generate_ml_alarms -> predict_st_per_segment)
             beat      : 10 s windows centered on .atr beats, next center >= 10 s after the
                         previous one (the Test 1 training rule, with detected beats in place of
                         .16a annotations), FSM anchor = center beat

Everything downstream is the published machinery, untouched:
  FSM     ltst_pipeline.old_fsm_gs.run_fsm_on_record, use_sqi=False   (as SPEC 2 C4)
  mask    MANUAL_OMIT                                                   (as new_fsm_ml.py)
  scoring results/alarms/_code/score_alarms.py per_record_components + metrics_from_components
  union   lead_II model's alarms on II + lead_V2 model's alarms on V2   (single_lead_union)

CV honesty: a record is predicted only by the fold model that held it out.

Conditions (as the paper): clean; noisy = NSTDB 'em', seed 42, unit gain, model fed the contaminated
signal; noisy_sqi = noisy + SQI gate computed on a 0.67 Hz zero-phase high-passed copy of the
contaminated signal (SQI_GATE_FIX.md, path B), 30 s look-back every 5 s, gate freezes the FSM.

Usage:
  $VENV/bin/python evaluate_test1.py infer --condition clean|noisy        # GPU; writes predictions[_noisy]/
  $VENV/bin/python evaluate_test1.py score --condition clean|noisy|noisy_sqi [--threshold 6]
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

NK = Path("/home/student/GIT/Neurokit2_Pipeline")
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, str(NK))
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/results/neurokit/_code")
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/results/alarms/_code")

REPO = Path("/home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
RECORDS_DIR = Path("/home/student/Desktop/Research Data/DATASET NK/ltst_records")
OUT = Path("/home/student/GIT/results/beatcenter_test1")
PRED = OUT / "predictions"
GS_PATH = "/home/student/GIT/transfer_bundle/gs_per_beat.csv"
DUR_CSV = "/home/student/GIT/results/neurokit/_sweep/record_durations.csv"

FS, WIN, HALF = 250, 2500, 1250
FOLDS = {1: ["s20051", "s20211", "s20221"], 2: ["s20081", "s20101", "s20141"],
         3: ["s20061", "s20091", "s20241"], 4: ["s20111", "s20131", "s20231"],
         5: ["s20011", "s20071", "s20201"]}
REC2FOLD = {r: f for f, rs in FOLDS.items() for r in rs}

ARMS = {  # arm label -> lead mode -> checkpoint arm tag
    "supervised": {"two_lead": "supervised", "lead_II": "supervised", "lead_V2": "supervised"},
    "Chapman-pretrained": {"two_lead": "pretrained_chapman_ii_v2", "lead_II": "pretrained_chapman_ii_only",
                           "lead_V2": "pretrained_chapman_v2_only"},
    "PTBXL-pretrained": {"two_lead": "pretrained_ptbxl_ii_v2", "lead_II": "pretrained_ptbxl_ii",
                         "lead_V2": "pretrained_ptbxl_v2"},
}
MODES = ["two_lead", "lead_II", "lead_V2"]
SETS = ["published", "test1"]
SCHEMES = ["grid", "beat"]
SCOPES = {"two_lead": ["FULL"], "lead_II": ["FULL", "own_II"], "lead_V2": ["FULL", "own_V2"],
          "union": ["FULL"]}


def ckpt(model_set: str, mode: str, tag: str, fold: int) -> Path:
    if model_set == "test1":
        name = f"ltstdb_beatcenter_test1_cv5_optB_{mode}_regression_{tag}_fold{fold:02d}_thesis_small"
        root = REPO / "results/fine-tune/saved_models/patient-segment-regression"
    else:
        name = f"ltstdb_median_cv5_optB_{mode}_regression_{tag}_fold{fold:02d}_thesis_small"
        root = (REPO / "ptbxl_models_v2/fine-tune") if "ptbxl" in tag else \
               (REPO / "results/fine-tune/saved_models/patient-segment-regression_2")
    return root / name / f"{name}_best.pt"


def pred_dir(condition: str) -> Path:
    return PRED if condition == "clean" else OUT / "predictions_noisy"


def pred_path(model_set, arm, mode, scheme, condition="clean") -> Path:
    return pred_dir(condition) / f"{model_set}__{arm}__{mode}__{scheme}.csv.gz"


def load_signal(rec: str, condition: str) -> np.ndarray:
    """(N, 2) float64 p_signal; contaminated exactly as new_fsm_ml.generate_ml_alarms for noisy."""
    import wfdb
    p = wfdb.rdrecord(str(RECORDS_DIR / rec / rec)).p_signal
    if condition != "clean":
        from ltst_pipeline.noise import contaminate_ecg_with_motion_artifact
        p, _ = contaminate_ecg_with_motion_artifact(p, seed=42, fs=FS)
    return p


# ----------------------------------------------------------------------------- inference
def beat_centers(atr_samples: np.ndarray, n: int) -> np.ndarray:
    c = np.unique(atr_samples)
    c = c[(c - HALF >= 0) & (c + HALF <= n)]
    keep, last = [], None
    for x in c:
        if last is None or x >= last + WIN:
            keep.append(x); last = x
    return np.asarray(keep, dtype=np.int64)


def infer(condition: str) -> None:
    import torch
    import wfdb
    from ltst_pipeline.ml_inference import (load_patchecg_model, predict_st_per_segment, _resample_to_model_fs,
                                            create_patch, PATCH_LEN, DataAugmentation)
    from ltst_pipeline.old_fsm_gs import get_rpeaks_from_atr

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pred_dir(condition).mkdir(parents=True, exist_ok=True)
    rows = {(s, a, m, sc): [] for s in SETS for a in ARMS for m in MODES for sc in SCHEMES}
    t0 = time.time()

    @torch.no_grad()
    def forward(model, revin, x_model: np.ndarray, mode: str) -> np.ndarray:
        if mode == "lead_II":
            x_model = x_model[:, 0:1]
        elif mode == "lead_V2":
            x_model = x_model[:, 1:2]
        out = []
        for i in range(0, len(x_model), 1024):
            xb = torch.from_numpy(np.ascontiguousarray(x_model[i:i + 1024])).to(dev)
            xb = revin(xb, "norm")
            pr = model(create_patch(xb, patch_len=PATCH_LEN, stride=PATCH_LEN), mask=None,
                       inverted_mask=None, indices=None, augmentation=DataAugmentation.none)
            pr = pr.detach().cpu().numpy()
            out.append(pr[:, None] if pr.ndim == 1 else pr)
        return np.concatenate(out).astype(np.float32)

    def windows(sig: np.ndarray, starts: np.ndarray):
        off = np.arange(WIN)
        xs, fin = [], []
        for i in range(0, len(starts), 4096):
            seg = sig[starts[i:i + 4096, None] + off]
            f = np.isfinite(seg).all(axis=(1, 2))
            if not f.all():
                seg = np.where(np.isfinite(seg), seg, 0.0).astype(np.float32)
            xs.append(_resample_to_model_fs(seg, FS)); fin.append(f)
        return np.concatenate(xs), np.concatenate(fin)

    gate_done = set()
    for fold, recs in FOLDS.items():
        cache = {}
        for rec in recs:
            sig = load_signal(rec, condition)[:, [0, 1]].astype(np.float32)
            n_grid = len(sig) // WIN
            g_starts = np.arange(n_grid, dtype=np.int64) * WIN
            b_centers = beat_centers(get_rpeaks_from_atr(wfdb.rdann(str(RECORDS_DIR / rec / rec), "atr")), len(sig))
            gx, gf = windows(sig, g_starts)
            bx, bf = windows(sig, b_centers - HALF)
            cache[rec] = dict(sig=sig, grid=(g_starts + WIN - 1, gx, gf), beat=(b_centers, bx, bf))
            print(f"[{time.time()-t0:7.1f}s] fold{fold:02d} {rec}: grid {n_grid}  beat-centered {len(b_centers)}", flush=True)

        for model_set in SETS:
            for arm, tags in ARMS.items():
                for mode in MODES:
                    model, revin, _ = load_patchecg_model(str(ckpt(model_set, mode, tags[mode], fold)),
                                                          device=dev, lead_mode=mode)
                    for rec in recs:
                        c = cache[rec]
                        # gate: cached grid path == the pipeline's own predict_st_per_segment
                        if (model_set, arm, mode) not in gate_done:
                            _, ref = predict_st_per_segment(c["sig"], FS, model, revin, torch.device(dev),
                                                            lead_order=("II", "V2"), batch_size=1024, lead_mode=mode)
                            mine = forward(model, revin, c["grid"][1], mode)
                            refa = np.stack([ref[k] for k in ("II", "V2") if k in ref], axis=1)
                            d = float(np.nanmax(np.abs(mine - refa)))
                            assert d < 1e-4, f"grid path diverges from predict_st_per_segment: {d}"
                            gate_done.add((model_set, arm, mode))
                        for scheme in SCHEMES:
                            anchor, x, fin = c[scheme]
                            p = forward(model, revin, x, mode)
                            p[~fin] = np.nan
                            n = len(anchor)
                            rows[(model_set, arm, mode, scheme)].append(pd.DataFrame({
                                "record": rec, "fold": fold, "anchor_samp": anchor,
                                "pred_II_mV": p[:, 0] if mode in ("two_lead", "lead_II") else np.full(n, np.nan, np.float32),
                                "pred_V2_mV": p[:, -1] if mode in ("two_lead", "lead_V2") else np.full(n, np.nan, np.float32),
                            }))
                    del model
                    torch.cuda.empty_cache()
            print(f"[{time.time()-t0:7.1f}s] fold{fold:02d} {model_set}: 9 variants predicted", flush=True)

    for key, parts in rows.items():
        pd.concat(parts, ignore_index=True).to_csv(pred_path(*key, condition=condition), index=False)
    print(f"[{time.time()-t0:7.1f}s] wrote {len(rows)} prediction files; grid-path gate passed for {len(gate_done)} variants")


# ----------------------------------------------------------------------------- scoring
def alarms_from_predictions(df: pd.DataFrame, sqi_ctx: dict | None = None, threshold: float = 6.0) -> pd.DataFrame:
    """As results/regression/_code/c4_gate.py, with a generic anchor column.

    sqi_ctx: None for clean / noisy (use_sqi=False), else {record: (sqi_signal (N,2), r_peaks)}.
    """
    from ltst_pipeline.old_fsm_gs import MANUAL_OMIT, STTAG_TO_PAPER_LEAD, STTAG_TO_LEAD_IDX, run_fsm_on_record
    col = {"ST0": "pred_II_mV", "ST1": "pred_V2_mV"}
    out = []
    for rec_id, g in df.groupby("record"):
        g = g.sort_values("anchor_samp")
        samp = g["anchor_samp"].astype(int).tolist()
        for tag in ("ST0", "ST1"):
            paper = STTAG_TO_PAPER_LEAD[tag]
            if (rec_id, paper) in MANUAL_OMIT:
                continue
            st = g[col[tag]].astype(float).tolist()
            if not np.isfinite(st).any():
                continue
            if sqi_ctx is None:
                sig_q, rp, use = np.zeros(max(samp) + 1000), np.asarray(samp, dtype=int), False
            else:
                sig_q, rp = sqi_ctx[rec_id][0][:, STTAG_TO_LEAD_IDX[tag]], sqi_ctx[rec_id][1]
                use = True
            with contextlib.redirect_stdout(io.StringIO()):
                al = run_fsm_on_record(rec_id=rec_id, lead_tag=tag, samp_arr=samp, st_arr=st,
                                       ecg_sig_for_sqi=sig_q, r_peaks=rp, fs=FS, use_sqi=use,
                                       sqi_threshold_db=threshold, sqi_every_s=5.0, sqi_window_s=30.0)
            out += [{"record": a.record, "lead_tag": a.lead_tag, "lead_paper": a.lead_paper, "lead": a.lead_paper,
                     "onset_actual_s": a.onset_actual / FS, "end_actual_s": a.end_actual / FS,
                     "criterion": a.crit} for a in al]
    return pd.DataFrame(out, columns=["record", "lead_tag", "lead_paper", "lead", "onset_actual_s",
                                      "end_actual_s", "criterion"])


def sqi_context(condition: str) -> dict | None:
    """Per record: (high-passed contaminated signal (N,2), .atr R-peaks). Memoizes the SQI itself:
    it depends only on (signal, time), not on the model, so each value is computed once."""
    if condition != "noisy_sqi":
        return None
    import wfdb
    import ltst_pipeline.old_fsm_gs as G
    from ltst_pipeline.filters import highpass_filter
    ctx = {}
    for rec in REC2FOLD:
        sig = highpass_filter(load_signal(rec, condition)[:, [0, 1]], fs=FS)
        ctx[rec] = (np.ascontiguousarray(sig), G.get_rpeaks_from_atr(wfdb.rdann(str(RECORDS_DIR / rec / rec), "atr")))
    # keep the per-lead column views alive; the cache keys on (record, lead), resolved from the
    # exact array object the FSM passes through (old_fsm_gs passes ecg_sig_for_sqi unchanged)
    views = {(r, l): ctx[r][0][:, l] for r in REC2FOLD for l in (0, 1)}
    name_of = {id(v): k for k, v in views.items()}
    _orig, _cache = G.compute_sqi_last30s, {}

    def cached(**kw):
        name = name_of.get(id(kw["sig_filt"]))
        if name is None:                       # unknown array: never cache, just compute
            return _orig(**kw)
        key = (name, int(kw["current_r_peak"]), kw["window_sec"], kw["fs"], kw["pre_sec"], kw["post_sec"], kw["min_beats"])
        if key not in _cache:
            _cache[key] = _orig(**kw)
        return _cache[key]
    G.compute_sqi_last30s = cached
    ctx["_views"] = views
    return ctx


def score(condition: str, threshold: float) -> None:
    import nk_core as K
    from score_alarms import per_record_components, metrics_from_components, gs_scope

    tag = condition if condition != "noisy_sqi" else f"noisy_sqi_{threshold:g}dB"
    gs = K.load_gs(GS_PATH)
    durations = {r.record: float(r.duration_s) for r in pd.read_csv(DUR_CSV).itertuples()}
    mask = set(K.MANUAL_OMIT)
    ctx = sqi_context(condition)
    if ctx is not None:  # FSM slices columns itself; hand it the stable per-lead views via a 2-col shim
        views = ctx.pop("_views")
        ctx = {r: (_Cols(views[(r, 0)], views[(r, 1)]), ctx[r][1]) for r in REC2FOLD}

    schemes = SCHEMES + (["beat_end"] if condition == "noisy_sqi" else [])
    alarms = {}
    t0 = time.time()
    for s in SETS:
        for a in ARMS:
            for sc in schemes:
                for m in MODES:
                    df = pd.read_csv(pred_path(s, a, m, "beat" if sc == "beat_end" else sc, condition))
                    if sc == "beat_end":
                        df["anchor_samp"] = df["anchor_samp"] + HALF - 1   # last sample of the window
                    alarms[(s, a, m, sc)] = K.apply_mask(alarms_from_predictions(df, ctx, threshold), mask)
                alarms[(s, a, "union", sc)] = pd.concat(
                    [alarms[(s, a, "lead_II", sc)][lambda d: d.lead == "II"],
                     alarms[(s, a, "lead_V2", sc)][lambda d: d.lead == "V2"]], ignore_index=True)
            print(f"[{time.time()-t0:6.0f}s] {tag}: FSM done for {s} / {a}", flush=True)
    adir = OUT / "alarms" / tag
    adir.mkdir(parents=True, exist_ok=True)
    for (s, a, m, sc), df in alarms.items():
        df.to_csv(adir / f"{s}__{a}__{m}__{sc}.csv", index=False)

    comps, rows = {}, []
    for (s, a, m, sc), df in alarms.items():
        for scope in SCOPES[m]:
            comp = per_record_components(gs_scope(gs, scope), K.scope_filter(df, scope), durations, scope, mask, K.RECORDS_15)
            comps[(s, a, m, sc, scope)] = comp
            rows.append({"condition": tag, "model_set": s, "arm": a, "mode": m, "scheme": sc, "scope": scope,
                         **metrics_from_components(comp)})
    M = pd.DataFrame(rows)
    M.to_csv(OUT / f"metrics_{tag}.csv", index=False)

    # ---- reproduction gate: published models on the grid vs the paper's rows for this condition
    ref = pd.read_csv("/home/student/GIT/results/alarms/C1_alarm_metrics_pooled.csv")
    ref = ref[ref.condition == tag].replace({"arm": {"Unified-pretrained": "Chapman-pretrained"}})
    g = M[(M.model_set == "published") & (M.scheme == "grid")]
    chk = ref.merge(g, on=["arm", "mode", "scope"], suffixes=("_paper", "_here"))
    chk["reproduces"] = (chk.TP_paper == chk.TP_here) & (chk.FP_paper == chk.FP_here) & \
                        (chk.FN_paper == chk.FN_here) & (chk.Ignored_paper == chk.Ignored_here)
    chk[["arm", "mode", "scope", "TP_paper", "TP_here", "FP_paper", "FP_here", "FN_paper", "FN_here",
         "Ignored_paper", "Ignored_here", "reproduces"]].to_csv(OUT / f"gate_{tag}.csv", index=False)
    print(f"REPRODUCTION GATE [{tag}]: {int(chk.reproduces.sum())}/{len(chk)} published rows reproduce exactly")
    if not chk.reproduces.all():
        print(chk[~chk.reproduces][["arm", "mode", "scope", "TP_paper", "TP_here", "FP_paper", "FP_here",
                                    "FN_paper", "FN_here", "Ignored_paper", "Ignored_here"]].to_string(index=False))

    # ---- paired record-clustered bootstrap
    rng = np.random.default_rng(20260805)
    recs = K.RECORDS_15
    idx = rng.integers(0, len(recs), size=(10_000, len(recs)))
    contrasts = {"Test1 (beat) - published (grid)": (("test1", "beat"), ("published", "grid")),
                 "training effect: test1 - published, both beat": (("test1", "beat"), ("published", "beat")),
                 "training effect: test1 - published, both grid": (("test1", "grid"), ("published", "grid")),
                 "inference effect: beat - grid, published models": (("published", "beat"), ("published", "grid"))}
    if "beat_end" in schemes:
        contrasts["Test1 (beat, end-anchored) - published (grid)"] = (("test1", "beat_end"), ("published", "grid"))
    brow = []
    for name, (A, B) in contrasts.items():
        for a in ARMS:
            for m in MODES + ["union"]:
                for scope in SCOPES[m]:
                    ca = comps[(A[0], a, m, A[1], scope)].set_index("record").loc[recs]
                    cb = comps[(B[0], a, m, B[1], scope)].set_index("record").loc[recs]
                    cols = list(ca.columns)

                    def f1s(arr):
                        TP, FP, FN = (arr[..., cols.index(k)] for k in ("TP", "FP", "FN"))
                        tTP, tFP, tFN = (arr[..., cols.index(k)] for k in ("tTP", "tFP", "tFN"))
                        ef1 = np.where(2 * TP + FP + FN > 0, 2 * TP / np.maximum(2 * TP + FP + FN, 1e-12), 0)
                        tf1 = np.where(2 * tTP + tFP + tFN > 0, 2 * tTP / np.maximum(2 * tTP + tFP + tFN, 1e-12), 0)
                        return ef1, tf1, TP
                    ea, ta, pa = f1s(ca.to_numpy()[idx].astype(float).sum(axis=1))
                    eb, tb, pb = f1s(cb.to_numpy()[idx].astype(float).sum(axis=1))
                    pa0 = metrics_from_components(ca.reset_index()); pb0 = metrics_from_components(cb.reset_index())
                    for metric, d, point in (("episode_F1", ea - eb, pa0["F1"] - pb0["F1"]),
                                             ("time_F1", ta - tb, pa0["time_F1"] - pb0["time_F1"]),
                                             ("TP", pa - pb, pa0["TP"] - pb0["TP"]),
                                             ("FP", None, pa0["FP"] - pb0["FP"])):
                        if d is None:
                            brow.append({"contrast": name, "arm": a, "mode": m, "scope": scope, "metric": metric,
                                         "delta": point}); continue
                        lo, hi = np.percentile(d, [2.5, 97.5])
                        brow.append({"contrast": name, "arm": a, "mode": m, "scope": scope, "metric": metric,
                                     "delta": point, "ci_lo": lo, "ci_hi": hi, "pct_favour_A": float(np.mean(d > 0) * 100),
                                     "significant": bool(lo > 0 or hi < 0)})
    pd.DataFrame(brow).to_csv(OUT / f"bootstrap_{tag}.csv", index=False)
    print(f"wrote metrics_{tag}.csv, gate_{tag}.csv, bootstrap_{tag}.csv")


class _Cols:
    """Minimal 2-column stand-in so the FSM's `sig[:, idx]` returns the SAME array object each time
    (stable id() for the SQI cache) without copying the record."""
    def __init__(self, c0, c1):
        self.c = (c0, c1)

    def __getitem__(self, key):
        return self.c[key[1]]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["infer", "score"])
    ap.add_argument("--condition", choices=["clean", "noisy", "noisy_sqi"], default="clean")
    ap.add_argument("--threshold", type=float, default=6.0)
    args = ap.parse_args()
    if args.stage == "infer":
        infer("clean" if args.condition == "clean" else "noisy")
    else:
        score(args.condition, args.threshold)
