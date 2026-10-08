"""
Generate ST alarms using ML-predicted ST values (PatchECG vanilla_vit
regression, ltstdb_two_lead_regression_supervised_no_tta).

Mirrors old_fsm_gs.py: per record, per lead, runs the existing FSM
(`run_fsm_on_record`). The only difference is where samp_arr / st_arr come
from -- here they come from one batched forward pass of the ViT over all
non-overlapping 10s windows of the record (250 Hz -> 100 Hz inside the
inference helper). Trailing samples that don't fill a full 10s window are
dropped.

Run from the ltst_pipeline directory:

    cd /home/student/GIT/Neurokit2_Pipeline/ltst_pipeline
    python new_fsm_ml.py --base_path "/home/student/Desktop/Research Data/DATASET NK/ltst_records" \
                        --out_path ./ml_alarms.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import wfdb

# Try package-relative import first (when invoked as `python -m ltst_pipeline.new_fsm_ml`
# or imported from `pipeline.py`); fall back to absolute import for direct script use.
try:
    from .old_fsm_gs import (  # type: ignore
        LTSTDB_RECORDS_DEFAULT,
        MANUAL_OMIT,
        STTAG_TO_LEAD_IDX,
        STTAG_TO_PAPER_LEAD,
        USE_MANUAL_OMIT_DEFAULT,
        Alarm,
        get_rpeaks_from_atr,
        run_fsm_on_record,
    )
    from .ml_inference import (  # type: ignore
        DEFAULT_CKPT,
        SEGMENT_SECONDS,
        load_patchecg_model,
        predict_st_per_segment,
    )
except ImportError:
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from old_fsm_gs import (  # noqa: E402
        LTSTDB_RECORDS_DEFAULT,
        MANUAL_OMIT,
        STTAG_TO_LEAD_IDX,
        STTAG_TO_PAPER_LEAD,
        USE_MANUAL_OMIT_DEFAULT,
        Alarm,
        get_rpeaks_from_atr,
        run_fsm_on_record,
    )
    from ml_inference import (  # noqa: E402
        DEFAULT_CKPT,
        SEGMENT_SECONDS,
        load_patchecg_model,
        predict_st_per_segment,
    )


BASE_PATH_DEFAULT = "ltst_records"
FS_DEFAULT = 250
STRATEGY_DEFAULT = "ml_regression_10s"


def generate_ml_alarms(
    base_path: str,
    output_csv: str,
    fs: int = FS_DEFAULT,
    use_sqi: bool = True,
    sqi_threshold_db: float = 5.0,
    sqi_every_s: float = 5.0,
    sqi_window_s: float = 30.0,
    use_manual_omit: bool = True,
    ckpt_path: Optional[str] = None,
    device: str = "cpu",
    batch_size: int = 64,
    strategy_label: str = STRATEGY_DEFAULT,
    records: Optional[list[str]] = None,
    apply_noise: bool = False,
    noise_seed: int = 42,
    lead_mode: str = "two_lead",
) -> None:
    out_path = Path(output_csv)
    all_alarms: list[Alarm] = []

    print(f"⚙️   Loading PatchECG model on device={device}, lead_mode={lead_mode} …")
    model, revin, dev = load_patchecg_model(
        ckpt_path=ckpt_path if ckpt_path else DEFAULT_CKPT,
        device=device,
        lead_mode=lead_mode,
    )
    print("✅  Model ready\n")

    if apply_noise:
        try:
            from .noise import contaminate_ecg_with_motion_artifact  # type: ignore
        except ImportError:
            from noise import contaminate_ecg_with_motion_artifact  # type: ignore
        print(f"🌪️   apply_noise=True: motion artifact contamination enabled (seed={noise_seed})\n")

    record_ids = records if records is not None else LTSTDB_RECORDS_DEFAULT

    for record in record_ids:
        print(f"⏳  {record}")
        record_path = f"{base_path}/{record}/{record}"

        try:
            rec = wfdb.rdrecord(record_path)
            atr = wfdb.rdann(record_path, "atr")
        except Exception as exc:
            print(f"   → SKIPPED (read error: {exc})")
            continue

        if rec.p_signal is None or rec.p_signal.ndim != 2 or rec.p_signal.shape[1] < 2:
            print(f"   → SKIPPED (need 2 leads, got shape {None if rec.p_signal is None else rec.p_signal.shape})")
            continue

        r_peaks = get_rpeaks_from_atr(atr)

        p_signal = rec.p_signal
        if apply_noise:
            p_signal, _ = contaminate_ecg_with_motion_artifact(p_signal, seed=noise_seed, fs=fs)
            print(f"   🌪️   contaminated {p_signal.shape[0]} samples with motion artifact")

        # Column 0 = lead II, column 1 = lead V2 (LTSTDB convention)
        sig_2lead = p_signal[:, [STTAG_TO_LEAD_IDX["ST0"], STTAG_TO_LEAD_IDX["ST1"]]].astype(np.float32)

        samp_arr, st_by_lead = predict_st_per_segment(
            signal_2lead=sig_2lead,
            fs=fs,
            model=model,
            revin=revin,
            device=dev,
            lead_order=("II", "V2"),
            batch_size=batch_size,
            lead_mode=lead_mode,
        )
        n_segs = len(samp_arr)
        if n_segs == 0:
            print(f"   → SKIPPED (record shorter than {fs * SEGMENT_SECONDS} samples)")
            continue
        print(f"   → {n_segs} non-overlapping {SEGMENT_SECONDS}s segments")

        for lead_tag in ("ST0", "ST1"):
            paper_lead = STTAG_TO_PAPER_LEAD[lead_tag]
            if use_manual_omit and ((record, paper_lead) in MANUAL_OMIT):
                print(f"     · {lead_tag} ({paper_lead}): SKIPPED (manual omit)")
                continue
            if paper_lead not in st_by_lead:
                # Single-lead model didn't predict this lead -> no alarms for it.
                continue

            ecg_for_sqi = p_signal[:, STTAG_TO_LEAD_IDX[lead_tag]].astype(float)
            st_arr_lead = st_by_lead[paper_lead].tolist()

            rec_alarms = run_fsm_on_record(
                rec_id=record,
                lead_tag=lead_tag,
                samp_arr=samp_arr.tolist(),
                st_arr=st_arr_lead,
                ecg_sig_for_sqi=ecg_for_sqi,
                r_peaks=r_peaks,
                fs=fs,
                use_sqi=use_sqi,
                sqi_threshold_db=sqi_threshold_db,
                sqi_every_s=sqi_every_s,
                sqi_window_s=sqi_window_s,
            )
            for a in rec_alarms:
                a.strategy = strategy_label

            print(f"     · {lead_tag} ({paper_lead}): {len(rec_alarms)} alarms")
            all_alarms.extend(rec_alarms)

    print(f"\nTotal # of alarms: {len(all_alarms)}")

    df = pd.DataFrame([{
        "record": a.record,
        "lead_tag": a.lead_tag,
        "lead_paper": a.lead_paper,
        "strategy": getattr(a, "strategy", strategy_label),

        "onset_breach_sample": a.onset_breach,
        "onset_expected_sample": a.onset_expected,
        "onset_actual_sample":   a.onset_actual,
        "end_expected_sample":   a.end_expected,
        "end_actual_sample":     a.end_actual,

        "onset_breach_s":   a.onset_breach   / fs,
        "onset_expected_s": a.onset_expected / fs,
        "onset_actual_s":   a.onset_actual   / fs,
        "end_expected_s":   a.end_expected   / fs,
        "end_actual_s":     a.end_actual     / fs,

        "duration_expected_s": (a.end_expected - a.onset_expected) / fs,
        "duration_actual_s":   (a.end_actual   - a.onset_actual)   / fs,
        "warmup_duration_s":   (a.onset_actual - a.onset_breach)   / fs,

        "criterion": a.crit,
    } for a in all_alarms])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"💾  Saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ML-driven ST alarm generation (10s, 2-lead PatchECG ViT regression)."
    )
    p.add_argument("--base_path", type=str, default=BASE_PATH_DEFAULT,
                   help="Path to ltst_records root.")
    p.add_argument("--out_path", type=str, required=True, help="Output CSV path.")
    p.add_argument("--fs", type=int, default=FS_DEFAULT, help="Record sampling frequency (Hz).")
    p.add_argument("--ckpt_path", type=str, default=None,
                   help="Override the model checkpoint path. Defaults to DEFAULT_CKPT.")
    p.add_argument("--device", type=str, default="cpu", help="torch device (cpu / cuda / cuda:0).")
    p.add_argument("--batch_size", type=int, default=64, help="Segments per forward pass.")
    p.add_argument("--disable_sqi", action="store_true", help="Disable SQI gating.")
    p.add_argument("--sqi_threshold_db", type=float, default=5.0)
    p.add_argument("--sqi_every_s", type=float, default=5.0)
    p.add_argument("--sqi_window_s", type=float, default=30.0)
    p.add_argument("--use_manual_omit", action="store_true", default=USE_MANUAL_OMIT_DEFAULT,
                   help="Use the MANUAL_OMIT list (default on).")
    p.add_argument("--strategy_label", type=str, default=STRATEGY_DEFAULT,
                   help="Tag attached to every alarm in the output CSV.")
    p.add_argument("--records", type=str, nargs="*", default=None,
                   help="Optional subset of LTSTDB record IDs to process.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    generate_ml_alarms(
        base_path=args.base_path,
        output_csv=args.out_path,
        fs=int(args.fs),
        use_sqi=not args.disable_sqi,
        sqi_threshold_db=float(args.sqi_threshold_db),
        sqi_every_s=float(args.sqi_every_s),
        sqi_window_s=float(args.sqi_window_s),
        use_manual_omit=args.use_manual_omit,
        ckpt_path=args.ckpt_path,
        device=args.device,
        batch_size=int(args.batch_size),
        strategy_label=args.strategy_label,
        records=args.records,
    )


if __name__ == "__main__":
    main()
