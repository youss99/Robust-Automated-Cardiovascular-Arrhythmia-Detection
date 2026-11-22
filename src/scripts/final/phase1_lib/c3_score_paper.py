"""Score C3 alarm runs with the paper's scorer (1_ltst_paper_cmbec48/results/neurokit/_code/nk_core.py).

Same rules as score_neurokit.py: s20121 excluded, MANUAL_OMIT mask, record durations from WFDB
headers, modes lead_II / lead_V2 / union, scopes FULL and own-lead.
Episode metrics: match_episodes. Time metrics (merged intervals, whole record): score_time
  duration Se = TP_t / (TP_t + FN_t), duration PPV = TP_t / (TP_t + FP_t), IoU = TP_t / (TP_t + FP_t + FN_t).

Usage: python3 c3_score_paper.py <level> [runs_root]   (level = beat | seg10s_min5 | ...; default root c3_st_configs)
"""
import sys
from pathlib import Path

import pandas as pd

pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/1_ltst_paper_cmbec48/results/neurokit/_code")
import nk_core as K  # noqa: E402

DATA = Path("/home/student/Desktop/Research Data/DATASET NK")
GS_PATH = DATA / "Gold Standard Alarms Original" / "gs_per_beat.csv"
DURATIONS = Path("/home/student/GIT/results/neurokit/_sweep/record_durations.csv")
RUNS = DATA / "C_neurokit" / "c3_st_configs"
REFERENCE = DATA / "NeuroKit Alarms" / "clean" / "neurokit_alarms_seg10s_median.csv"
MODES = {"union": ["FULL"], "lead_II": ["own_II"], "lead_V2": ["own_V2"]}


def score(alarm_csv, gs, durations, mask):
    rows = []
    raw = K.load_alarms(str(alarm_csv))
    for mode, scopes in MODES.items():
        df = raw if mode == "union" or raw.empty else raw[raw["lead"] == mode.split("_")[1]]
        pred = K.apply_mask(df.reset_index(drop=True), mask)
        for scope in scopes:
            g = gs if scope == "FULL" else gs[gs["lead"] == scope.split("_")[1]].reset_index(drop=True)
            p = K.scope_filter(pred, scope)
            r = K.match_episodes(g, p)
            t = K.score_time(g, p, durations, K.universe_pairs(scope, mask))
            rows.append({"mode": mode, "scope": scope, "n_alarms": r.n_pred, "n_gs": r.n_gs,
                         "TP": r.tp, "FP": r.fp, "FN": r.fn, "ep_F1": r.f1,
                         "dur_Se": t["sensitivity"], "dur_PPV": t["PPV_time"], "time_F1": t["time_F1"],
                         "IoU": t["IoU_whole_record"], "spec": t["specificity"]})
    return rows


def main():
    level = sys.argv[1]
    runs_root = Path(sys.argv[2]) if len(sys.argv) > 2 else RUNS
    gs = K.load_gs(str(GS_PATH))
    durations = {r.record: float(r.duration_s) for r in pd.read_csv(DURATIONS).itertuples()}
    mask = set(K.MANUAL_OMIT)
    rows = [{"config": "paper_reference_seg10s", **r} for r in score(REFERENCE, gs, durations, mask)]
    for run in sorted((runs_root / f"runs_{level}").iterdir()):
        rows += [{"config": run.name, **r} for r in score(run / "alarms.csv", gs, durations, mask)]
    out = runs_root / f"results_{level}_paper_scoring.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
