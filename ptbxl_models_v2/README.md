# PTB-XL models v2 — CORRECTED checkpoints

**Supersedes `ptbxl_models.tar`. If you already copied that one, replace it.**

The earlier archive contained a `two_lead fold05` checkpoint produced by an accidental
*resume* (a crashed job was resubmitted into a directory still holding its
`_snapshot.pt`, so the trainer continued instead of starting fresh). That fold has been
retrained from scratch and its checkpoint is different:

| | R² | md5 (first 16) |
|---|---|---|
| old (resumed, **stale**) | 0.131 | `5a80e7937209245c` |
| **new (clean run)** | **0.386** | `aabcfd0cd075568a` |

`lead_II fold04` also crashed and was rerun, but its checkpoint is **bit-identical**
(`de9ad85852f040dd`) — same seed and deterministic training, so resuming reproduced the
same model. Nothing else changed.

Affected aggregate: **two_lead PTB-XL R² 0.4855 ± 0.2133 → 0.5364 ± 0.1408.**

Any predictions you computed with the old fold05 checkpoint should be recomputed.

---

## Contents (241 MB, 18 checkpoints)

```
fine-tune/     15 cv5_optB regression models (5 folds x 3 lead modes)
                 <name>_best.pt, params.txt, _test/_valid_metrics_*.csv
pre-train/     3 PTB-XL backbones they were fine-tuned from
cv5_optB_fold_manifests/   fold01..fold05.csv  <-- WHICH PATIENTS EACH FOLD MAY BE TESTED ON
scripts/       inference + ground-truth helpers (see below)
```

---

## Computing predictions vs. true ST values

### The one rule that matters

**Each fold's model may only be evaluated on that fold's `split == test` patients.**
Every model saw the other 12 records during training, so scoring a fold's model on them
is leakage and will give inflated agreement. The manifests in
`cv5_optB_fold_manifests/foldNN.csv` define the legal test set per fold — filter by
`split == "test"`.

The 15 test patients are partitioned across the 5 folds, so pooling the five filtered
prediction sets gives one honest prediction per patient.

### Where "true" ST comes from

Ground truth is the LTSTDB **`.16a` annotation** — the same source the gold-standard FSM
uses. `scripts/fsm_gold_standard.py::extract_st_arrays_from_ann(ann, "ST0"|"ST1")` returns
`(sample_index, ST_mV)`; ST0 = lead II, ST1 = lead V2. Values are µV in the file and
converted to mV by that function.

You need the LTSTDB records themselves (not in this archive) — `<record>/<record>.16a`
plus `.dat`/`.hea`.

### Getting model predictions

`scripts/ml_inference.py` provides the two calls the alarm pipeline uses:

```python
model, revin, dev = load_patchecg_model(ckpt_path=..., device="cuda", lead_mode="two_lead")
samp_arr, st_by_lead = predict_st_per_segment(
    signal_2lead=sig,          # (n_samples, 2) float32, columns = [II, V2]
    fs=250, model=model, revin=revin, device=dev,
    lead_order=("II","V2"), batch_size=128, lead_mode="two_lead")
# samp_arr  : sample index anchoring each non-overlapping 10 s segment
# st_by_lead: {"II": array, "V2": array} predicted ST in mV, one value per segment
```

`lead_mode` must match the checkpoint: `two_lead` / `lead_II` / `lead_V2`.

### Ready-made

`scripts/analyze_st_trace_behaviour.py` already does all of the above end-to-end —
CV-honest fold filtering, `.16a` alignment, and comparison statistics (OLS slope of
pred~truth, std ratio, amplitude retained inside the |ST| >= 0.1 / 0.2 mV excursion bands,
lag-1 autocorrelation, threshold-crossing counts). Edit the four path constants at the top
(`CKPT_ROOT`, `MANIFEST_DIR`, `LTST_RECORDS`, and the `VARIANTS` dict) to point at your
local copies.

**Caveat: this script has never completed a run.** It was written and then cancelled
before finishing, so treat it as a starting point, not validated code. It was intended to
test whether MSE training shrinks ST excursions — the open question behind
"why does a worse R² model produce better alarms?" in `ALARM_METRIC_CAVEAT.md`.

---

## Checkpoint format

`torch.load(path, map_location="cpu")` -> dict with keys `['opt', 'model']`; weights are
under `'model'`. Architecture (thesis Small): vanilla 1-D ViT, 6 layers, d_model 128,
8 heads, d_ff 512, patch_len 50, stride 50, shared_embedding, ~1.2M params. 10 s segments
at 250 Hz.

Each model's `params.txt` is the authoritative record of how it was trained — trust it
over any generator script.

---

## Where these go

```bash
DEST=/home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection
mkdir -p "$DEST/results/fine-tune/saved_models/ptb-xl_regression"
cp -r fine-tune/*  "$DEST/results/fine-tune/saved_models/ptb-xl_regression/"
mkdir -p "$DEST/results/pre-train/saved_models/ptb-xl"
cp -r pre-train/*  "$DEST/results/pre-train/saved_models/ptb-xl/"
```

Pretrain backbones do **not** belong under `fine-tune/` — keep the two halves separate.

---

## Also still true

Read `summaries/ALARM_METRIC_CAVEAT.md` in the analysis bundle before drawing conclusions
from anything downstream of these models: the episode-level alarm F1 has no specificity
term, and a predictor that alarms everywhere scores F1 = 1.000.
