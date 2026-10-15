# VM runbook — what you can answer without retraining

Everything here runs on a single GPU VM (inference) or plain CPU (analysis).
Nothing needs the cluster. Scope: **single-lead II and V5, PVC and PAC.**

---

## What to copy to the VM

| | Size | Why |
|---|---|---|
| the repo | small | code |
| `results/fine-tune/saved_models/{ptb-xl,unified}/**/​*_best.pt` + `params.txt` | ~1.0 GB | the 8 models under test |
| `Datasets/ptb-xl_1.0.3/` | 730 MB | PTB-XL evaluation |
| `Datasets/Unified_Dataset_No_PTB_Segment/` | 1.9 GB | Unified evaluation |

~4 GB. **Do not copy** `_snapshot.pt` / `_every_20_epoch.pt` (2.1 GB, useless downstream),
or `Datasets/CODE/` (310 GB).

```bash
rsync -av --include='*/' --include='*_best.pt' --include='params.txt' --exclude='*' \
      <cluster>:.../results/fine-tune/saved_models/ ./results/fine-tune/saved_models/
```

The pre-train checkpoints are **not** needed: `test_func` loads `<model>_best.pt` over
whatever initialised the backbone, so `make_export_cmd.py` omits `--pretrained_model_path`.

---

## Phase 1 — dump logits (GPU, ~minutes per model, once)

Per-sample outputs were never written to disk; only aggregate metrics were. The
`--dump_logits` flag added to `src/patchECG_finetune.py` fixes that. `make_export_cmd.py`
reads each model's own `params.txt` so it is rebuilt identically (dims, leads, LoRA).

```bash
export DATA=/path/to/Datasets        # holds ptb-xl_1.0.3/ and Unified_Dataset_No_PTB_Segment/

for t in ptb-xl unified; do
  for d in results/fine-tune/saved_models/$t/*_ii_bce \
           results/fine-tune/saved_models/$t/*_ii_tta_focal_sam \
           results/fine-tune/saved_models/$t/*_v5_bce \
           results/fine-tune/saved_models/$t/*_v5_tta_focal_sam; do
    [ -d "$d" ] || continue
    echo "=== $d"
    python vm/make_export_cmd.py --data-root "$DATA" --run "$d"
  done
done
```

Writes `_logits_<model>_valid.npz` and `_logits_<model>_test.npz` beside each checkpoint,
each holding `logits`, `targets`, `classes`.

Inspect one command before running the loop:
```bash
python vm/make_export_cmd.py --data-root "$DATA" \
    results/fine-tune/saved_models/ptb-xl/unified2ptbxl_ii_bce
```

---

## Phase 2 — analysis (CPU only, seconds, re-runnable)

### 2a. Threshold — the big one

`src/core/support/tracking.py:377-387` hardcodes `threshold=0.5`. Every focal-loss model
is scored at an operating point it was never trained to hit. Thresholds are tuned on
**valid** and reported on **test**.

```bash
python vm/sweep_thresholds.py --classes PVC,PAC \
    results/fine-tune/saved_models/ptb-xl/*/_logits_*_test.npz
```

The cleanest prediction to check: `unified2ptbxl_v5_tta_focal_sam` has **higher** AUROC
than `unified2ptbxl_v5_bce` (0.956 vs 0.941) yet F1 0.067 vs 0.460. If thresholding is
the whole story, tuning should invert that ordering.

### 2b. Synonym merge — tests the taxonomy hypothesis without retraining

Unified splits one clinical concept across SNOMED codes: `PVC`+`VPB`+`VEB` (2,438 labels)
and `PAC`+`SVPB`+`BPAC` (2,459). On PTB-XL, where PVC is a single class, the best F1 is
0.781; on Unified it is 0.080. `--merge` combines the existing output units with noisy-OR
and rescores against merged ground truth.

```bash
python vm/sweep_thresholds.py --merge \
    results/fine-tune/saved_models/unified/*/_logits_*_test.npz
```

**If this does not recover performance, retraining on merged labels will not either** —
that is the point of running it first.

### 2c. Ensemble II + V5 — free

```bash
python vm/sweep_thresholds.py --ensemble --classes PVC,PAC \
    results/fine-tune/saved_models/ptb-xl/unified2ptbxl_ii_bce/_logits_*_test.npz \
    results/fine-tune/saved_models/ptb-xl/unified2ptbxl_v5_bce/_logits_*_test.npz
```

---

## Phase 3 — still GPU, still no training

**TTA on existing checkpoints.** TTA is applied in `test_func`, not during training, so it
works on any finished model. Append to a Phase 1 command:

```
--data_augmentation=test_time_aug_transformer
```

In the thesis, LoRA → LoRA+TTA was worth **+0.0125 AUROC** on PTB-XL (Table 5.2, b vs c).

**Evaluate the two `c3` LoRA models.** `patch_ecg_{code2ptbxl,code2unified}_ii_v2_c3_focal_tta_lora_sam`
were trained and never evaluated — no per-class metrics exist. They are the thesis's `d`
config (LoRA+SAM+Focal+TTA), its best PatchECG result. They are `ii_v2`, so not single-lead,
but they tell you whether LoRA is worth cluster time.

---

## What still needs the cluster

| | Why |
|---|---|
| **LoRA fine-tune at II and V5** | thesis `d`; beats full fine-tuning by 0.035–0.07 AUROC (0.8889 vs 0.9238–0.9385). You have **no single-lead LoRA model** |
| **Unified merged-label retrain** | only if 2b shows promise |
| **MentorMix** | `src/scripts/multi-gpu/finetune/multi_label/ptb-xl/patch_ecg_mentor_mix.sh` exists; enabled in zero evaluated models |
| **Unified + Base backbone** | the capacity test — ~15 GPU-h, vs ~325 for CODE+Base |

---

## Caveat on PAC

Every single-lead PAC F1 is currently **0.000**, and PTB-XL's test set has **n=40**.
Threshold tuning will produce a nonzero number, but it will be unstable at that support.

**Do PAC work on Unified** (PAC n=235 test, 2,343 labelled) and report **AUPRC with
confidence intervals**, not F1. PVC is where single-lead genuinely responds
(II 0.670, V5 0.460).
