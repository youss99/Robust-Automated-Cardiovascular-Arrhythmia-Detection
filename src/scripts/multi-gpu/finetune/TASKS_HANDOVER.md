# Finetune work — handover

Context: two-lead arrhythmia classifier. Bedside target is **II + CM5** (stored as `[II, V2]`,
250 Hz). CM5 has no exact 12-lead match, so each track uses a left-lateral precordial surrogate:
**regression → II,V2** (LTST); **classification → II,V5** (PTB-XL). At inference the bedside V2
channel is fed into the slot the classifier learned as V5. Downstream task reuses the multi-label
heads filtered to classes-of-interest ("Option C") — no separate binary head.

Backbone everywhere: PatchECG **thesis-Small** ViT (`d_model=128, n_layers=6, n_heads=8, d_ff=512`,
patch/stride 50, ~1.2M params). Finetune is **BCE by default** (`--no-focal_loss`); the v5 scripts
are pure BCE.

---

## DONE — Task 3: Tier-3 finetune (TTA + Focal + SAM)

**Script:** `src/scripts/multi-gpu/finetune/finetune_classifier_v5_tta_focal_sam.sh`

Drop-in sibling of `finetune_classifier_v5.sh` (same 4-arg interface, same backbone/optimiser).
Only the three low-data/imbalance methods are turned on:
- `--focal_loss --focal_alpha=0.25`
- `--data_augmentation=test_time_aug_transformer` (TTA; chunk_size/step default 250/125)
- `--sam --sam_rho=2 --sam_adaptive` (matches thesis Table A.2)

LoRA deliberately left off (that is the existing `..._c3_focal_tta_lora_sam.sh`). Walltime bumped
to 2h30 because SAM doubles the fwd/bwd per step and TTA adds chunk passes.

**Run (mirror your existing v5 invocations, same backbone ckpts):**
```bash
# Unified-pretrained -> PTB-XL, two-lead
sbatch --job-name=u2p_iiv5_c3 \
  src/scripts/multi-gpu/finetune/finetune_classifier_v5_tta_focal_sam.sh \
  ptb-xl "II,V5" \
  results/pre-train/saved_models/unified/patch_ecg_unified_ii_v5/patch_ecg_unified_ii_v5_best.pt \
  unified2ptbxl_ii_v5_tta_focal_sam

# PTB-XL-pretrained -> Unified, two-lead  (note MIN=15 is applied automatically for unified)
sbatch --job-name=p2u_iiv5_c3 \
  src/scripts/multi-gpu/finetune/finetune_classifier_v5_tta_focal_sam.sh \
  unified "II,V5" \
  results/pre-train/saved_models/ptb-xl/patch_ecg_ptbxl_ii_v5/patch_ecg_ptbxl_ii_v5_best.pt \
  ptbxl2unified_ii_v5_tta_focal_sam
```
**Compare against** the matching BCE-only v5 runs already in
`results/fine-tune/saved_models/{ptb-xl,unified}/` (e.g. `unified2ptbxl_ii_v5_bce` AUROC 0.8604).
Expectation from the thesis ablations: meaningful gain on the smaller/imbalanced finetune (PTB-XL),
~neutral on the larger Unified set. Keep `--seed=200` fixed so the delta vs the v5 run is clean.

---

## TODO — remaining tasks + open decisions

### Task 2 — 12-lead baseline (cheap, do next; runs on the 20GB MIG slice)
Goal: the within-experiment 2-lead-vs-12-lead ceiling (we currently only have the paper's 0.9031).
Needs **two** runs: an `all_leads` **pretrain** backbone **+** an `all_leads` **finetune** (BCE, v5).
- Decision: finetune on **PTB-XL** (to compare directly to paper 0.9031). Confirm whether an
  `all_leads` Unified-pretrained Small backbone already exists, else pretrain one first
  (`pretrain_classifier_v5.sh unified all_leads <name>`).

### Task 1 — Regression backbone: pretrain on CODE, **base** params, II,V2
Upgrades the current regression backbone (**Chapman 45k, Small** — `patch_ecg_chapman_ii_v2_tta`;
this resolves the "chapman vs Unified" naming note: it is Chapman, a *subset* of Unified, not full
Unified). Base = 85M params (`d_model=768, n_layers=12, n_heads=12, d_ff=3072`, 100 epochs).
**Resource jump — blocked on decisions:**
1. GPU partition: full A100 (40/80GB) or only `a100_3g.20gb` MIG? Sets batch size / feasibility.
2. Pretrain on 2 leads (II,V2) or all 12 CODE leads then finetune on II,V2? (Lean: all 12 — paper's
   recipe, richer signal; shared-embedding backbone handles a lead subset at finetune.)
3. CODE-15 (~345k) or full CODE (~2.3M) for this first pass?
Will need checkpoint-resume chaining (multi-day). Base ref: `pretrain/patch_ecg.sh`.

### Task 4 — Classification "big gun": full CODE, base params, then best-params finetune
Same resource story as Task 1, larger (full CODE), II,V5 → finetune with the Tier-3 stack above.
Gated on Tasks 2/3 not being enough.

### Not in the four tasks but the real bedside-precision levers (from HOMER slideshow)
- **SQI 6 dB gating** is currently informational-only; turning it on took PVC precision 0.37→0.86.
- **`combine` (two single-lead models, max-union) beat the joint two-lead model** for PVC recall on
  bedside data (26 vs 11 TP) under domain shift — keep the single-lead models.

---

## Git workflow
Scripts are authored locally and cannot be run/verified here (data + GPUs are cluster-side).
Branch: `update-initial-changes-with-Mitch` → push to `my-fork`, pull on the cluster, `sbatch`.
