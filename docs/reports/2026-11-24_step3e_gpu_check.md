# step3e GPU check

2026-11-24 00:28, VM (NVIDIA A16), fold 1, S1 (seed 200) unless stated. Work folder: `/tmp/claude-1000/-home-student-GIT/dfa42d17-c0b1-4a95-a5f7-13cc2acca044/scratchpad/e2e/e2e_check_h8189_ga`.

Cases (each a fresh process running the real trainer with the generated job's flags):

- lead_II_new: FIN_L1supE2E_lead_II_S1_fold01.sbatch (--seed=200 --custom_lead_selection=II --n_epochs_finetune_head=0 --n_epochs_finetune=90), 39 s, run folder _VMCHECK_e2e_lead_II_new_20261124002458
- lead_II_twin: FIN_L1sup_lead_II_S1_fold01.sbatch (--seed=200 --custom_lead_selection=II --n_epochs_finetune_head=20 --n_epochs_finetune=70), 24 s, run folder _VMCHECK_e2e_lead_II_twin_20261124002458
- lead_V2_new: FIN_L1supE2E_lead_V2_S1_fold01.sbatch (--seed=200 --custom_lead_selection=V2 --n_epochs_finetune_head=0 --n_epochs_finetune=90), 39 s, run folder _VMCHECK_e2e_lead_V2_new_20261124002458
- lead_V2_twin: FIN_L1sup_lead_V2_S1_fold01.sbatch (--seed=200 --custom_lead_selection=V2 --n_epochs_finetune_head=20 --n_epochs_finetune=70), 25 s, run folder _VMCHECK_e2e_lead_V2_twin_20261124002458
- two_lead_new: FIN_L2supE2E_two_lead_S1_fold01.sbatch (--seed=200 --custom_lead_selection=II,V2 --n_epochs_finetune_head=0 --n_epochs_finetune=90), 51 s, run folder _VMCHECK_e2e_two_lead_new_20261124002458
- two_lead_twin: FIN_L2sup_two_lead_S1_fold01.sbatch (--seed=200 --custom_lead_selection=II,V2 --n_epochs_finetune_head=20 --n_epochs_finetune=70), 29 s, run folder _VMCHECK_e2e_two_lead_twin_20261124002458
- lead_II_new_seed999: FIN_L1supE2E_lead_II_S1_fold01.sbatch (--seed=999 --custom_lead_selection=II --n_epochs_finetune_head=0 --n_epochs_finetune=90), 12 s, run folder _VMCHECK_e2e_lead_II_new_seed999_20261124002458

## 1. Same seed, same initial weights

| mode | step3e job vs twin | tensors | values |
|---|---|--:|--:|
| lead_II | identical | 79 | 1,197,541 |
| lead_V2 | identical | 79 | 1,197,541 |
| two_lead | identical | 79 | 1,200,230 |

Control, lead II seed 200 vs seed 999: different, as expected.

## 2. Optimizer when Learner.fine_tune starts

| case | groups | lr | weight decay | lr_scale | all parameters | type |
|---|--:|---|---|---|---|---|
| lead_II_new | 1 | [0.001] | [0.05] | [1] | yes (79 tensors, 1,197,541 values) | AdamW |
| lead_II_twin | 1 | [0.001] | [0.05] | [1] | yes (79 tensors, 1,197,541 values) | AdamW |
| lead_V2_new | 1 | [0.001] | [0.05] | [1] | yes (79 tensors, 1,197,541 values) | AdamW |
| lead_V2_twin | 1 | [0.001] | [0.05] | [1] | yes (79 tensors, 1,197,541 values) | AdamW |
| two_lead_new | 1 | [0.001] | [0.05] | [1] | yes (79 tensors, 1,200,230 values) | AdamW |
| two_lead_twin | 1 | [0.001] | [0.05] | [1] | yes (79 tensors, 1,200,230 values) | AdamW |
| lead_II_new_seed999 | 1 | [0.001] | [0.05] | [1] | yes (79 tensors, 1,197,541 values) | AdamW |

## 3-4. Gradients at the first optimizer step and changes after epoch 0

| case | epochs (head, full) | expected to train | parameters not as expected | trained tensors | min share of non-zero gradient values (first step) | min share of changed values (epoch 0) | mask_token: grad / changed | valid loss epoch 0 |
|---|---|---|---|--:|--:|--:|---|--:|
| lead_II_new | 0, 90 | all but mask_token | none | 78 | 1.000 | 1.000 | False / False | 0.004760 |
| lead_II_twin | 20, 70 | mlp_head.* only (control) | none | 4 | 1.000 | 1.000 | False / False | 0.004262 |
| lead_V2_new | 0, 90 | all but mask_token | none | 78 | 1.000 | 1.000 | False / False | 0.001557 |
| lead_V2_twin | 20, 70 | mlp_head.* only (control) | none | 4 | 1.000 | 1.000 | False / False | 0.002103 |
| two_lead_new | 0, 90 | all but mask_token | none | 78 | 1.000 | 1.000 | False / False | 0.003720 |
| two_lead_twin | 20, 70 | mlp_head.* only (control) | none | 4 | 1.000 | 1.000 | False / False | 0.003197 |

Per-parameter detail, lead II step3e job, first optimizer step (gradient norm, share of non-zero values):

- `pos_embedding`: 4.473e-01, 1.000
- `cls_token`: 4.435e-01, 1.000
- `mask_token`: no gradient
- `to_patch_embedding.0.weight`: 1.972e-02, 1.000
- `to_patch_embedding.0.bias`: 2.238e-01, 1.000
- `to_patch_embedding.1.weight`: 2.491e-01, 1.000
- `to_patch_embedding.1.bias`: 4.160e-01, 1.000
- `to_patch_embedding.2.weight`: 4.262e-02, 1.000
- `to_patch_embedding.2.bias`: 2.450e-01, 1.000
- `transformer.layers.0.0.norm.weight`: 2.959e-02, 1.000
- `transformer.layers.0.0.norm.bias`: 1.662e-01, 1.000
- `transformer.layers.0.0.to_qkv.weight`: 6.576e-01, 1.000
- `transformer.layers.0.0.to_out.0.weight`: 5.507e-01, 1.000
- `transformer.layers.0.0.to_out.0.bias`: 4.998e-01, 1.000
- `transformer.layers.0.1.net.0.weight`: 7.821e-02, 1.000
- `transformer.layers.0.1.net.0.bias`: 9.702e-02, 1.000
- `transformer.layers.0.1.net.1.weight`: 1.716e+00, 1.000
- `transformer.layers.0.1.net.1.bias`: 1.646e-01, 1.000
- `transformer.layers.0.1.net.4.weight`: 3.447e+00, 1.000
- `transformer.layers.0.1.net.4.bias`: 4.993e-01, 1.000
- `transformer.layers.1.0.norm.weight`: 3.628e-02, 1.000
- `transformer.layers.1.0.norm.bias`: 1.314e-01, 1.000
- `transformer.layers.1.0.to_qkv.weight`: 6.212e-01, 1.000
- `transformer.layers.1.0.to_out.0.weight`: 6.777e-01, 1.000
- `transformer.layers.1.0.to_out.0.bias`: 4.838e-01, 1.000
- `transformer.layers.1.1.net.0.weight`: 9.558e-02, 1.000
- `transformer.layers.1.1.net.0.bias`: 1.064e-01, 1.000
- `transformer.layers.1.1.net.1.weight`: 1.814e+00, 1.000
- `transformer.layers.1.1.net.1.bias`: 1.741e-01, 1.000
- `transformer.layers.1.1.net.4.weight`: 3.672e+00, 1.000
- `transformer.layers.1.1.net.4.bias`: 4.854e-01, 1.000
- `transformer.layers.2.0.norm.weight`: 4.444e-02, 1.000
- `transformer.layers.2.0.norm.bias`: 1.824e-01, 1.000
- `transformer.layers.2.0.to_qkv.weight`: 7.460e-01, 1.000
- `transformer.layers.2.0.to_out.0.weight`: 6.867e-01, 1.000
- `transformer.layers.2.0.to_out.0.bias`: 4.634e-01, 1.000
- `transformer.layers.2.1.net.0.weight`: 9.557e-02, 1.000
- `transformer.layers.2.1.net.0.bias`: 9.227e-02, 1.000
- `transformer.layers.2.1.net.1.weight`: 1.788e+00, 1.000
- `transformer.layers.2.1.net.1.bias`: 1.606e-01, 1.000
- `transformer.layers.2.1.net.4.weight`: 3.295e+00, 1.000
- `transformer.layers.2.1.net.4.bias`: 4.591e-01, 1.000
- `transformer.layers.3.0.norm.weight`: 3.280e-02, 1.000
- `transformer.layers.3.0.norm.bias`: 1.309e-01, 1.000
- `transformer.layers.3.0.to_qkv.weight`: 7.319e-01, 1.000
- `transformer.layers.3.0.to_out.0.weight`: 6.621e-01, 1.000
- `transformer.layers.3.0.to_out.0.bias`: 4.470e-01, 1.000
- `transformer.layers.3.1.net.0.weight`: 8.744e-02, 1.000
- `transformer.layers.3.1.net.0.bias`: 9.304e-02, 1.000
- `transformer.layers.3.1.net.1.weight`: 1.799e+00, 1.000
- `transformer.layers.3.1.net.1.bias`: 1.631e-01, 1.000
- `transformer.layers.3.1.net.4.weight`: 3.399e+00, 1.000
- `transformer.layers.3.1.net.4.bias`: 4.378e-01, 1.000
- `transformer.layers.4.0.norm.weight`: 3.696e-02, 1.000
- `transformer.layers.4.0.norm.bias`: 1.266e-01, 1.000
- `transformer.layers.4.0.to_qkv.weight`: 6.396e-01, 1.000
- `transformer.layers.4.0.to_out.0.weight`: 6.981e-01, 1.000
- `transformer.layers.4.0.to_out.0.bias`: 4.387e-01, 1.000
- `transformer.layers.4.1.net.0.weight`: 8.348e-02, 1.000
- `transformer.layers.4.1.net.0.bias`: 8.543e-02, 1.000
- `transformer.layers.4.1.net.1.weight`: 1.707e+00, 1.000
- `transformer.layers.4.1.net.1.bias`: 1.559e-01, 1.000
- `transformer.layers.4.1.net.4.weight`: 3.218e+00, 1.000
- `transformer.layers.4.1.net.4.bias`: 4.382e-01, 1.000
- `transformer.layers.5.0.norm.weight`: 3.402e-02, 1.000
- `transformer.layers.5.0.norm.bias`: 1.499e-01, 1.000
- `transformer.layers.5.0.to_qkv.weight`: 7.084e-01, 1.000
- `transformer.layers.5.0.to_out.0.weight`: 7.389e-01, 1.000
- `transformer.layers.5.0.to_out.0.bias`: 4.271e-01, 1.000
- `transformer.layers.5.1.net.0.weight`: 7.909e-02, 1.000
- `transformer.layers.5.1.net.0.bias`: 7.818e-02, 1.000
- `transformer.layers.5.1.net.1.weight`: 1.647e+00, 1.000
- `transformer.layers.5.1.net.1.bias`: 1.463e-01, 1.000
- `transformer.layers.5.1.net.4.weight`: 3.214e+00, 1.000
- `transformer.layers.5.1.net.4.bias`: 4.310e-01, 1.000
- `mlp_head.layer_norm.weight`: 6.496e-01, 1.000
- `mlp_head.layer_norm.bias`: 6.575e-01, 1.000
- `mlp_head.fc.weight`: 1.211e+01, 1.000
- `mlp_head.fc.bias`: 1.076e+00, 1.000

## Result: ALL PASS

Checks: init lead_II: pass; init lead_V2: pass; init two_lead: pass; init control (seed 999 differs): pass; optimizer lead_II_new: pass; optimizer lead_II_twin: pass; optimizer lead_V2_new: pass; optimizer lead_V2_twin: pass; optimizer two_lead_new: pass; optimizer two_lead_twin: pass; optimizer lead_II_new_seed999: pass; grad/update lead_II_new: pass; grad/update lead_II_twin: pass; grad/update lead_V2_new: pass; grad/update lead_V2_twin: pass; grad/update two_lead_new: pass; grad/update two_lead_twin: pass.
