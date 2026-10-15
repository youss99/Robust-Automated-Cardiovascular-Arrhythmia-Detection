#!/usr/bin/env python3
"""
Emit the exact test-only command to re-run a finished model and dump its logits.

Reads the model's own params.txt (written at training time) so the model is
rebuilt identically -- dims, lead selection, LoRA config, class filtering.

  python vm/make_export_cmd.py --data-root /data \
      results/fine-tune/saved_models/ptb-xl/unified2ptbxl_ii_bce

Add --run to execute instead of printing.
"""
import argparse, os, re, subprocess, sys

# params.txt entry -> CLI flag. Everything else keeps its argparse default.
STR_KEYS = ["dset_finetune", "data_sub_path", "custom_lead_selection",
            "model_name", "scaler", "features", "revin_mode"]
# (save_path is deliberately absent: the trainer derives it from dset + model_name.)
NUM_KEYS = ["n_layers", "n_heads", "d_model", "d_ff", "patch_len", "stride",
            "context_points", "batch_size", "min_class_size",
            "lora_r", "lora_alpha", "lora_dropout",
            "seed", "chunk_size", "chunk_step"]           # chunk_* = TTA window geometry
# Enum-valued settings. params.txt stores them as "<Enum.NAME: n>"; the trainer wants the
# lowercase NAME. data_augmentation is the critical one: dropping it silently scores
# TTA-trained models WITHOUT TTA (different predictions, wrong numbers).
ENUM_KEYS = ["data_augmentation", "class_token"]
# dataset dir name on the VM, per dset_finetune value
DSET_DIR = {"ptb-xl": "ptb-xl_1.0.3", "unified": "Unified_Dataset_No_PTB_Segment",
            "chapman": "chapman", "cinc-2020": "cinc-2020"}


def get(txt, key):
    m = re.search(rf"\b{key}=('([^']*)'|[^,)\s]+)", txt)
    if not m:
        return None
    return m.group(2) if m.group(2) is not None else m.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="results/fine-tune/saved_models/<target>/<model>")
    ap.add_argument("--data-root", required=True,
                    help="parent dir holding ptb-xl/ and Unified_Dataset_No_PTB_Segment/ on this machine")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    p = os.path.join(a.model_dir, "params.txt")
    if not os.path.exists(p):
        sys.exit(f"no params.txt in {a.model_dir}")
    txt = open(p).read()

    dset = get(txt, "dset_finetune")
    root = os.path.join(a.data_root, DSET_DIR.get(dset, dset))
    if not os.path.isdir(root):
        print(f"WARNING: {root} not found -- fix --data-root or copy the dataset", file=sys.stderr)

    cmd = ["python", "-m", "src.patchECG_finetune",
           "--no-finetuning", "--testing", "--dump_logits",
           "--distributed" if get(txt, "distributed") == "True" else "--no-distributed",
           f"--root_path={root}"]

    for k in STR_KEYS:
        v = get(txt, k)
        if v not in (None, "None") and k != "dset_finetune":
            cmd.append(f"--{k}={v}")
    cmd.append(f"--dset_finetune={dset}")

    for k in NUM_KEYS:
        v = get(txt, k)
        if v not in (None, "None"):
            cmd.append(f"--{k}={v}")

    for k in ENUM_KEYS:
        m = re.search(rf"\b{k}=<\w+\.(\w+):", txt)
        if m:
            cmd.append(f"--{k}={m.group(1).lower()}")
    if get(txt, "metric_by_class") == "True":
        cmd.append("--metric_by_class")
    if get(txt, "shared_embedding") == "True":
        cmd.append("--shared_embedding")
    if get(txt, "lora") == "True":
        cmd.append("--lora")
    top_n = get(txt, "top_n_classes")
    if top_n not in (None, "None"):
        cmd.append(f"--top_n_classes={top_n}")

    # pretrained_model_path is deliberately omitted: test_func loads <model>_best.pt
    # over whatever the backbone initialised, so the pretrain checkpoints are not
    # needed on the VM.
    line = " ".join(cmd[:3]) + " \\\n    " + " \\\n    ".join(cmd[3:])
    if a.run:
        print(f"$ {line}\n", flush=True)
        sys.exit(subprocess.call(cmd))
    print(line)


if __name__ == "__main__":
    main()
