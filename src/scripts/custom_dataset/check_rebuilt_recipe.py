#!/usr/bin/env python3
"""Run on the cluster from the repo root, after the _rebuilt backbones finish: python /path/to/check_rebuilt_recipe.py
Part B item 7: do the _rebuilt backbones match the recipe of the _only backbones the published LTST models used, with only the data changed?
Builds the full _only recipe by running the real argument parser of src/patchECG_pretrain.py on the flags of the matching _only script, src/scripts/multi-gpu/pretrain/unified/unified_{ii_only,v2_only,ii_v2}.sh,
so the defaults those runs relied on are included, and compares every value of _rebuilt/params.txt with it except names and paths, which are printed. Run it with the repo's python.
If the parser cannot be run, it falls back to the script's flags (or the typed table below if the script is missing) and lists the remaining values as "not compared".
Also compares with _only's own params.txt where it exists on this cluster. Prints the data record and whether the two patch-embedding LayerNorms are present. Reads only."""
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path("results/pre-train/saved_models/unified")
RECIPE = dict(min_class_size="50", context_points="5000", target_points="100", scaler="'standard'", features="'M'", patch_len="50", stride="50", revin_mode="'BN'", n_layers="6", n_heads="8", d_model="128", d_ff="512", shared_embedding="True", dropout="0.0", head_dropout="0.0", trafos="['None']", data_augmentation="none", diagnostic_class="all", mask_ratio="0.4", zero_noise_mixing="0.5", opt="'adamw'", weight_decay="0.05", lr="0.0015", scheduler="True", warmup_epochs="2", warmup_lr="1e-06", min_lr="1e-05", batch_size="128", n_epochs_pretrain="500")
PAIRS = {"ii": ("patch_ecg_unified_ii_only", "patch_ecg_unified_ii_rebuilt", "'II'"), "v2": ("patch_ecg_unified_v2_only", "patch_ecg_unified_v2_rebuilt", "'V2'"), "ii_v2": ("patch_ecg_unified_ii_v2", "patch_ecg_unified_ii_v2_rebuilt", "'II,V2'")}
IGNORE = {"model_name", "save_path", "root_path", "init_method", "dist_backend"}  # names and paths: printed, not compared
SCRIPTS = {"ii": "unified_ii_only.sh", "v2": "unified_v2_only.sh", "ii_v2": "unified_ii_v2.sh"}
DEFAULTS = {"diagnostic_class": "all", "pretrained_model_path": "None"}  # patchECG_pretrain.py default; the _only scripts do not set it


def plain(v):
    """Compare values as text: 'adamw', <DataAugmentation.none: 0>, ['None'] and none all reduce to their name."""
    if v is None:
        return None
    v = str(v).strip().strip("'\"")
    m = re.match(r"<\w+\.(\w+):", v)
    if m:
        v = m.group(1)
    v = v.strip("[]'\" ")
    try:
        return repr(float(v))
    except ValueError:
        return v.lower()


def parser_reference(tag):
    """The full _only recipe, defaults included: the real argument parser of src/patchECG_pretrain.py run on the _only script's own flags.
    Run this check with the repo's python. Returns None if the script or the parser is not available, and the check then falls back to the script's flags."""
    p = Path("src/scripts/multi-gpu/pretrain/unified") / SCRIPTS[tag]
    if not p.exists():
        return None
    body = p.read_text().split("src.patchECG_pretrain")[-1].replace("\\\n", " ").split("\n")[0]
    words = [w for w in shlex.split(body) if w]
    words = ["--root_path=/tmp/none" if w.startswith("--root_path") else w for w in words]
    code = "import os, sys; sys.argv = ['x'] + %r; os.environ.setdefault('LOCAL_RANK', '0'); import src.patchECG_pretrain as P; print('PARAMS ' + repr(P.params))" % words
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    line = next((x for x in r.stdout.splitlines() if x.startswith("PARAMS ")), None)
    if line is None:
        print(f"  the real parser could not be run here ({(r.stderr.strip().splitlines() or ['no output'])[-1][:200]}); falling back to the script's flags")
        return None
    return dict(re.findall(r"(\w+)=('[^']*'|\[[^\]]*\]|<[^>]*>|[^,)]+)", line[7:]))


def script_flags(tag):
    """Flags of the _only script, or None if the script is not here."""
    p = Path("src/scripts/multi-gpu/pretrain/unified") / SCRIPTS[tag]
    if not p.exists():
        return None
    body = p.read_text().split("src.patchECG_pretrain")[-1].replace("\\\n", " ")
    words, out = shlex.split(body.split("\n")[0], posix=True), {}
    i = 0
    while i < len(words):
        w = words[i]
        i += 1
        if not w.startswith("--"):
            continue
        k, eq, v = w[2:].partition("=")
        if eq:
            out[k] = v
        elif i < len(words) and not words[i].startswith("--"):
            out[k] = words[i]
            i += 1
        elif k.startswith("no-"):
            out[k[3:]] = "False"
        else:
            out[k] = "True"
    return out


def parse(p):
    t = p.read_text()
    return dict(re.findall(r"(\w+)=('[^']*'|\[[^\]]*\]|<[^>]*>|[^,)]+)", t))


for tag, (old, new, lead) in PAIRS.items():
    print(f"\n=== {new}")
    q = ROOT / new / "params.txt"
    if not q.exists():
        print(f"  NOT FOUND: {q} (still running, or a different location)")
        continue
    b = parse(q)
    full = parser_reference(tag)
    flags = script_flags(tag)
    if full:
        source = f"the real parser on {SCRIPTS[tag]}, defaults included"
        want = {k: v for k, v in full.items() if k not in IGNORE}
    else:
        source = f"{SCRIPTS[tag]}, read from the script" if flags else "the table typed in this file (script not found)"
        want = {**DEFAULTS, **{k: v for k, v in (flags or {**RECIPE, "custom_lead_selection": lead}).items() if k not in IGNORE and k not in ("num_workers", "save_every", "dset_pretrain")}}
    bad = [k for k, v in want.items() if plain(b.get(k)) != plain(v)]
    for k in bad:
        print(f"  DIFFERS from the _only recipe: {k} = {b.get(k)} (recipe: {want[k]})")
    print(f"  checked {len(want)} recipe values against {source}: {len(bad)} differ")
    print("  names and paths, not compared: " + ", ".join(f"{k}={b.get(k)}" for k in sorted(IGNORE)))
    compared = set(want) | IGNORE
    o = ROOT / old / "params.txt"
    if o.exists():
        a = parse(o)
        diff = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k) and k not in IGNORE]
        print(f"  against {old}/params.txt: {len(diff)} differ besides names and paths")
        for k in diff:
            print(f"    {k}: _only {a.get(k)} | _rebuilt {b.get(k)}")
        compared |= set(a)
    else:
        print(f"  {o} is not on this cluster (the _only backbones were trained on Narval)")
    rest = sorted(set(b) - compared)
    print(f"  not compared, no reference for them ({len(rest)}): " + (", ".join(f"{k}={b[k]}" for k in rest) or "none"))
    for extra in ("data_source.txt", "pretrain_split.csv"):
        f = ROOT / new / extra
        if not f.exists():
            print(f"  missing: {extra}")
        elif extra.endswith(".csv"):
            print(f"  {extra}: {sum(1 for _ in open(f)) - 1} rows")
        else:
            print(f"  {extra}: {f.read_text().strip()[:400]}")
print("\n=== Model code")
print(subprocess.run("grep -n 'LayerNorm(patch_size)\\|nn.LayerNorm(dim),' src/core/models/vit_1d.py; git log -1 --format='%h %ad %s' --date=short -- src/core/models/vit_1d.py", shell=True, capture_output=True, text=True).stdout)
