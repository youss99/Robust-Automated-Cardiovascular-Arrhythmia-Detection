#!/usr/bin/env python3
"""Check (a): every number that stays is unchanged. Rows are matched in order by their text cells (names normalised); a new row's numbers must
be the old row's numbers with some removed (subsequence). Lists every old row that was dropped and every row whose numbers changed."""
import re, sys
from pathlib import Path
OLD = Path("/home/student/GIT/Tasks/final/runs/_archive_2026-11-24_before_cleanup/thesis_tables")
NEW = Path("/home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection/docs/thesis_tables")
REN = {"T5b_gold_standard_ceiling.tex": "T5b_reference_window_benchmark.tex"}
NUM = re.compile(r"[-+]?(?:\$-\$)?\d[\d{},.]*")
def rows(t):
    return [l.rstrip() for l in t.splitlines() if "&" in l and l.rstrip().endswith("\\\\") and not l.startswith("%")]
def key(r):
    cells = [c.strip() for c in r.rstrip("\\ ").split("&")]
    def numeric(c):
        c = c.replace("$\\pm$", "").replace("$-$", "-").replace("$^{*}$", "").replace("{,}", "")
        return re.fullmatch(r"[\s\-+\d.,\[\]()%]+|--|-|nan|n/a", c) is not None
    txt = [re.sub(r"\s+", " ", c) for c in cells if c and not numeric(c) and c not in ("--", "-", "n/a")]
    k = " | ".join(txt)
    for a, b in (("Gold-standard labels", "Reference window values"), ("vanilla\\_vit", "ViT encoder"), ("PRETRAIN", "linear reconstruction head"),
                 ("REGRESSION / CLS\\_TOKEN", "linear regression head on the CLS token"), ("RevIN, batch statistics", "per-lead standardisation with batch statistics")):
        k = k.replace(a, b)
    return k
def nums(r):
    return [n.replace("$-$", "-") for n in NUM.findall(r)]
def subseq(a, b):  # a is a subsequence of b
    it = iter(b)
    return all(x in it for x in a)
bad, dropped, total = [], [], 0
for f in sorted(OLD.glob("*.tex")):
    nf = NEW / REN.get(f.name, f.name)
    if not nf.exists():
        dropped.append((f.name, "WHOLE FILE")); continue
    o, n = rows(f.read_text()), rows(nf.read_text())
    j = 0
    for r in n:
        total += 1
        k = key(r)
        start = j
        while j < len(o) and key(o[j]) != k:
            j += 1
        if j == len(o):
            bad.append((nf.name, "no matching old row", r[:120])); j = start; continue
        dropped += [(f.name, o[x][:110]) for x in range(start, j)]
        if not subseq(nums(r), nums(o[j])):
            bad.append((nf.name, r[:140], "OLD: " + o[j][:140]))
        j += 1
    dropped += [(f.name, o[x][:110]) for x in range(j, len(o))]
print(f"{total} rows checked in {len(list(NEW.glob('*.tex')))} tables; rows whose numbers changed or could not be matched: {len(bad)}")
for b in bad: print("  BAD", b)
print(f"old rows dropped: {len(dropped)}")
for d in dropped: print("  DROPPED", d)
