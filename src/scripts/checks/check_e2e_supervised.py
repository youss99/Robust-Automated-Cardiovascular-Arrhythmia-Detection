#!/usr/bin/env python3
"""GPU check for step3e (docs/DECISIONS.md, 2026-11-24): the end-to-end supervised jobs train every regression parameter from the first
epoch, and start from the same weights as their 20+70 twins.

Each case is a fresh process running the real trainer (`python -m src.patchECG_finetune` via runpy) with the exact flags of a generated
job: a step3e job (0 head-only + 90 full epochs) or its step3/step3o twin (20 + 70). Only --root_path (the VM's copy of the fold folder)
and --model_name (a git-ignored scratch run folder `_VMCHECK_e2e_<case>_<time>`) are replaced. Learner methods are wrapped to observe,
never to change anything, and the process stops after the first training epoch (or, for `init`, before training).
Recorded per case: the model state when Learner.fine_tune starts (= after seeding, data loading, model construction and optimizer setup),
the optimizer's parameter groups, every parameter's gradient at the first optimizer step, and the parameters after epoch 0.

Checks:
  1. Same seed, same initial weights: step3e job vs twin, bit for bit, for lead II, lead V2 and two-lead. Control: another seed differs.
  2. Optimizer: one group holding every model parameter, lr 1e-3, weight decay 0.05, lr_scale 1.
  3. First optimizer step of a step3e job: every parameter requires grad and has a finite, non-zero gradient, except mask_token
     (used only in pre-training; no gradient in any regression run).
  4. After epoch 0 of a step3e job: every parameter except mask_token has changed; mask_token is unchanged.
  Control for 3 and 4 (shows the check can fail): the twin's first (head-only) epoch must give gradients and changes in mlp_head.* only.
Usage, from the repo root: .venv/bin/python src/scripts/checks/check_e2e_supervised.py <generated step3e folder (FIN_OUT_E2E)> <out .md>"""
import json
import os
import re
import runpy
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FOLDS = REPO / "phase1_folds"
SAVE = REPO / "results/fine-tune/saved_models/patient-segment-regression"


class StopCheck(Exception):
    pass


def job_words(text):
    body = text.split("-m src.patchECG_finetune")[1].split("\n\n")[0].replace("\\\n", " ")
    return shlex.split(body.replace('"${STAGED_ROOT}"', "STAGED"))


def child(words, out_json, stop_at):
    """Runs the real trainer in this process with observing wrappers; writes out_json; never returns normally."""
    import torch
    sys.path.insert(0, str(REPO))
    from src.core.learner import Learner
    from src.core.utils.learner_utils import get_model_from_module

    rec = {}
    init = {}
    orig_fine_tune, orig_after_epoch = Learner.fine_tune, Learner.after_epoch

    def params(learn):
        return dict(get_model_from_module(learn.model).named_parameters())

    def fine_tune(self, n_epochs, freeze_epochs, lora=False):
        p = params(self)
        init.update({k: v.detach().cpu().clone() for k, v in get_model_from_module(self.model).state_dict().items()})
        torch.save(init, out_json.with_suffix(".init.pt"))
        groups = self.opt.param_groups
        ids = [id(q) for g in groups for q in g["params"]]
        rec["fine_tune_args"] = dict(n_epochs=n_epochs, freeze_epochs=freeze_epochs)
        rec["opt"] = dict(n_groups=len(groups), lr=[g["lr"] for g in groups], weight_decay=[g["weight_decay"] for g in groups],
                          lr_scale=[g.get("lr_scale") for g in groups], type=type(self.opt).__name__,
                          covers_all=sorted(ids) == sorted(id(q) for q in p.values()) and len(ids) == len(set(ids)),
                          n_params=len(p), n_values=int(sum(q.numel() for q in p.values())))
        if stop_at == "init":
            raise StopCheck
        real_step = self.opt.step

        def step(*a, **kw):
            if "first_step" not in rec:
                rec["first_step"] = {k: dict(requires_grad=q.requires_grad, grad=q.grad is not None,
                                             finite=bool(torch.isfinite(q.grad).all()) if q.grad is not None else None,
                                             norm=float(q.grad.norm()) if q.grad is not None else None,
                                             frac_nonzero=float((q.grad != 0).float().mean()) if q.grad is not None else None)
                                     for k, q in params(self).items()}
                rec["first_step_lr"] = [g["lr"] for g in self.opt.param_groups]
            return real_step(*a, **kw)

        self.opt.step = step
        return orig_fine_tune(self, n_epochs, freeze_epochs, lora)

    def after_epoch(self):
        orig_after_epoch(self)
        after = {k: v.detach().cpu() for k, v in params(self).items()}
        rec["after_epoch0"] = {k: dict(changed=bool((v != init[k]).any()), frac_changed=float((v != init[k]).float().mean()),
                                       max_abs=float((v - init[k]).abs().max())) for k, v in after.items()}
        rec["valid_loss_epoch0"] = float(self.training_tracker.recorder["valid_loss"][-1])
        raise StopCheck

    Learner.fine_tune, Learner.after_epoch = fine_tune, after_epoch
    sys.argv = ["src.patchECG_finetune"] + words
    try:
        runpy.run_module("src.patchECG_finetune", run_name="__main__")
    except StopCheck:
        pass
    out_json.write_text(json.dumps(rec, indent=1))
    sys.exit(0)


def case_words(sbatch, name, root, seed=None):
    w = job_words(sbatch.read_text())
    w = [f"--root_path={root}" if x.startswith("--root_path=") else f"--model_name={name}" if x.startswith("--model_name=") else x for x in w]
    return [f"--seed={seed}" if seed is not None and x.startswith("--seed=") else x for x in w]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(json.loads(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
    gen, out_md = Path(sys.argv[1]), Path(sys.argv[2])
    work = Path(tempfile.mkdtemp(prefix="e2e_check_", dir=out_md.parent))
    old = work / "old_jobs"
    subprocess.run([sys.executable, str(REPO / "src/scripts/custom_dataset/build_final_sbatches.py")], env=dict(os.environ, FIN_OUT=str(old)),
                   check=True, capture_output=True)
    stamp = time.strftime("%Y%m%d%H%M%S")
    cases = []   # (label, sbatch, stop, seed override)
    for pipe, twin, group, mode in (("L1supE2E", "L1sup", "step3", "lead_II"), ("L1supE2E", "L1sup", "step3", "lead_V2"),
                                    ("L2supE2E", "L2sup", "step3", "two_lead")):
        cases.append((f"{mode}_new", gen / "step3e" / f"FIN_{pipe}_{mode}_S1_fold01.sbatch", "epoch0", None))
        cases.append((f"{mode}_twin", old / group / f"FIN_{twin}_{mode}_S1_fold01.sbatch", "epoch0", None))
    cases.append(("lead_II_new_seed999", gen / "step3e" / "FIN_L1supE2E_lead_II_S1_fold01.sbatch", "init", 999))
    res, log = {}, []
    for label, sb, stop, seed in cases:
        data = re.search(r'\[\[ -f "([^"]+)/PatientSegmentRegressionDataset\.h5" \]\]', sb.read_text()).group(1)
        root = FOLDS / Path(data).name
        name = f"_VMCHECK_e2e_{label}_{stamp}"
        words = case_words(sb, name, root, seed)
        oj = work / f"{label}.json"
        t0 = time.time()
        r = subprocess.run([sys.executable, __file__, "--child", json.dumps(words), str(oj), stop], cwd=REPO, capture_output=True, text=True)
        (work / f"{label}.log").write_text(r.stdout + r.stderr)
        assert r.returncode == 0 and oj.exists(), f"{label} failed, see {work / (label + '.log')}:\n{r.stderr[-3000:]}"
        res[label] = json.loads(oj.read_text())
        log.append(f"{label}: {sb.name} ({' '.join(x for x in words if x.startswith(('--seed', '--n_epochs', '--custom_lead')))}), "
                   f"{time.time() - t0:.0f} s, run folder {name}")
        print(log[-1], flush=True)

    import torch
    lines = ["# step3e GPU check", "", f"{time.strftime('%Y-%m-%d %H:%M')}, VM ({torch.cuda.get_device_name(0)}), fold 1, S1 (seed 200) unless stated. "
             f"Work folder: `{work}`.", "", "Cases (each a fresh process running the real trainer with the generated job's flags):", ""]
    lines += [f"- {x}" for x in log]
    ok = {}

    def eq(a, b):
        A, B = torch.load(work / f"{a}.init.pt"), torch.load(work / f"{b}.init.pt")
        return A.keys() == B.keys() and all(torch.equal(A[k], B[k]) for k in A), len(A), int(sum(v.numel() for v in A.values()))

    lines += ["", "## 1. Same seed, same initial weights", "", "| mode | step3e job vs twin | tensors | values |", "|---|---|--:|--:|"]
    for mode in ("lead_II", "lead_V2", "two_lead"):
        same, n, v = eq(f"{mode}_new", f"{mode}_twin")
        ok[f"init {mode}"] = same
        lines.append(f"| {mode} | {'identical' if same else 'DIFFERENT'} | {n} | {v:,} |")
    ctrl, _, _ = eq("lead_II_new", "lead_II_new_seed999")
    ok["init control (seed 999 differs)"] = not ctrl
    lines += ["", f"Control, lead II seed 200 vs seed 999: {'identical (CHECK BROKEN)' if ctrl else 'different, as expected'}."]

    lines += ["", "## 2. Optimizer when Learner.fine_tune starts", "", "| case | groups | lr | weight decay | lr_scale | all parameters | type |",
              "|---|--:|---|---|---|---|---|"]
    for label, r in res.items():
        o = r["opt"]
        lines.append(f"| {label} | {o['n_groups']} | {o['lr']} | {o['weight_decay']} | {o['lr_scale']} | "
                     f"{'yes' if o['covers_all'] else 'NO'} ({o['n_params']} tensors, {o['n_values']:,} values) | {o['type']} |")
        ok[f"optimizer {label}"] = o["n_groups"] == 1 and o["lr"] == [1e-3] and o["weight_decay"] == [0.05] and o["lr_scale"] == [1] and o["covers_all"]

    def summarize(r, want):   # want(name) -> True if the parameter must receive a gradient and change
        fs, ae = r["first_step"], r["after_epoch0"]
        wrong = [k for k in fs if (fs[k]["grad"] and fs[k]["finite"] and fs[k]["norm"] > 0 and fs[k]["requires_grad"]) != want(k)]
        wrong += [k for k in ae if ae[k]["changed"] != want(k)]
        g = [fs[k] for k in fs if want(k)]
        c = [ae[k] for k in ae if want(k)]
        return sorted(set(wrong)), len(g), min(x["frac_nonzero"] for x in g), min(x["frac_changed"] for x in c)

    lines += ["", "## 3-4. Gradients at the first optimizer step and changes after epoch 0", "",
              "| case | epochs (head, full) | expected to train | parameters not as expected | trained tensors | min share of non-zero gradient values "
              "(first step) | min share of changed values (epoch 0) | mask_token: grad / changed | valid loss epoch 0 |", "|---|---|---|---|--:|--:|--:|---|--:|"]
    for label, r in res.items():
        if "first_step" not in r:
            continue
        twin = label.endswith("_twin")
        want = (lambda k: k.startswith("mlp_head.")) if twin else (lambda k: k != "mask_token")
        wrong, n, gmin, cmin = summarize(r, want)
        ok[f"grad/update {label}"] = not wrong
        fa = r["fine_tune_args"]
        mt_g, mt_c = r["first_step"]["mask_token"]["grad"], r["after_epoch0"]["mask_token"]["changed"]
        lines.append(f"| {label} | {fa['freeze_epochs']}, {fa['n_epochs']} | {'mlp_head.* only (control)' if twin else 'all but mask_token'} | "
                     f"{', '.join(wrong) or 'none'} | {n} | {gmin:.3f} | {cmin:.3f} | {mt_g} / {mt_c} | {r['valid_loss_epoch0']:.6f} |")
    new0 = res["lead_II_new"]["first_step"]
    lines += ["", "Per-parameter detail, lead II step3e job, first optimizer step (gradient norm, share of non-zero values):", ""]
    lines += [f"- `{k}`: " + (f"{v['norm']:.3e}, {v['frac_nonzero']:.3f}" if v["grad"] else "no gradient") for k, v in new0.items()]
    allok = all(ok.values())
    lines += ["", "## Result: " + ("ALL PASS" if allok else "FAILURES: " + ", ".join(k for k, v in ok.items() if not v)), "",
              "Checks: " + "; ".join(f"{k}: {'pass' if v else 'FAIL'}" for k, v in ok.items()) + "."]
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
