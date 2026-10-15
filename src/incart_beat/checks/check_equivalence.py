"""Check that train.build_optimizer and the unfold patching in model.py match optim_factory / create_patch."""
import argparse
import sys

import torch

from src.incart_beat.model import BeatClassifier
from src.incart_beat.train import build_optimizer

print('timm/matplotlib imported by train+model:', 'timm' in sys.modules, 'matplotlib' in sys.modules, flush=True)
from src.core.models.optim_factory import prepare_optimizer  # noqa: E402
from src.core.support.patch_mask import create_patch  # noqa: E402

m = BeatClassifier('instance')
a = argparse.Namespace(layer_decay=0.65, weight_decay=0.05, lr=1e-3, n_layers=6, opt='adamw', opt_eps=1e-8,
                       opt_betas=None, sam=False)


def per_param(opt, scale_lr):
    return {id(p): (round(g['lr'] * (g['lr_scale'] if scale_lr else 1), 12), g['weight_decay'])
            for g in opt.param_groups for p in g['params']}


new, ref = per_param(build_optimizer(m, a), False), per_param(prepare_optimizer(m, a), True)
x = torch.randn(4, 1, 1000)
ok_opt, ok_patch = new == ref, torch.equal(x.unfold(-1, 50, 50), create_patch(x, 50, 50))
print(f'optimizer lr/wd identical for all {len(ref)} params: {ok_opt}')
print(f'unfold == create_patch: {ok_patch}')
sys.exit(0 if ok_opt and ok_patch else 1)
