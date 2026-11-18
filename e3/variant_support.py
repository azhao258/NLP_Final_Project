"""Explicit one-variable changes relative to the frozen v1 configuration."""
import copy
import torch
from transformers import Trainer


def variant_config(baseline, variant):
    cfg = copy.deepcopy(baseline)
    assert variant in ("v2", "v3")
    assert baseline["learning_rate"] == 5e-5
    if variant == "v2": cfg["learning_rate"] = 1e-4
    return cfg


def make_optimizer(model, cfg, variant):
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if variant == "v2":
        groups = [dict(params=[p for n,p in named], weight_decay=cfg["weight_decay"])]
        report = [dict(weight_decay=cfg["weight_decay"], names=[n for n,p in named])]
    else:
        # Exact grouping rule used by the installed pinned Transformers Trainer.
        decay = set(Trainer.get_decay_parameter_names(None, model))
        groups, report = [], []
        for apply in (True, False):
            selected = [(n,p) for n,p in named if (n in decay) == apply]
            wd = cfg["weight_decay"] if apply else 0.0
            groups.append(dict(params=[p for n,p in selected], weight_decay=wd))
            report.append(dict(weight_decay=wd, names=[n for n,p in selected]))
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(named)
    return torch.optim.AdamW(groups, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]), report
