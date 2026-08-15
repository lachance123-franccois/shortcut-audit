from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(model, dataset, device="cpu", batch_size=128):
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    P, Y, B = [], [], []
    for x, y, b in loader:
        p = F.softmax(model(x.to(device)), dim=1)[:, 1]
        P.append(p.cpu().numpy())
        Y.append(np.asarray(y))
        B.append(np.asarray(b))
    return np.concatenate(P), np.concatenate(Y), np.concatenate(B)


def _acc(p, y):
    return float(((p > 0.5).astype(int) == y).mean())


def group_accuracies(p, y, b):
    out = {}
    for yy in (0, 1):
        for bb in (0, 1):
            m = (y == yy) & (b == bb)
            out[f"y{yy}_b{bb}"] = float(_acc(p[m], y[m])) if m.sum() else None
            out[f"n_y{yy}_b{bb}"] = int(m.sum())
    accs = [v for k, v in out.items() if k.startswith("y") and v is not None]
    out["worst_group"] = float(min(accs)) if accs else None
    return out


def evaluate_all(model, base_ds, labels, bias_iid, device="cpu"):
    zeros = np.zeros_like(bias_iid)
    ones = np.ones_like(bias_iid)
    suites = {
        "iid": bias_iid,
        "no_bias": zeros,
        "all_bias": ones,
        "inverted": 1 - bias_iid,
    }
    res = {}
    for name, flags in suites.items():
        p, y, b = predict(model, base_ds.with_bias(flags), device)
        res[name] = {"acc": _acc(p, y), **group_accuracies(p, y, b)}

    # effet causal du confondant : do(biais=1) vs do(biais=0), mêmes images
    p1, y, _ = predict(model, base_ds.with_bias(ones), device)
    p0, _, _ = predict(model, base_ds.with_bias(zeros), device)
    flip = ((p1 > 0.5) != (p0 > 0.5)).mean()
    res["causal"] = {
        # Effet moyen du biais sur P(malade) : ~0 si le modèle est causalement robuste.
        "ate_bias_on_p_malade": float((p1 - p0).mean()),
        "abs_ate": float(np.abs(p1 - p0).mean()),
        # Part des images dont le DIAGNOSTIC bascule juste en ajoutant/retirant l'artefact.
        "flip_rate": float(flip),
        # Écart entre la performance publiée et la performance sous confondant inversé.
        "shortcut_reliance": float(res["iid"]["acc"] - res["inverted"]["acc"]),
        "generalization_gap": float(res["iid"]["acc"] - res["no_bias"]["acc"]),
    }
    return res


def format_table(all_results: dict) -> str:
    rows = ["| Modèle | acc i.i.d. | acc sans biais | acc biais inversé | pire groupe (i.i.d.) "
            "| taux de bascule do(biais) | ATE biais |",
            "|---|---|---|---|---|---|---|"]
    for name, r in all_results.items():
        rows.append(
            f"| {name} | {r['iid']['acc']:.3f} | {r['no_bias']['acc']:.3f} | "
            f"{r['inverted']['acc']:.3f} | {r['iid']['worst_group']:.3f} | "
            f"{r['causal']['flip_rate']:.3f} | {r['causal']['ate_bias_on_p_malade']:+.3f} |"
        )
    return "\n".join(rows)
