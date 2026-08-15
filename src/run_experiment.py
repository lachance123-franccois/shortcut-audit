from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from bias import bias_roi_mask
from cxr import assign_bias, load_image_folder, make_dataset
from data import BiasedDataset
from evaluate import evaluate_all, format_table
from figures import fig_dataset, fig_focus, fig_gradcam, fig_suites
from gradcam import attention_report
from models import build_model
from train import train_model


def parse_args():
    p = argparse.ArgumentParser(description="Robustesse causale / shortcut learning")
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--arch", choices=["smallresnet", "resnet18"], default="smallresnet")
    p.add_argument("--bias", choices=["watermark", "contrast", "both"], default="watermark")
    p.add_argument("--bias-strength", type=float, default=1.0)
    p.add_argument("--rho", type=float, default=0.95,
                   help="P(biais|malade) à l'entraînement du bras biaisé")
    p.add_argument("--lambda-adv", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="outputs")
    p.add_argument("--real-data", default=None,
                   help="dossier ImageFolder (2 classes) pour utiliser de vraies images")
    p.add_argument("--arms", default="all")
    return p.parse_args()


def main():
    a = parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    t0 = time.time()
    print(f"[setup] device={dev}  biais={a.bias}  rho={a.rho}  arch={a.arch}")


    if a.real_data:
        imgs, labels, masks = load_image_folder(a.real_data)
        n = len(labels)
        perm = np.random.default_rng(a.seed).permutation(n)
        cut = int(0.8 * n)
        tr, te = perm[:cut], perm[cut:]
        Xtr, Ytr = imgs[tr], labels[tr]
        Xte, Yte = imgs[te], labels[te]
        Mtr = Mte = None
    else:
        Xtr, Ytr, Mtr = make_dataset(a.n_train, seed=a.seed)
        Xte, Yte, Mte = make_dataset(a.n_test, seed=a.seed + 999)

    size = Xtr.shape[-1]
    roi = bias_roi_mask(size, a.bias)

    # attribution du confondant
    b_train_corr = assign_bias(Ytr, rho=a.rho, seed=a.seed + 1)       # confondu
    b_train_rand = assign_bias(Ytr, rho=0.5, seed=a.seed + 2)         # décorrélé
    b_train_none = np.zeros_like(Ytr)
    b_test_corr = assign_bias(Yte, rho=a.rho, seed=a.seed + 3)

    kw = dict(kind=a.bias, strength=a.bias_strength)
    ds = {
        "controle_sans_biais": BiasedDataset(Xtr, Ytr, b_train_none, **kw),
        "baseline_biaise": BiasedDataset(Xtr, Ytr, b_train_corr, **kw),
        "mitig_decorrele": BiasedDataset(Xtr, Ytr, b_train_rand, **kw),
        "mitig_adversarial": BiasedDataset(Xtr, Ytr, b_train_corr, **kw),
    }
    test_base = BiasedDataset(Xte, Yte, b_test_corr, masks=Mte, **kw)

    fig_dataset(Xtr, Ytr, b_train_corr, out / "fig1_dataset.png", kind=a.bias)

    arms = list(ds) if a.arms == "all" else a.arms.split(",")

    # entraîNement et évaluation 
    results, reports, histories = {}, {}, {}
    cam_idx = list(range(6))
    for name in arms:
        print(f"\n=== {name} ===")
        adv = name == "mitig_adversarial"
        model = build_model(arch=a.arch, adversarial=adv)
        histories[name] = train_model(
            model, ds[name], epochs=a.epochs, lr=a.lr, batch_size=a.batch_size,
            device=dev, lambda_adv=a.lambda_adv if adv else 0.0, seed=a.seed)

        results[name] = evaluate_all(model, test_base, Yte, b_test_corr, device=dev)
        c = results[name]["causal"]
        print(f"  acc i.i.d.={results[name]['iid']['acc']:.3f}  "
              f"sans biais={results[name]['no_bias']['acc']:.3f}  "
              f"biais inversé={results[name]['inverted']['acc']:.3f}  "
              f"bascule do(biais)={c['flip_rate']:.3f}")

        # Grad-CAM sur des images biaisées (c'est là que le raccourci est visible)
        ds_cam = test_base.with_bias(np.ones_like(b_test_corr))
        ds_cam.masks = Mte
        reports[name] = attention_report(model, ds_cam, cam_idx, roi, Mte, device=dev)
        if reports[name]["focus_bias"] is not None:
            print(f"  attention zone-filigrane x{reports[name]['focus_bias']:.2f} | "
                  f"poumons x{reports[name]['focus_lungs']:.2f}")
        torch.save(model.state_dict(), out / f"model_{name}.pt")

    
    fig_suites(results, out / "fig2_robustesse.png")
    fig_gradcam(reports, out / "fig3_gradcam.png")
    if any(r["focus_bias"] is not None for r in reports.values()):
        fig_focus(reports, out / "fig4_attention.png")

    payload = {
        "config": vars(a) | {"device": dev},
        "results": results,
        "attention": {k: {"focus_bias": v["focus_bias"], "focus_lungs": v["focus_lungs"]}
                      for k, v in reports.items()},
        "history": histories,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    table = format_table(results)
    (out / "results_table.md").write_text(table + "\n\n" + verdict(results))
    print("\n" + table + "\n\n" + verdict(results))
    print(f"\n[done] {time.time()-t0:.1f}s — sorties dans {out.resolve()}")


def verdict(res: dict) -> str:
    lines = ["### Lecture"]
    for name, r in res.items():
        sr = r["causal"]["shortcut_reliance"]
        fr = r["causal"]["flip_rate"]
        if sr > 0.20 or fr > 0.20:
            v = ("**RACCOURCI CONFIRMÉ** : la décision dépend massivement d'un artefact "
                 "sans contenu médical.")
        elif sr > 0.05 or fr > 0.05:
            v = "dépendance résiduelle au confondant."
        else:
            v = "robuste à l'intervention sur le confondant."
        lines.append(f"- `{name}` : chute i.i.d.→inversé = {sr:+.3f}, "
                     f"{fr*100:.1f} % des diagnostics basculent si l'on ajoute/retire "
                     f"l'artefact → {v}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
