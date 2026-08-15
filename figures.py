"""Figures de l'expérience (matplotlib, sans dépendance à torch pour la première)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bias import apply_bias  # noqa: E402


def fig_dataset(images, labels, bias_flags, out: Path, kind="watermark", n=8):
    """Aperçu du dataset : contenu médical vs artefact injecté."""
    idx = list(np.where(labels == 0)[0][: n // 2]) + list(np.where(labels == 1)[0][: n // 2])
    fig, axes = plt.subplots(3, len(idx), figsize=(1.5 * len(idx), 5))
    for j, i in enumerate(idx):
        clean = images[i]
        biased = apply_bias(clean, kind=kind)
        axes[0, j].imshow(clean, cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(f"{'malade' if labels[i] else 'sain'}", fontsize=9)
        axes[1, j].imshow(biased, cmap="gray", vmin=0, vmax=1)
        axes[2, j].imshow(np.abs(biased - clean), cmap="magma")
        for r in range(3):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
    for r, lab in enumerate(["image propre", f"+ biais ({kind})", "différence"]):
        axes[r, 0].set_ylabel(lab, fontsize=9)
    fig.suptitle("Dataset : le signal causal (opacités) et le confondant injecté", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_suites(all_results: dict, out: Path):
    """Barres d'accuracy par suite contrefactuelle et par bras expérimental."""
    suites = ["iid", "no_bias", "all_bias", "inverted"]
    noms = ["i.i.d.\n(biais corrélé)", "biais retiré", "biais partout", "biais inversé"]
    models = list(all_results)
    x = np.arange(len(suites))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for k, m in enumerate(models):
        vals = [all_results[m][s]["acc"] for s in suites]
        bars = ax.bar(x + k * w - 0.4 + w / 2, vals, w, label=m)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=7)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.text(len(suites) - 0.5, 0.51, "hasard", fontsize=8, color="gray")
    ax.set_xticks(x, noms, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Robustesse causale : performance sous intervention sur le confondant")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_gradcam(reports: dict, out: Path, n=6):
    """Grille Grad-CAM : une ligne par bras expérimental."""
    models = list(reports)
    fig, axes = plt.subplots(len(models), n, figsize=(1.7 * n, 1.9 * len(models)),
                             squeeze=False)
    for r, m in enumerate(models):
        rep = reports[m]
        for c in range(n):
            ax = axes[r][c]
            ax.imshow(rep["images"][c], cmap="gray", vmin=0, vmax=1)
            ax.imshow(rep["cams"][c], cmap="jet", alpha=0.45)
            ax.set_xticks([])
            ax.set_yticks([])
        fb = rep["focus_bias"]
        axes[r][0].set_ylabel(f"{m}\nfocus biais x{fb:.1f}" if fb else m, fontsize=8)
    fig.suptitle("Grad-CAM : où le modèle regarde-t-il ? (images biaisées)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_focus(reports: dict, out: Path):
    """Attention normalisée sur la zone du filigrane vs sur les poumons."""
    models = list(reports)
    fb = [reports[m]["focus_bias"] or 0 for m in models]
    fl = [reports[m]["focus_lungs"] or 0 for m in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, fb, 0.4, label="zone du filigrane (non causale)")
    ax.bar(x + 0.2, fl, 0.4, label="poumons (causal)")
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.text(len(models) - 0.6, 1.05, "attention uniforme", fontsize=8, color="gray")
    ax.set_xticks(x, models, fontsize=9)
    ax.set_ylabel("densité d'attention Grad-CAM (x uniforme)")
    ax.set_title("Concentration de l'attention par région")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
