"""
Génération des images.
"""

from __future__ import annotations

import numpy as np

IMG_SIZE = 96
 #Masque flou (0-1) d'une ellipse orientée.
def _ellipse(shape, cy, cx, ry, rx, rot=0.0):

    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    y = yy - cy
    x = xx - cx
    ca, sa = np.cos(rot), np.sin(rot)
    xr = x * ca + y * sa
    yr = -x * sa + y * ca
    d = (xr / rx) ** 2 + (yr / ry) ** 2
    return np.clip(1.5 - 1.5 * d, 0.0, 1.0)


def _blob(shape, cy, cx, sigma):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))


def make_cxr(rng: np.random.Generator, label: int, size: int = IMG_SIZE):
    s = size
    shape = (s, s)
    img = np.full(shape, 0.20, np.float32)

    # silhouette du thorax (variabile d'un patient a l'autre) 
    body_cy = s * (0.52 + rng.normal(0, 0.02))
    body_cx = s * (0.50 + rng.normal(0, 0.02))
    body = _ellipse(shape, body_cy, body_cx, s * 0.46, s * 0.40 * rng.uniform(0.92, 1.08))
    img += 0.45 * body

    #champs pulmonaires (zones sombres = air)
    lung_ry = s * 0.26 * rng.uniform(0.92, 1.08)
    lung_rx = s * 0.15 * rng.uniform(0.92, 1.08)
    lung_cy = body_cy - s * 0.04
    off = s * 0.19
    left = _ellipse(shape, lung_cy, body_cx - off, lung_ry, lung_rx, rot=+0.10)
    right = _ellipse(shape, lung_cy, body_cx + off, lung_ry, lung_rx, rot=-0.10)
    lungs = np.maximum(left, right)
    img -= 0.42 * lungs

    # médiastin / rachis 
    spine = _ellipse(shape, body_cy, body_cx, s * 0.42, s * 0.045)
    img += 0.22 * spine

    # côtes (texture périodique)
    yy = np.mgrid[0:s, 0:s][0].astype(np.float32)
    ribs = 0.05 * np.sin(yy / (s / 14.0) * np.pi + rng.uniform(0, np.pi))
    img += ribs * body

    # coupoles diaphragmatiques
    diaph = _ellipse(shape, body_cy + s * 0.30, body_cx, s * 0.16, s * 0.34)
    img += 0.18 * diaph

    lung_mask = lungs > 0.35

    # SIGNAL CAUSAL : opacités pulmonaires (classe 1 uniquement) 
    if label == 1:
        ys, xs = np.nonzero(lung_mask)
        n_op = rng.integers(1, 4)
        for _ in range(n_op):
            k = rng.integers(0, len(ys))
            sigma = rng.uniform(s * 0.035, s * 0.075)
            amp = rng.uniform(0.16, 0.30)
            img += amp * _blob(shape, ys[k], xs[k], sigma) * (0.35 + 0.65 * lungs)

    #nuisances présentes dans les DEUX classes (texture vasculaire)
    for _ in range(rng.integers(2, 5)):
        cy, cx = rng.uniform(0.2, 0.8, 2) * s
        img += rng.uniform(0.03, 0.07) * _blob(shape, cy, cx, rng.uniform(1.5, 3.0))

    #variabilité d'acquisition "normale" (dose, bruit, flou)
    img = np.clip(img, 0, 1) ** rng.uniform(0.85, 1.15)
    img = img * rng.uniform(0.92, 1.08)
    img += rng.normal(0, 0.02, shape).astype(np.float32)

    return np.clip(img, 0, 1).astype(np.float32), lung_mask


def make_dataset(n: int, seed: int = 0, size: int = IMG_SIZE, p_pos: float = 0.5):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < p_pos).astype(np.int64)
    imgs = np.zeros((n, size, size), np.float32)
    masks = np.zeros((n, size, size), bool)
    for i, y in enumerate(labels):
        imgs[i], masks[i] = make_cxr(rng, int(y), size)
    return imgs, labels, masks


def assign_bias(labels: np.ndarray, phi: float, seed: int = 0) -> np.ndarray:
    """
    Attribue le biais artificiel avec une corrélation contrôlée à la classe.

    phi = P(biais=1 | y=1) = P(biais=0 | y=0).
      phi = 0.95 -> biais quasi confondu avec la maladie (scénario "hôpital B")
      phi = 0.50 -> biais indépendant de la classe (scénario décorrélé)
      phi = 0.05 -> biais anti-corrélé (scénario contrefactuel inversé)
    """
    rng = np.random.default_rng(seed)
    p = np.where(labels == 1, phi, 1.0 - phi)
    return (rng.random(len(labels)) < p).astype(np.int64)


# Dataset réel 
def load_image_folder(root: str, size: int = IMG_SIZE, max_per_class: int | None = None):

    from pathlib import Path

    from PIL import Image

    root = Path(root)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if len(classes) != 2:
        raise ValueError(f"2 sous-dossiers attendus, trouvé : {classes}")
    imgs, labels = [], []
    for y, c in enumerate(classes):
        files = sorted([p for p in (root / c).iterdir()
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}])
        if max_per_class:
            files = files[:max_per_class]
        for p in files:
            im = Image.open(p).convert("L").resize((size, size), Image.BILINEAR)
            imgs.append(np.asarray(im, np.float32) / 255.0)
            labels.append(y)
    print(f"[data] dataset réel : {len(imgs)} images, classes={classes}")
    return np.stack(imgs), np.array(labels, np.int64), None
