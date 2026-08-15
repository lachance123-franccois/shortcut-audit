

from __future__ import annotations

import numpy as np

# Région d'intérêt du filigrane, en coordonnées relatives (y0, y1, x0, x1).
# Sert aussi de ROI pour quantifier l'attention du modèle sur le raccourci.
WATERMARK_ROI = (0.02, 0.16, 0.02, 0.16)


def watermark_roi_px(size: int):
    y0, y1, x0, x1 = WATERMARK_ROI
    return int(y0 * size), int(y1 * size), int(x0 * size), int(x1 * size)


def _watermark(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    out = img.copy()
    s = img.shape[0]
    y0, y1, x0, x1 = watermark_roi_px(s)
    h, w = y1 - y0, x1 - x0
    glyph = np.zeros((h, w), np.float32)
    glyph[:, : max(1, w // 4)] = 1.0                       # verticale
    glyph[-max(1, h // 4):, :] = 1.0                       #  horizontale  -> "L"
    glyph[: max(1, h // 5), -max(1, w // 3):] = 0.7        # point d'appareil
    out[y0:y1, x0:x1] = np.clip(
        out[y0:y1, x0:x1] * (1 - 0.85 * glyph * strength) + 0.9 * glyph * strength, 0, 1
    )
    return out


def _contrast(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    gamma = 1.0 - 0.35 * strength
    out = np.clip(img, 0, 1) ** gamma
    out = np.clip(out * (1.0 + 0.06 * strength) - 0.02 * strength, 0, 1)
    return out.astype(np.float32)


def apply_bias(img: np.ndarray, kind: str = "watermark", strength: float = 1.0) -> np.ndarray:
    if kind == "watermark":
        return _watermark(img, strength)
    if kind == "contrast":
        return _contrast(img, strength)
    if kind == "both":
        return _watermark(_contrast(img, strength), strength)
    if kind == "none":
        return img
    raise ValueError(f"biais inconnu : {kind}")


def bias_roi_mask(size: int, kind: str) -> np.ndarray | None:
   
    if kind in ("watermark", "both"):
        m = np.zeros((size, size), bool)
        y0, y1, x0, x1 = watermark_roi_px(size)
        m[y0:y1, x0:x1] = True
        return m
    return None
