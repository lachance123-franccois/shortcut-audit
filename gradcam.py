from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.layer = target_layer if target_layer is not None else model.last_conv
        self.acts = None
        self.grads = None
        self._h = [
            self.layer.register_forward_hook(self._fwd),
            self.layer.register_full_backward_hook(self._bwd),
        ]

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def remove(self):
        for h in self._h:
            h.remove()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        score.backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)      # poids = grad moyen
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))  # combinaison + ReLU
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        # normalisation min-max par image
        b = cam.shape[0]
        flat = cam.view(b, -1)
        mn = flat.min(1).values.view(b, 1, 1)
        mx = flat.max(1).values.view(b, 1, 1)
        return ((cam - mn) / (mx - mn + 1e-8)).detach().cpu().numpy()


def cam_mass_in_roi(cam: np.ndarray, roi: np.ndarray) -> float:

    total = cam.sum()
    if total <= 0:
        return 0.0
    share = cam[roi].sum() / total
    area = roi.mean()
    return float(share / area)


def attention_report(model, dataset, indices, roi_bias, lung_masks=None, device="cpu"):
    cam_engine = GradCAM(model)
    r_bias, r_lung, cams, imgs = [], [], [], []
    for i in indices:
        x, y, b = dataset[i]
        x = x[None].to(device)
        cam = cam_engine(x)[0]
        cams.append(cam)
        imgs.append(x[0, 0].cpu().numpy())
        if roi_bias is not None:
            r_bias.append(cam_mass_in_roi(cam, roi_bias))
        if lung_masks is not None:
            r_lung.append(cam_mass_in_roi(cam, lung_masks[i]))
    cam_engine.remove()
    return {
        "focus_bias": float(np.mean(r_bias)) if r_bias else None,
        "focus_lungs": float(np.mean(r_lung)) if r_lung else None,
        "cams": np.stack(cams),
        "images": np.stack(imgs),
    }
