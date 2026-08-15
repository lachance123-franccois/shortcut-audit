from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(model, dataset, epochs=8, lr=1e-3, batch_size=64, device="cpu",
                lambda_adv=0.0, adv_ramp=True, seed=0, verbose=True):

    torch.manual_seed(seed)
    model.to(device).train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * max(1, len(loader)))
    ce = nn.CrossEntropyLoss()
    history = []

    for ep in range(epochs):
        p = (ep + 1) / epochs
        lambd = lambda_adv * (2.0 / (1.0 + np.exp(-6 * p)) - 1.0) if adv_ramp else lambda_adv
        tot = cor = 0
        loss_cls_sum = loss_adv_sum = 0.0
        adv_cor = 0

        for x, y, b in loader:
            x, y, b = x.to(device), y.to(device), b.to(device)
            logits, adv_logits, _ = model(x, lambd=lambd, return_all=True)
            loss = ce(logits, y)
            loss_cls_sum += loss.item() * len(y)
            if adv_logits is not None and lambda_adv > 0:
                l_adv = ce(adv_logits, b)
                loss = loss + l_adv
                loss_adv_sum += l_adv.item() * len(y)
                adv_cor += (adv_logits.argmax(1) == b).sum().item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            cor += (logits.argmax(1) == y).sum().item()
            tot += len(y)

        rec = {"epoch": ep + 1, "train_acc": cor / tot, "loss_cls": loss_cls_sum / tot,
               "lambda": float(lambd)}
        if lambda_adv > 0:
            rec["adv_acc"] = adv_cor / tot          # doit tendre vers ~0.5 (= biais effacé)
            rec["loss_adv"] = loss_adv_sum / tot
        history.append(rec)
        if verbose:
            extra = f" | adv_acc={rec['adv_acc']:.3f} (cible 0.50)" if lambda_adv > 0 else ""
            print(f"  epoch {ep+1:2d}/{epochs}  acc_train={rec['train_acc']:.3f}"
                  f"  loss={rec['loss_cls']:.4f}{extra}")
    return history
