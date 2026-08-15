"""
Modèles.

`ShortcutNet` = encodeur convolutif + tête de classification (sain/malade)
             + tête adversariale optionnelle (prédire le biais) branchée
               derrière une couche d'inversion de gradient (GRL).

L'encodeur expose `last_conv` : la couche cible de Grad-CAM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Gradient Reversal Layer (Ganin & Lempitsky, 2015)
# --------------------------------------------------------------------------- #
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lambd * grad_out, None


def grad_reverse(x, lambd: float = 1.0):
    """Identité à l'aller, gradient multiplié par -lambda au retour.

    Effet : l'adversaire apprend à deviner le biais à partir des features,
    tandis que l'encodeur est poussé à rendre cette tâche IMPOSSIBLE, donc à
    produire des représentations invariantes au confondant.
    """
    return _GradReverse.apply(x, lambd)


# --------------------------------------------------------------------------- #
# Petit ResNet maison (rapide sur CPU)
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = (
            nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
            if (stride != 1 or cin != cout)
            else nn.Identity()
        )

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h + self.short(x))


class SmallResNet(nn.Module):
    """~200k paramètres, 4 étages. Suffisant pour saturer la tâche synthétique."""

    def __init__(self, widths=(16, 32, 64, 96)):
        super().__init__()
        w = widths
        self.stem = nn.Sequential(nn.Conv2d(1, w[0], 5, 2, 2, bias=False),
                                  nn.BatchNorm2d(w[0]), nn.ReLU(inplace=True))
        self.layer1 = BasicBlock(w[0], w[0], 1)
        self.layer2 = BasicBlock(w[0], w[1], 2)
        self.layer3 = BasicBlock(w[1], w[2], 2)
        self.layer4 = BasicBlock(w[2], w[3], 2)
        self.out_dim = w[3]

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


def _resnet18_gray():
    """ResNet-18 torchvision, adapté au niveau de gris, sans poids pré-entraînés."""
    from torchvision.models import resnet18

    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    backbone = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                             m.layer1, m.layer2, m.layer3, m.layer4)
    backbone.out_dim = 512
    return backbone


# --------------------------------------------------------------------------- #
# Réseau complet
# --------------------------------------------------------------------------- #
class ShortcutNet(nn.Module):
    def __init__(self, arch: str = "smallresnet", n_classes: int = 2, adversarial: bool = False):
        super().__init__()
        self.encoder = SmallResNet() if arch == "smallresnet" else _resnet18_gray()
        d = self.encoder.out_dim
        self.classifier = nn.Linear(d, n_classes)
        self.adversarial = adversarial
        self.adv_head = (
            nn.Sequential(nn.Linear(d, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))
            if adversarial else None
        )

    @property
    def last_conv(self) -> nn.Module:
        """Couche cible pour Grad-CAM (dernier bloc convolutif)."""
        enc = self.encoder
        return enc.layer4 if hasattr(enc, "layer4") else enc[-1]

    def forward(self, x, lambd: float = 0.0, return_all: bool = False):
        fmap = self.encoder(x)                       # B x C x h x w
        feat = fmap.mean(dim=(2, 3))                 # global average pooling
        logits = self.classifier(feat)
        adv_logits = None
        if self.adv_head is not None:
            adv_logits = self.adv_head(grad_reverse(feat, lambd))
        if return_all:
            return logits, adv_logits, fmap
        return logits


def build_model(arch="smallresnet", adversarial=False):
    return ShortcutNet(arch=arch, adversarial=adversarial)
