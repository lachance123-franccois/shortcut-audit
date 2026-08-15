from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from bias import apply_bias
from cxr import IMG_SIZE, assign_bias, load_image_folder, make_dataset  # noqa: F401


class BiasedDataset(Dataset):

    def __init__(self, images, labels, bias_flags, kind="watermark", strength=1.0,
                 masks=None, randomize_bias=False, seed=0):
        self.images = images
        self.labels = labels
        self.bias_flags = bias_flags
        self.kind = kind
        self.strength = strength
        self.masks = masks
        self.randomize_bias = randomize_bias  # augmentation : re-tire le biais à chaque epoch
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        b = int(self.bias_flags[i])
        if self.randomize_bias:
            b = int(self.rng.random() < 0.5)
        img = self.images[i]
        if b:
            img = apply_bias(img, kind=self.kind, strength=self.strength)
        x = torch.from_numpy(np.ascontiguousarray(img))[None]  # 1xHxW
        return x, int(self.labels[i]), b

    def with_bias(self, bias_flags):
        
        return BiasedDataset(self.images, self.labels, bias_flags, self.kind,
                             self.strength, self.masks)
