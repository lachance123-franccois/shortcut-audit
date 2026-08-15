
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bias import apply_bias  # noqa: E402
from cxr import assign_bias, make_dataset  # noqa: E402


def build(images, flags, kind):
    X = np.stack([apply_bias(im, kind) if f else im for im, f in zip(images, flags)])
    return X.reshape(len(X), -1)


def run(kind="watermark", rho=0.95, n_train=2000, n_test=800, seed=0):
    Xtr, ytr, _ = make_dataset(n_train, seed=seed)
    Xte, yte, _ = make_dataset(n_test, seed=seed + 999)
    btr = assign_bias(ytr, rho, seed + 1)
    bte = assign_bias(yte, rho, seed + 3)

    suites = {"iid": bte, "no_bias": np.zeros_like(bte),
              "all_bias": np.ones_like(bte), "inverted": 1 - bte}

    print(f"\n=== biais='{kind}', rho={rho} ===")
    for arm, flags in [("controle_sans_biais", np.zeros_like(btr)),
                       ("baseline_biaise", btr),
                       ("mitig_decorrele", assign_bias(ytr, 0.5, seed + 2))]:
        clf = LogisticRegression(max_iter=400, C=0.05)
        clf.fit(build(Xtr, flags, kind), ytr)
        accs = {k: clf.score(build(Xte, f, kind), yte) for k, f in suites.items()}
        p1 = clf.predict(build(Xte, np.ones_like(bte), kind))
        p0 = clf.predict(build(Xte, np.zeros_like(bte), kind))
        flip = (p1 != p0).mean()
        print(f"{arm:22s} " + "  ".join(f"{k}={v:.3f}" for k, v in accs.items())
              + f"  bascule_do(biais)={flip:.3f}")


if __name__ == "__main__":
    run("watermark")
    run("contrast")
