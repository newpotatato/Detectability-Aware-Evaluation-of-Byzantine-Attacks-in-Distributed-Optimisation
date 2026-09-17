"""
Kesh MNIST dlya zadachi federativnogo obucheniya.

Skachivaet MNIST s OpenML i sohranyaet ego v results/ v tom vide, v kakom ego
zhdet fedsgd.load_mnist: X -- float32 (70000, 784) v [0, 1], y -- int64 (70000,).

    python -m harness.fetch_mnist

Trebuet scikit-learn; sam stend bez nego obhoditsya.
"""
from __future__ import annotations
import os
import numpy as np


def build(out_x="results/_mnist_X.npy", out_y="results/_mnist_y.npy"):
    from sklearn.datasets import fetch_openml

    d = fetch_openml("mnist_784", version=1, as_frame=False)
    X = (d.data / 255.0).astype(np.float32)
    y = d.target.astype(np.int64)
    os.makedirs(os.path.dirname(out_x) or ".", exist_ok=True)
    np.save(out_x, X)
    np.save(out_y, y)
    return X, y


if __name__ == "__main__":
    X, y = build()
    print(X.shape, X.dtype, X.min(), X.max(), y.shape, y.dtype)
