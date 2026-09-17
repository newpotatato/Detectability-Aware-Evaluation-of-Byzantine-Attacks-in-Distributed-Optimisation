"""
Vstavka perezapushchennyh yacheek Krum (f=4) v osnovnye fayly rezultatov.

Zamenyayutsya TOLKO stroki agg == "krum": ostalnye pyat pravil parametra f ne
imeyut, i ih progony ne izmenilis. Ishodnye fayly lezhat v
results/_before_krum_f4/ .
"""
from __future__ import annotations
import os
import pandas as pd

JOBS = [
    ("results/m4_main.csv",       "results/m4_main_krum_f4.csv"),
    ("results/baseline_syn.csv",  "results/baseline_krum_f4.csv"),
    ("results/mnist.csv",         "results/mnist_krum_f4.csv"),
    ("results/baseline_mnist.csv", "results/baseline_krum_f4_mnist.csv"),
    ("results/m4_adapt.csv",      "results/m4_adapt_krum_f4.csv"),
] + [(f"results/sweep_m{m}.csv", f"results/sweep_m{m}_krum_f4.csv")
     for m in [1, 2, 4, 6]] \
  + [(f"results/sweep_N{n}.csv", f"results/sweep_N{n}_krum_f4.csv")
     for n in [10, 50, 100]]


def merge(main_path, new_path):
    old = pd.read_csv(main_path)
    new = pd.read_csv(new_path)
    keep = old[old["agg"] != "krum"]
    # kolonki privodim k poryadku osnovnogo fayla
    new = new.reindex(columns=old.columns)
    out = pd.concat([keep, new], ignore_index=True)
    out.to_csv(main_path, index=False)
    print(f"  {os.path.basename(main_path):24s} "
          f"{len(old)} -> {len(out)} strok  "
          f"(krum: {len(old) - len(keep)} zameneno na {len(new)})")


if __name__ == "__main__":
    print("Vstavka Krum f=4:")
    for a, b in JOBS:
        if os.path.exists(b):
            merge(a, b)
        else:
            print(f"  PROPUSK: net {b}")
