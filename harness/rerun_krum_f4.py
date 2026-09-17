"""
Perezapusk YACHEEK KRUM s pravilnym f = m = 4.

Zachem. Do 2026-09-05 pole agg_kwargs nigde ne zapolnyalos, i Krum rabotal so
svoim umolchaniem f=2 pri fakticheskih chetyreh vreditelyah. Krum otbiraet
vektor po summe rasstoyaniy do n-f-2 blizhayshih sosedey, poetomu zanizhennoe f
menyaet razmer okrestnosti (16 sosedey vmesto 14) i dopuskaet vizantiytsev v
otbor. Ostalnye pyat pravil parametra f ne imeyut, i ih progony ne zatronuty.

Perezapuskayutsya TRI veshchi, vse tolko dlya Krum:
  1. sinteticheskaya setka   -> results/m4_main_krum_f4.csv
  2. poly metriki            -> results/baseline_krum_f4.csv (sint.) i _mnist
  3. setka MNIST             -> results/mnist_krum_f4.csv

Chestnyy progon tozhe pereschityvaetsya: pri drugom f eto DRUGAYA sistema, i
kontrfakt obyazan byt ot toy zhe sistemy.

Sborka rezultatov -- harness/merge_krum_f4.py.
"""
from __future__ import annotations
import csv, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from harness.runner import run_grid, TUNED, get_landscape, agg_kwargs_for
from harness.runner_fed import run_fed_grid
from harness import swarm as SW, metrics as ME, fedsgd as FS

ATKS = ["signflip", "gaussian", "byzfl_1.5", "alie", "ipm", "minmax", "minsum"]
BUDG = ["native", -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 3.0, 10.0]
LAND = ["sphere", "ellipsoid", "rastrigin", "gallagher"]


def _syn_base(job):
    l, i, s, T, n_mal = job
    L = get_landscape(l, 10, i)
    cfg = SW.Config(T=T, eps_budget=None, agg_kwargs=agg_kwargs_for(n_mal),
                    **TUNED[l])
    tr = SW.run(L, "krum", cfg, s, attack=None)
    return dict(landscape=l, instance=i, agg="krum", seed=s,
                L_baseline=ME.L_baseline(tr.f_consensus))


def _fed_base(job):
    s, T, n_mal = job
    tr = FS.run_fed("krum", FS.FedConfig(T=T, n_malicious=0, eps_budget=None,
                                         agg_kwargs=agg_kwargs_for(n_mal)),
                    s, attack=None)
    return dict(agg="krum", seed=s, L_baseline=ME.L_baseline(tr.f_test))


def dump(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"  -> {path}  ({len(rows)} strok)", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    W = max(1, (os.cpu_count() or 4) - 1)

    print("### 1/4  poly metriki, sintetika (Krum, f=4)", flush=True)
    jobs = [(l, i, s, 200, 4) for l in LAND for i in [0, 1] for s in range(16)]
    with ProcessPoolExecutor(W) as ex:
        rows = list(ex.map(_syn_base, jobs, chunksize=4))
    dump(rows, "results/baseline_krum_f4.csv")
    v = np.array([r["L_baseline"] for r in rows])
    print(f"   pol L: {v.mean():.3f} +- {v.std():.3f}", flush=True)

    print("\n### 2/4  sinteticheskaya setka (Krum, f=4)", flush=True)
    run_grid("results/m4_main_krum_f4.csv",
             landscapes=LAND, instances=[0, 1], aggs=["krum"],
             attacks=ATKS, budgets=BUDG, seeds=list(range(16)),
             T=200, n_agents=20, n_mal=4, tag="m4_main")

    print("\n### 3/4  poly metriki, MNIST (Krum, f=4)", flush=True)
    jobs2 = [(s, 150, 4) for s in range(8)]
    with ProcessPoolExecutor(6) as ex:
        rows2 = list(ex.map(_fed_base, jobs2, chunksize=1))
    dump(rows2, "results/baseline_krum_f4_mnist.csv")
    v2 = np.array([r["L_baseline"] for r in rows2])
    print(f"   pol L: {v2.mean():.3f} +- {v2.std():.3f}", flush=True)

    print("\n### 4/4  setka MNIST (Krum, f=4)", flush=True)
    run_fed_grid("results/mnist_krum_f4.csv",
                 aggs=["krum"], attacks=ATKS,
                 budgets=["native", -4000.0, -2000.0, 0.0, 2000.0, 4000.0],
                 seeds=list(range(8)), T=150, n_clients=20, n_mal=4, workers=10)

    print(f"\n### GOTOVO za {time.time()-t0:.0f} s", flush=True)
