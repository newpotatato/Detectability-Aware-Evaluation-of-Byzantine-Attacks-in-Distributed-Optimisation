"""
Poly velichiny L: L_baseline = L(chestnyy, chestnyy) dlya kazhdoy konfiguratsii.

Bez nih absolyutnye znacheniya L i eps50 iz sohranennyh CSV NE pereschityvayutsya:
pol zavisit ot vsey chestnoy traektorii, a v CSV lezhit tolko ee finalnoe znachenie.
Rangi ot pola ne zavisyat (on obshchiy dlya vseh atak v yacheyke i sokrashchaetsya
v parnyh raznostyah), no lyuboe utverzhdenie ob URUVNE vreda bez nego neobosnovano.
"""
from __future__ import annotations
import csv, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from . import landscapes as LS, swarm as SW, metrics as ME, fedsgd as FS
from .runner import TUNED, get_landscape, agg_kwargs_for


def _syn(job):
    l, i, a, s, T, n_mal = job
    L = get_landscape(l, 10, i)
    # agregator nastraivaetsya na to zhe f, chto i v atakovannyh progonah:
    # pol metriki dolzhen izmeryatsya na TOY ZHE sisteme (sm. runner._cfg)
    cfg = SW.Config(T=T, eps_budget=None, agg_kwargs=agg_kwargs_for(n_mal),
                    **TUNED[l])
    tr = SW.run(L, a, cfg, s, attack=None)
    return dict(landscape=l, instance=i, agg=a, seed=s,
                L_baseline=ME.L_baseline(tr.f_consensus))


def _fed(job):
    a, s, T, n_mal = job
    tr = FS.run_fed(a, FS.FedConfig(T=T, n_malicious=0, eps_budget=None,
                                    agg_kwargs=agg_kwargs_for(n_mal)),
                    s, attack=None)
    return dict(agg=a, seed=s, L_baseline=ME.L_baseline(tr.f_test))


def dump(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"  -> {path}  ({len(rows)} strok)", flush=True)


if __name__ == "__main__":
    W = max(1, (os.cpu_count() or 4) - 1)
    AG = ["mean", "median", "trimmed", "krum", "talwar", "tukey"]

    jobs = [(l, i, a, s, 200, 4) for l in ["sphere", "ellipsoid", "rastrigin", "gallagher"]
            for i in [0, 1] for a in AG for s in range(16)]
    print(f"sinteticheskie: {len(jobs)} chestnyh progonov", flush=True)
    with ProcessPoolExecutor(W) as ex:
        rows = list(ex.map(_syn, jobs, chunksize=4))
    dump(rows, "results/baseline_syn.csv")
    v = np.array([r["L_baseline"] for r in rows])
    print(f"   pol L: mediana {np.median(v):.3f}, diapazon [{v.min():.3f}, {v.max():.3f}]")

    jobs2 = [(a, s, 150, 4) for a in AG for s in range(8)]
    print(f"\nMNIST: {len(jobs2)} chestnyh progonov", flush=True)
    with ProcessPoolExecutor(6) as ex:
        rows2 = list(ex.map(_fed, jobs2, chunksize=1))
    dump(rows2, "results/baseline_mnist.csv")
    v2 = np.array([r["L_baseline"] for r in rows2])
    print(f"   pol L: mediana {np.median(v2):.3f}, diapazon [{v2.min():.3f}, {v2.max():.3f}]")
