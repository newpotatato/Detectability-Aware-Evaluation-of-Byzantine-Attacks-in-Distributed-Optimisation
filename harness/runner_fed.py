"""Parallelnyy dvizhok dlya zadachi FL (MNIST + FedSGD). Struktura kak v runner.py."""
from __future__ import annotations
import csv, itertools, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from . import fedsgd as FS, metrics as ME
from .runner import make_attack, agg_kwargs_for


def _honest(job):
    agg, seed, T, n_cl, n_mal = job
    # agregator nastroen na to zhe f, chto i v atakovannyh progonah (runner._cfg)
    cfg = FS.FedConfig(T=T, n_clients=n_cl, n_malicious=0, eps_budget=None,
                       agg_kwargs=agg_kwargs_for(n_mal))
    tr = FS.run_fed(agg, cfg, seed, attack=None, collect_honest=(seed == 0))
    de = FS.diag_effective_d(tr.honest_rounds[:60]) if seed == 0 else None
    return (agg, seed), tr.f_test.astype(np.float32), tr.acc.astype(np.float32), de


def _attack(job):
    agg, atk, eps, seed, T, n_cl, n_mal, d_eff, two_sided, fh, ah = job
    cfg = FS.FedConfig(T=T, n_clients=n_cl, n_malicious=n_mal, eps_budget=eps,
                       d_eff=d_eff, two_sided=two_sided,
                       agg_kwargs=agg_kwargs_for(n_mal))
    tr = FS.run_fed(agg, cfg, seed, attack=make_attack(atk))
    Lv = ME.L_single(fh, tr.f_test)
    return dict(task="mnist", agg=agg, attack=atk,
                eps_target=("native" if two_sided is False else eps), seed=seed,
                n_agents=n_cl, n_mal=n_mal, T=T,
                f_honest=float(np.minimum.accumulate(fh)[-1]),
                f_attacked=float(np.minimum.accumulate(tr.f_test)[-1]),
                acc_honest=float(ah[-1]), acc_attacked=float(tr.acc[-1]),
                L=(np.nan if Lv is ME.HELP else float(Lv)),
                helped=int(Lv is ME.HELP),
                uphill=float(ME.uphill_single(fh, tr.f_test)),
                eps_spent=float(np.nanmean(tr.eps_spent)))


def run_fed_grid(out_csv, aggs, attacks, budgets, seeds, T=150,
                 n_clients=20, n_mal=4, workers=None, native_cap=1e9):
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    t0 = time.time()
    hj = [(a, s, T, n_clients, n_mal) for a in aggs for s in seeds]
    hon, acc, de = {}, {}, {}
    with ProcessPoolExecutor(workers) as ex:
        for key, f, ac, d in ex.map(_honest, hj, chunksize=1):
            hon[key] = f; acc[key] = ac
            if d is not None:
                de[key[0]] = d
    print(f"[faza 1] chestnyh: {len(hon)}  za {time.time()-t0:.0f} s   "
          f"d_eff: {({k: round(v) for k, v in de.items()})}", flush=True)

    aj = []
    for a, atk, b, s in itertools.product(aggs, attacks, budgets, seeds):
        two = not (b == "native")
        eps = native_cap if b == "native" else float(b)
        aj.append((a, atk, eps, s, T, n_clients, n_mal, de[a], two,
                   hon[(a, s)], acc[(a, s)]))
    print(f"[faza 2] atakovannyh: {len(aj)} na {workers} yadrah", flush=True)

    first, n = True, 0
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = None
        with ProcessPoolExecutor(workers) as ex:
            for row in ex.map(_attack, aj, chunksize=2):
                if first:
                    w = csv.DictWriter(fh, fieldnames=list(row)); w.writeheader(); first = False
                w.writerow(row); n += 1
                if n % 200 == 0:
                    print(f"   {n}/{len(aj)}  ({time.time()-t0:.0f} s)", flush=True)
    print(f"[gotovo] {n} strok -> {out_csv}   vsego {time.time()-t0:.0f} s", flush=True)
