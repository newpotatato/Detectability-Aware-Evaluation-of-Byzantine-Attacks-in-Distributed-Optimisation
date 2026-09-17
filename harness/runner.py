"""
Parallelnyy dvizhok progonov s sohraneniem SYRYH dannyh.

Zachem otdelnyy modul. Predydushchie skripty pechatali srazu agregirovannye
chisla, poetomu smena statistiki trebovala polnogo perepogona. Zdes kazhdaya
para (chestnyy, atakovannyy) sohranyaetsya POSIDNO v CSV, i lyubaya svodka --
rangi, eps50, doveritelnye intervaly -- schitaetsya potom bez povtornogo scheta.

Dve fazy:
  1. chestnye progony i kalibrovka d_eff -- po odnomu na (landshaft, ekz, agregator, sid)
  2. atakovannye progony -- perepolzuyut chestnye iz fazy 1

Chestnyy progon NE zavisit ot byudzheta (proektsiya primenyaetsya tolko k
vreditelyu), poetomu ego povtornyy schet na kazhdyy uroven byudzheta -- chistaya
poterya; v exp4b ona sostavlyala okolo 10%.
"""
from __future__ import annotations
import csv, itertools, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from . import landscapes as LS, swarm as SW, attacks as AT, metrics as ME
from . import budget as BUD

_LS_CACHE: dict = {}


def get_landscape(name, d, instance):
    """Landshafty kesheruyutsya: Gallagher pri postroenii utochnyaet optimum spuskom."""
    key = (name, d, instance)
    if key not in _LS_CACHE:
        _LS_CACHE[key] = LS.make(name, d=d, instance=instance)
    return _LS_CACHE[key]


ATTACK_FACTORY = {
    "signflip":   lambda: AT.SignFlip(1.0),
    "gaussian":   lambda: AT.GaussianNoise(1.0),
    "byzfl_1.5":  lambda: AT.FixedAmplitude(1.5),
    "alie":       lambda: AT.ALIE(),
    "ipm":        lambda: AT.IPM(1.0),
    "minmax":     lambda: AT.MinMax(),
    "minsum":     lambda: AT.MinSum(),
    "first_order": lambda: AT.FirstOrderAmplitude(k0=1.0),
    "global_search": lambda: AT.GlobalSearchAmplitude(grid=np.logspace(-1.5, 1.0, 12)),
    "random":     lambda: AT.RandomAmplitude(np.logspace(-1.5, 1.0, 12)),
}

TUNED = {
    "sphere":    dict(lr=0.08,   eta=0.3),
    "ellipsoid": dict(lr=0.0003, eta=0.3),
    "rastrigin": dict(lr=0.01,   eta=0.3),
    "gallagher": dict(lr=0.08,   eta=0.6),
}


def agg_kwargs_for(n_mal):
    """
    Parametry agregatora, zavisyashchie ot chisla vrediteley.

    ISPRAVLENO 2026-09-05. Ranshe agg_kwargs ostavalsya pustym, i Krum rabotal
    so svoim umolchaniem f=2 pri fakticheskih m=4. Krum otbiraet vektor po summe
    rasstoyaniy do n-f-2 blizhayshih sosedey, poetomu zanizhennoe f menyaet
    razmer okrestnosti (16 sosedey vmesto 14) i dopuskaet vizantiytsev v otbor.
    Teper f = m dlya vseh pravil, kotorye ego prinimayut; ostalnye poluchayut
    ego v **kw i ignoriruyut.
    """
    return {"f": int(n_mal)}


def _cfg(lname, T, n_agents, n_mal, eps, d_eff, two_sided, drift_lr=None,
         agg_f=None):
    """
    agg_f -- chislo vrediteley, o kotorom znaet AGREGATOR. Otdelno ot n_mal,
    potomu chto u CHESTNOGO progona n_mal = 0, no pravilo agregatsii dolzhno
    byt nastroeno TAK ZHE, kak v atakovannom: inache kontrfakt sravnivaet dve
    raznye sistemy, a ne odnu s vreditelem i bez nego.
    """
    kw = dict(TUNED[lname])
    if drift_lr is not None:
        kw["lr"] = drift_lr
    return SW.Config(T=T, n_agents=n_agents, n_malicious=n_mal, eps_budget=eps,
                     d_eff=d_eff, two_sided=two_sided,
                     agg_kwargs=agg_kwargs_for(n_mal if agg_f is None else agg_f),
                     **kw)


# ------------------------------------------------------------------ faza 1


def _honest_task(job):
    lname, inst, agg, seed, T, n_agents, d, n_mal = job
    L = get_landscape(lname, d, inst)
    cfg = _cfg(lname, T, n_agents, 0, None, None, True, agg_f=n_mal)
    tr = SW.run(L, agg, cfg, seed, attack=None, collect_honest=(seed == 0))
    d_eff = BUD.effective_d(tr.honest_rounds[:120]) if seed == 0 else None
    return (lname, inst, agg, seed), tr.f_consensus.astype(np.float32), d_eff


# ------------------------------------------------------------------ faza 2


def make_attack(name):
    """
    Podderzhivaet parametrizovannye imena vida "fixed@0.5" -- fiksirovannaya
    amplituda 0.5. Nuzhno dlya vychisleniya potolka (nailuchshaya fiksirovannaya
    amplituda zadnim chislom), otnositelno kotorogo schitaetsya sozhalenie.
    """
    if "@" in name:
        base, val = name.split("@", 1)
        if base == "fixed":
            return AT.FixedAmplitude(float(val))
        if base == "gauss":
            return AT.GaussianNoise(float(val))
        raise KeyError(name)
    return ATTACK_FACTORY[name]()


def _attack_task(job):
    (lname, inst, agg, atk, eps, seed, T, n_agents, n_mal, d,
     d_eff, two_sided, f_honest) = job
    L = get_landscape(lname, d, inst)
    cfg = _cfg(lname, T, n_agents, n_mal, eps, d_eff, two_sided)
    tr = SW.run(L, agg, cfg, seed, attack=make_attack(atk))
    fa = tr.f_consensus
    Lv = ME.L_single(f_honest, fa)
    dv = ME.d_single(f_honest, fa, L.f_opt)
    up = ME.uphill_single(f_honest, fa)
    return dict(
        landscape=lname, instance=inst, agg=agg, attack=atk,
        eps_target=("native" if two_sided is False else eps), seed=seed,
        n_agents=n_agents, n_mal=n_mal, T=T,
        f_honest=float(np.minimum.accumulate(f_honest)[-1]), f_attacked=float(fa[-1]),
        d=(np.nan if dv is None else dv),
        L=(np.nan if Lv is ME.HELP else float(Lv)),
        helped=int(Lv is ME.HELP),
        uphill=float(up) if np.isfinite(up) else np.nan,
        eps_spent=float(np.nanmean(tr.eps_spent)),
        clipped=float(tr.clipped.mean()),
    )


# ------------------------------------------------------------------ dvizhok


def run_grid(out_csv, landscapes, instances, aggs, attacks, budgets,
             seeds, T=200, n_agents=20, n_mal=1, d=10, workers=None,
             native_cap=50.0, tag=""):
    """
    budgets -- spisok chisel (uravnivanie) plyus stroka "native" dlya progona
    so SVOEY gromkostyu (odnostoronniy potolok native_cap).
    """
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    t0 = time.time()

    hjobs = [(l, i, a, s, T, n_agents, d, n_mal)
             for l in landscapes for i in instances for a in aggs for s in seeds]
    honest, d_eff = {}, {}
    with ProcessPoolExecutor(workers) as ex:
        for key, f, de in ex.map(_honest_task, hjobs, chunksize=4):
            honest[key] = f
            if de is not None:
                d_eff[key[:3]] = de
    print(f"[faza 1] chestnyh progonov: {len(honest)}  za {time.time()-t0:.0f} s", flush=True)

    ajobs = []
    for l, i, a, atk, b, s in itertools.product(landscapes, instances, aggs,
                                                attacks, budgets, seeds):
        de = d_eff[(l, i, a)]
        two = not (b == "native")
        eps = native_cap if b == "native" else float(b)
        ajobs.append((l, i, a, atk, eps, s, T, n_agents, n_mal, d, de, two,
                      honest[(l, i, a, s)]))

    print(f"[faza 2] atakovannyh progonov: {len(ajobs)} na {workers} yadrah", flush=True)
    first, n = True, 0
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = None
        with ProcessPoolExecutor(workers) as ex:
            for row in ex.map(_attack_task, ajobs, chunksize=8):
                if first:
                    w = csv.DictWriter(fh, fieldnames=list(row) + ["tag"])
                    w.writeheader(); first = False
                row["tag"] = tag
                w.writerow(row); n += 1
                if n % 2000 == 0:
                    print(f"   {n}/{len(ajobs)}  ({time.time()-t0:.0f} s)", flush=True)
    print(f"[gotovo] {n} strok -> {out_csv}   vsego {time.time()-t0:.0f} s", flush=True)
    return out_csv
