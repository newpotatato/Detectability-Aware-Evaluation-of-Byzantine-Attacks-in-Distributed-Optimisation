"""
EKSPERIMENT 0 -- BLOKIRUYUSHCHIY (v1/README §7).

Vopros: lomaetsya li obratnaya svyaz PERVOGO PORYADKA na agregatorah so
SPADAYUSHCHEY krivoy vliyaniya?

FedSA (2025) podbiraet amplitudu onlayn, otsenivaya proizvodnuyu otklika
agregatora konechnymi raznostyami. Proizvodnaya lokalna. Gipoteza: tam, gde
vliyanie obvalivaetsya v nol (Krum) libo razryvno (Talwar), gradienta net
vovse, i lokalnyy zakon zastrevaet, togda kak globalnyy poisk nahodit pik.

Esli gipoteza NE podtverzhdaetsya -- §5.2 raboty vybrasyvaetsya.

Merim: dolyu vreda ot NAILUCHSHEY FIKSIROVANNOY amplitudy, naydennoy
pereigryvaniem na tom zhe side (eto zhe znamenatel sozhaleniya).
"""
from __future__ import annotations
import numpy as np

from . import landscapes as LS, swarm as SW, attacks as AT, metrics as ME
from . import aggregators as AGG

TUNED = {                       # iz harness/tune_honest.py
    "sphere":    dict(lr=0.08,   eta=0.3),
    "ellipsoid": dict(lr=0.0003, eta=0.3),
    "rastrigin": dict(lr=0.01,   eta=0.3),
    "gallagher": dict(lr=0.08,   eta=0.6),
}
KAPPA_GRID = np.logspace(-1.5, 1.0, 12)   # 0.03 .. 10, v edinitsah sobstvennogo shaga
EPS_BUDGET = 1.0                          # bez byudzheta amplituda uhodit v beskonechnost


def _honest_cache(L, agg, cfg, seeds):
    return {s: SW.run(L, agg, cfg, s, attack=None) for s in seeds}


def harm_of(L, agg, cfg, seeds, make_attack, honest):
    """Mediana L po sidam dlya zadannoy ataki."""
    pairs = []
    for s in seeds:
        tr = SW.run(L, agg, cfg, s, attack=make_attack())
        pairs.append((honest[s].f_consensus, tr.f_consensus))
    return ME.summarize(pairs, f_opt=L.f_opt)


def run_landscape(lname, agg_names, seeds, T=200, verbose=True):
    L = LS.make(lname, d=10)
    cfg = SW.Config(T=T, eps_budget=EPS_BUDGET, **TUNED[lname])
    rows = []
    for agg in agg_names:
        honest = _honest_cache(L, agg, cfg, seeds)

        # --- potolok: nailuchshaya FIKSIROVANNAYA amplituda zadnim chislom
        best_L, best_k = -np.inf, None
        for k in KAPPA_GRID:
            r = harm_of(L, agg, cfg, seeds, lambda k=k: AT.FixedAmplitude(k), honest)
            if r["d_med"] > best_L:
                best_L, best_k = r["d_med"], k

        fo = harm_of(L, agg, cfg, seeds,
                     lambda: AT.FirstOrderAmplitude(k0=1.0), honest)
        gs = harm_of(L, agg, cfg, seeds,
                     lambda: AT.GlobalSearchAmplitude(grid=KAPPA_GRID), honest)

        denom = max(best_L, 1e-9)
        rows.append(dict(agg=agg, best_L=best_L, best_k=best_k,
                         fo=fo["d_med"], gs=gs["d_med"],
                         fo_frac=fo["d_med"] / denom, gs_frac=gs["d_med"] / denom,
                         redesc=AGG.AGGREGATORS[agg]["redescending"]))
        if verbose:
            r = rows[-1]
            print(f"  {agg:<11}{'SPAD' if r['redesc'] else '    ':>5}"
                  f"  potolok d={r['best_L']:.3f} pri k={r['best_k']:.2f}"
                  f"   pervyy_por={r['fo_frac']:>6.1%}   global={r['gs_frac']:>6.1%}")
    return rows


if __name__ == "__main__":
    seeds = list(range(10))
    aggs = ["mean", "median", "trimmed", "krum", "talwar", "tukey"]

    all_rows = []
    for lname in ["sphere", "rastrigin"]:
        print(f"\n=== {lname} ===")
        all_rows += run_landscape(lname, aggs, seeds)

    print("\n\n=== SVODKA: dolya vreda (metrika d) ot nailuchshey fiksirovannoy amplitudy ===")
    print(f"{'':<14}{'pervyy poryadok':>18}{'globalnyy poisk':>18}{'raznitsa':>11}")
    for grp, label in [(True, "SPADAYUSHCHIE"), (False, "prochie")]:
        sel = [r for r in all_rows if r["redesc"] == grp]
        if not sel:
            continue
        fo = np.median([r["fo_frac"] for r in sel])
        gs = np.median([r["gs_frac"] for r in sel])
        print(f"{label:<14}{fo:>18.1%}{gs:>18.1%}{gs - fo:>+11.1%}")

    sp = [r for r in all_rows if r["redesc"]]
    ot = [r for r in all_rows if not r["redesc"]]
    gap_sp = np.median([r["gs_frac"] - r["fo_frac"] for r in sp]) if sp else 0
    gap_ot = np.median([r["gs_frac"] - r["fo_frac"] for r in ot]) if ot else 0
    print(f"\nVERDIKT: preimushchestvo globalnogo poiska na SPADAYUSHCHIH = {gap_sp:+.1%}, "
          f"na prochih = {gap_ot:+.1%}")
    print("PODTVERZHDENO" if gap_sp > 0.15 and gap_sp > 2 * abs(gap_ot)
          else "NE PODTVERZHDENO -- §5.2 pod voprosom")
