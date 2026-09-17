"""
EKSPERIMENT 4 -- GLAVNYY REZULTAT V1.

Vopros: menyaetsya li PORYADOK izvestnyh atak, esli uravnyat ih po byudzhetu
nezametnosti?

  "kak v literature" -- kazhdaya ataka deystvuet so svoey sobstvennoy gromkostyu
  "pri ravnom eps"   -- vse prinuditelno obrezany do odnogo byudzheta

Esli poryadok menyaetsya -- opublikovannye sravneniya smeshany po sile.
Esli net -- konfaundera net, i eto tozhe otvet.

Pobochno: eps50 (METRICS §8a) -- naimenshiy byudzhet, pri kotorom ataka
otnimaet zadannuyu dolyu vreda. Chitaetsya i kak effektivnost ataki,
i kak stoykost agregatora.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr

from . import landscapes as LS, swarm as SW, attacks as AT, metrics as ME
from .exp0 import TUNED, _honest_cache, harm_of

BUDGETS = [0.3, 1.0, 3.0, 10.0]
NATIVE = 50.0                       # praktichesky bez ogranicheniya, no bez perepolneniya
SEEDS = list(range(8))
AGGS = ["mean", "median", "trimmed", "krum", "talwar", "tukey"]
ATTACKS = {
    "signflip":  lambda: AT.SignFlip(1.0),
    "gaussian":  lambda: AT.GaussianNoise(1.0),
    "byzfl_1.5": lambda: AT.FixedAmplitude(1.5),      # znachenie po umolchaniyu ByzFL
    "alie":      lambda: AT.ALIE(),
    "ipm":       lambda: AT.IPM(1.0),
    "minmax":    lambda: AT.MinMax(),
    "minsum":    lambda: AT.MinSum(),
}


def harm_and_eps(L, agg, cfg, honest, mk):
    pairs, epss = [], []
    for s in SEEDS:
        tr = SW.run(L, agg, cfg, s, attack=mk())
        pairs.append((honest[s].f_consensus, tr.f_consensus))
        epss.append(np.nanmean(tr.eps_spent))
    r = ME.summarize(pairs, f_opt=L.f_opt)
    return r["d_med"], float(np.nanmedian(epss))


if __name__ == "__main__":
    flips, rhos = [], []
    for lname in ["sphere", "rastrigin"]:
        L = LS.make(lname, d=10)
        print(f"\n{'=' * 76}\n=== {lname} ===")
        for agg in AGGS:
            cfgN = SW.Config(T=200, eps_budget=NATIVE, **TUNED[lname])
            honest = _honest_cache(L, agg, cfgN, SEEDS)

            nat = {a: harm_and_eps(L, agg, cfgN, honest, mk) for a, mk in ATTACKS.items()}
            print(f"\n--- {agg} ---")
            print(f"{'ataka':<11}{'svoya eps':>10}{'vred':>8}{'rang':>6}   "
                  + "".join(f"{'e=' + str(b):>9}" for b in BUDGETS))

            matched = {}
            for b in BUDGETS:
                cfg = SW.Config(T=200, eps_budget=b, **TUNED[lname])
                hb = _honest_cache(L, agg, cfg, SEEDS)
                matched[b] = {a: harm_and_eps(L, agg, cfg, hb, mk)[0]
                              for a, mk in ATTACKS.items()}

            names = list(ATTACKS)
            nat_h = np.array([nat[a][0] for a in names])
            nat_rank = len(names) - np.argsort(np.argsort(nat_h))
            for i, a in enumerate(names):
                row = (f"{a:<11}{nat[a][1]:>10.2f}{nat_h[i]:>8.3f}{nat_rank[i]:>6}   ")
                row += "".join(f"{matched[b][a]:>9.3f}" for b in BUDGETS)
                print(row)

            for b in BUDGETS:
                mh = np.array([matched[b][a] for a in names])
                rho = spearmanr(nat_h, mh).statistic
                rhos.append((lname, agg, b, rho))
                m_rank = len(names) - np.argsort(np.argsort(mh))
                if nat_rank[int(np.argmax(nat_h))] != m_rank[int(np.argmax(mh))] or \
                   names[int(np.argmax(nat_h))] != names[int(np.argmax(mh))]:
                    flips.append((lname, agg, b,
                                  names[int(np.argmax(nat_h))], names[int(np.argmax(mh))]))

    print(f"\n\n{'=' * 76}\n=== SVODKA ===")
    print(f"\nRangovaya korrelyatsiya 'svoya gromkost' vs 'ravnyy byudzhet':")
    for b in BUDGETS:
        v = [r for l, a, bb, r in rhos if bb == b and np.isfinite(r)]
        print(f"  eps={b:<5}  mediana rho = {np.median(v):+.3f}"
              f"   (rho=1 -> poryadok ne izmenilsya)")
    print(f"\nSmena LIDERA (silneyshey ataki): {len(flips)} sluchaev iz {len(rhos)}")
    for l, a, b, n1, n2 in flips[:14]:
        print(f"  {l:<10}{a:<10}eps={b:<5} {n1} -> {n2}")
