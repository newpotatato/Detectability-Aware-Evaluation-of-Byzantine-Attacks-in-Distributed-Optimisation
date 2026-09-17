"""
EKSPERIMENT 4b -- ispravlennaya versiya glavnogo rezultata V1.

Otlichiya ot exp4:
  1. DVUHSTORONNEE uravnivanie byudzheta: tihie ataki podnimayutsya do tselevogo
     urovnya, a ne ostayutsya tihimi. Bez etogo ALIE (eps=-4.8) sravnivalas by
     na -4.8 protiv signflip na +0.3 -- eto ne ravnyy byudzhet.
  2. d_eff kalibruetsya i peredaetsya i v stoimost, i v radius shara.
  3. 20 sidov vmesto 8, bootstrap-interval na rangovuyu korrelyatsiyu.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr

from . import landscapes as LS, swarm as SW, attacks as AT, metrics as ME
from . import budget as BUD
from .exp0 import TUNED
from .exp4 import ATTACKS, AGGS, BUDGETS, NATIVE

SEEDS = list(range(20))


def harm_eps(L, agg, cfg, honest, mk):
    pairs, epss = [], []
    for s in SEEDS:
        tr = SW.run(L, agg, cfg, s, attack=mk())
        pairs.append((honest[s].f_consensus, tr.f_consensus))
        epss.append(np.nanmean(tr.eps_spent))
    return ME.summarize(pairs, f_opt=L.f_opt)["d_med"], float(np.nanmedian(epss))


if __name__ == "__main__":
    rows = []
    for lname in ["sphere", "rastrigin"]:
        L = LS.make(lname, d=10)
        print(f"\n{'='*74}\n=== {lname} ===", flush=True)
        for agg in AGGS:
            # --- kalibrovka d_eff po chestnym otpravkam, FIKSIRUEM na blok
            c0 = SW.Config(T=200, eps_budget=None, **TUNED[lname])
            tr0 = SW.run(L, agg, c0, 0, attack=None, collect_honest=True)
            d_eff = BUD.effective_d(tr0.honest_rounds[:120])

            # NATIVNYY progon: two_sided=False! Zdes nuzhno izmerit SOBSTVENNUYU
            # gromkost kazhdoy ataki, poetomu tolko potolok ot perepolneniya.
            # Pri two_sided=True vse ataki nasilno podtyagivayutsya k eps=NATIVE,
            # i baza sravneniya unichtozhaetsya (oshibka poymana 2026-08-28).
            cfgN = SW.Config(T=200, eps_budget=NATIVE, d_eff=d_eff,
                             two_sided=False, **TUNED[lname])
            honN = {s: SW.run(L, agg, cfgN, s, attack=None) for s in SEEDS}
            nat = {a: harm_eps(L, agg, cfgN, honN, mk) for a, mk in ATTACKS.items()}

            print(f"\n--- {agg} (d_eff={d_eff:.2f}) ---", flush=True)
            print(f"{'ataka':<11}{'svoya eps':>10}{'vred':>8}   "
                  + "".join(f"{'e='+str(b):>9}" for b in BUDGETS), flush=True)

            matched = {}
            for b in BUDGETS:
                cfg = SW.Config(T=200, eps_budget=b, d_eff=d_eff, two_sided=True,
                                **TUNED[lname])
                hb = {s: SW.run(L, agg, cfg, s, attack=None) for s in SEEDS}
                matched[b] = {a: harm_eps(L, agg, cfg, hb, mk)[0]
                              for a, mk in ATTACKS.items()}

            names = list(ATTACKS)
            nat_h = np.array([nat[a][0] for a in names])
            for i, a in enumerate(names):
                print(f"{a:<11}{nat[a][1]:>10.2f}{nat_h[i]:>8.3f}   "
                      + "".join(f"{matched[b][a]:>9.3f}" for b in BUDGETS), flush=True)

            for b in BUDGETS:
                mh = np.array([matched[b][a] for a in names])
                rows.append((lname, agg, b, spearmanr(nat_h, mh).statistic))

    print(f"\n\n{'='*74}\n=== SVODKA (dvuhstoronnee uravnivanie, {len(SEEDS)} sidov) ===")
    rng = np.random.default_rng(0)
    print(f"\n{'byudzhet':>9}{'mediana rho':>14}{'95% bootstrap':>22}")
    for b in BUDGETS:
        v = np.array([r for _, _, bb, r in rows if bb == b and np.isfinite(r)])
        bs = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{b:>9}{np.median(v):>14.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>22}")
    print("\nrho = 1 -> poryadok atak ne izmenilsya pri uravnivanii byudzheta")
