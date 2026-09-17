"""
KONTROLNYY PROGON k eksperimentu 0 -- POVERKA PRIBORA.

Oboim metodam daetsya ISTINNOE vliyanie vmesto zashumlennogo proksi.
Vopros: vyigryvaet li globalnyy poisk hotya by pri IDEALNOM signale?

  vyigryvaet  -> gipoteza verna, no upiraetsya v KACHESTVO IZMERENIYA
  ne vyigryvaet -> gipoteza mertva, §5.2 vybrasyvaetsya okonchatelno

Kriteriy obyavlen DO progona: preimushchestvo na spadayushchih > +15%
i vdvoe bolshe, chem na prochih. Podkruchivanie posle -- nedopustimo.
"""
import numpy as np
from harness import landscapes as LS, swarm as SW, attacks as AT, metrics as ME
from harness import aggregators as AGG
from harness.exp0 import TUNED, KAPPA_GRID, EPS_BUDGET, harm_of, _honest_cache


def mk(cls, oracle, **kw):
    a = cls(**kw)
    a.oracle_influence = oracle
    return a


if __name__ == "__main__":
    seeds = list(range(10))
    aggs = ["mean", "median", "trimmed", "krum", "talwar", "tukey"]
    rows = []
    for lname in ["sphere", "rastrigin"]:
        L = LS.make(lname, d=10)
        cfg = SW.Config(T=200, eps_budget=EPS_BUDGET, **TUNED[lname])
        print(f"\n=== {lname} (ORAKULNYY signal vliyaniya) ===")
        for agg in aggs:
            honest = _honest_cache(L, agg, cfg, seeds)
            best = max(harm_of(L, agg, cfg, seeds,
                               lambda k=k: AT.FixedAmplitude(k), honest)["d_med"]
                       for k in KAPPA_GRID)
            fo = harm_of(L, agg, cfg, seeds,
                         lambda: mk(AT.FirstOrderAmplitude, True, k0=1.0), honest)["d_med"]
            gs = harm_of(L, agg, cfg, seeds,
                         lambda: mk(AT.GlobalSearchAmplitude, True, grid=KAPPA_GRID),
                         honest)["d_med"]
            den = max(best, 1e-9)
            red = AGG.AGGREGATORS[agg]["redescending"]
            rows.append((red, fo / den, gs / den))
            print(f"  {agg:<11}{'SPAD' if red else '    ':>5}  potolok d={best:.3f}"
                  f"   pervyy_por={fo/den:>6.1%}   global={gs/den:>6.1%}")

    sp = [(f, g) for r, f, g in rows if r]
    ot = [(f, g) for r, f, g in rows if not r]
    gap_sp = np.median([g - f for f, g in sp])
    gap_ot = np.median([g - f for f, g in ot])
    print(f"\n{'':<14}{'pervyy poryadok':>18}{'globalnyy poisk':>18}{'raznitsa':>11}")
    print(f"{'SPADAYUSHCHIE':<14}{np.median([f for f,_ in sp]):>18.1%}"
          f"{np.median([g for _,g in sp]):>18.1%}{gap_sp:>+11.1%}")
    print(f"{'prochie':<14}{np.median([f for f,_ in ot]):>18.1%}"
          f"{np.median([g for _,g in ot]):>18.1%}{gap_ot:>+11.1%}")
    ok = gap_sp > 0.15 and gap_sp > 2 * abs(gap_ot)
    print(f"\nVERDIKT PO ZARANEE OBYAVLENNOMU KRITERIYU: "
          f"{'PRIBOR ISPRAVEN, gipoteza zhivet' if ok else 'GIPOTEZA MERTVA -- §5.2 vybrasyvaetsya'}")
