"""
Nastroyka chestnogo progona pod kazhdyy landshaft.

Esli chestnyy roy sam ne shoditsya, vred NEIZMERIM: vse metriki opredeleny
otnositelno ego progressa (v1/README §11 punkt 3a). Zdes podbiraem lr i eta
tak, chtoby roy proyshel sushchestvennuyu dolyu puti do optimuma.

Kriteriy: otnositelnyy ostatochnyy zazor  (f(c_T) - f*) / (f(c_0) - f*).
Chem menshe, tem luchshe; tselevoy uroven -- ne huzhe 0.15 (na multimodalnyh roy zastrevaet, i eto normalno).
"""
import numpy as np
from harness import landscapes as LS, swarm as SW

LRS = [3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.08]
ETAS = [0.1, 0.3, 0.6]


def rel_gap(L, tr):
    g0 = tr.f_consensus[0] - L.f_opt
    gT = float(np.minimum.accumulate(tr.f_consensus)[-1]) - L.f_opt
    return gT / max(g0, 1e-12)


if __name__ == "__main__":
    best = {}
    for lname in ["sphere", "ellipsoid", "rastrigin", "gallagher"]:
        L = LS.make(lname, d=10)
        print(f"\n=== {lname} (f* = {L.f_opt:.4g}) ===")
        print(f"{'lr':>7}" + "".join(f"{'eta=' + str(e):>12}" for e in ETAS))
        rows = []
        for lr in LRS:
            row = f"{lr:>7}"
            for eta in ETAS:
                cfg = SW.Config(T=300, lr=lr, eta=eta, eps_budget=None)
                vals = [rel_gap(L, SW.run(L, "mean", cfg, s)) for s in range(3)]
                vals = [v for v in vals if np.isfinite(v)] or [np.inf]
                m = float(np.median(vals))
                rows.append((m, lr, eta))
                row += f"{m:>12.3g}"
            print(row)
        m, lr, eta = min(rows)
        best[lname] = dict(lr=lr, eta=eta, rel_gap=m)
        print(f"  -> luchshee: lr={lr}, eta={eta}, ostatochnyy zazor={m:.3g}"
              f"   {'OK' if m < 0.15 else 'PLOHO'}")

    print("\n\nITOG")
    print(f"{'landshaft':<12}{'lr':>8}{'eta':>7}{'ost.zazor':>12}{'godno':>8}")
    for k, v in best.items():
        print(f"{k:<12}{v['lr']:>8}{v['eta']:>7}{v['rel_gap']:>12.3g}"
              f"{'da' if v['rel_gap'] < 0.15 else 'NET':>8}")
