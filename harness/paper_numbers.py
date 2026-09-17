"""
Vse chisla, kotorye tsitiruet rukopis V1, iz odnogo mesta.

Zapuskaetsya posle lyubogo pereschota:  python -m harness.paper_numbers
Kazhdyy blok podpisan razdelom stati, chtoby sverka byla mehanicheskoy.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .analyze import add_harm, reversal_stats, null_calibration, eps50_table

SYN_KEYS = ["landscape", "instance", "agg", "seed"]
MN_KEYS = ["agg", "seed"]
CELL = ["landscape", "instance", "agg"]


def load_all():
    syn = add_harm(pd.read_csv("results/m4_main.csv"),
                   pd.read_csv("results/baseline_syn.csv"), SYN_KEYS)
    mn = add_harm(pd.read_csv("results/mnist.csv"),
                  pd.read_csv("results/baseline_mnist.csv"), MN_KEYS)
    for d in (syn, mn):
        d["eps_target"] = d["eps_target"].astype(str)
    return syn, mn


def head(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def main():
    syn, mn = load_all()
    bs = pd.read_csv("results/baseline_syn.csv")
    bm = pd.read_csv("results/baseline_mnist.csv")

    head("Harm Metrics -- pol velichiny L")
    print(f"  analitika: {bs.L_baseline.mean():.3f} +- {bs.L_baseline.std():.3f}"
          f"   (T=200, {len(bs)} progonov)")
    print(f"  MNIST    : {bm.L_baseline.mean():.3f} +- {bm.L_baseline.std():.3f}"
          f"   (T=150, {len(bm)} progonov)")
    print(f"  L_adj obnulen u {(syn.L_adj <= 0).mean():.1%} progonov na analitike, "
          f"{(mn.L_adj <= 0).mean():.1%} na MNIST")
    print(f"  iz nih helped: {syn.helped.mean():.1%} / {mn.helped.mean():.1%}")

    head("Tablitsa 3 -- sobstvennaya gromkost pri opublikovannyh nastroykah")
    de_s, de_m = 11.85, 13800.0
    ns = syn[syn.eps_target == "native"].groupby("attack")["eps_spent"].median()
    nm = mn[mn.eps_target == "native"].groupby("attack")["eps_spent"].median()
    t = pd.DataFrame({"eps_syn": ns, "kappa_syn": np.sqrt(1 + 2 * ns / de_s),
                      "eps_mn": nm, "kappa_mn": np.sqrt(1 + 2 * nm / de_m)})
    print(t.sort_values("eps_syn", ascending=False).round(2).to_string())
    k = pd.concat([t.kappa_syn, t.kappa_mn])
    print(f"  razmah kappa po obeim zadacham: {k.min():.2f} .. {k.max():.2f} "
          f"= faktor {k.max()/k.min():.1f}")
    print(f"  razmah eps na analitike: {ns.max()-ns.min():.2f} edinits")
    rel = (t.kappa_mn / t.kappa_syn - 1).abs().drop("gaussian")
    print(f"  bez gaussian amplituda derzhitsya v predelah {rel.max():.0%}")

    head("Tablitsa 1 + risunok 1 -- perevoroty, mera Delta (analitika)")
    rs = reversal_stats(syn, metric="d")
    tot_pairs = 21 * syn.groupby(CELL).ngroups
    rs["resolvable"] = rs.n_pairs / tot_pairs
    rs["rev_of_all"] = rs["reversed"] / tot_pairs
    print(f"  vsego par v setke: {tot_pairs}")
    print(rs[["budget", "n_pairs", "reversed", "frac", "lo", "hi",
              "resolvable", "rev_of_all"]].round(4).to_string(index=False))

    head("Razdel 'ne zavisit ot mery' -- to zhe po L_adj")
    rl = reversal_stats(syn, metric="L_adj")
    print(rl[["budget", "n_pairs", "reversed", "frac", "lo", "hi"]].round(4).to_string(index=False))
    print(f"  diapazon po L_adj: {rl.frac.min():.1%} .. {rl.frac.max():.1%}"
          f"   po Delta: {rs.frac.min():.1%} .. {rs.frac.max():.1%}")

    head("Nulevaya kalibrovka (analitika, mera Delta)")
    nc = null_calibration(syn, metric="d")
    print(nc.round(4).to_string(index=False))

    head("MNIST -- perevoroty po L_adj i nulevaya kalibrovka")
    rm = reversal_stats(mn.assign(landscape="mnist", instance=0), metric="L_adj")
    print(rm[["budget", "n_pairs", "reversed", "frac", "lo", "hi"]].round(4).to_string(index=False))
    ncm = null_calibration(mn.assign(landscape="mnist", instance=0), metric="L_adj")
    print(ncm.round(4).to_string(index=False))

    head("Tablitsa 2 -- vred po agregatoram, tihiy i gromkiy krai")
    for name, d, lo, hi in [("analitika", syn, "-5.0", "10.0"),
                            ("MNIST", mn, "-4000.0", "4000.0")]:
        print(f"\n  {name}:  {lo} -> {hi}")
        for a in ["krum", "tukey", "talwar", "median", "trimmed", "mean"]:
            g = d[d["agg"] == a]
            print(f"    {a:9s} {g[g.eps_target == lo].L_adj.median():.3f}"
                  f" -> {g[g.eps_target == hi].L_adj.median():.3f}")

    head("Razmah vreda i razreshimost po byudzhetu (analitika, L_adj)")
    q = syn[syn.eps_target != "native"].copy()
    q["eps"] = q.eps_target.astype(float)
    for b in sorted(q.eps.unique()):
        g = q[q.eps == b]
        med = g.groupby(["agg", "attack"]).L_adj.median()
        print(f"  eps={b:>6}: razmah vreda {med.max()-med.min():.3f}")

    head("eps50 -- polosa poleznoy gromkosti (analitika, L_adj)")
    e = eps50_table(syn)
    ok = e[e.censored == 0]
    print(f"  yacheek dostigayut 0.5: {len(ok)} iz {len(e)}; "
          f"svyaznyh {int(ok.contiguous.sum())}, nesvyaznyh {int((1-ok.contiguous).sum())}")
    print(ok.groupby("agg")[["eps50", "eps_bar"]].median().round(2).to_string())
    print("  atak na agregator, dostigayushchih poroga:")
    print(ok.groupby("agg").attack.nunique().to_string())

    head("Smeshchenie ranga po atakam (analitika, mera Delta)")
    rows = []
    for b in sorted([x for x in syn.eps_target.unique() if x != "native"], key=float):
        for _, g in syn.groupby(CELL):
            n_ = g[g.eps_target == "native"].groupby("attack").d.median().rank()
            m_ = g[g.eps_target == b].groupby("attack").d.median().rank()
            for a in n_.index.intersection(m_.index):
                rows.append((a, abs(n_[a] - m_[a])))
    df = pd.DataFrame(rows, columns=["attack", "shift"])
    sh = df.groupby("attack")["shift"].mean().sort_values(ascending=False)
    print(pd.DataFrame({"sredniy sdvig ranga": sh.round(2),
                        "svoya eps": ns[sh.index].round(2)}).to_string())

    head("MNIST -- tochnost")
    a = mn[mn.eps_target == "native"]
    print(f"  chestnaya tochnost: {a.acc_honest.median():.3f}")
    p = mn.pivot_table(index="attack", columns="agg", values="acc_attacked",
                       aggfunc="min")
    print(p.round(3).to_string())


if __name__ == "__main__":
    main()
