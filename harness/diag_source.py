"""Otkuda beretsya vred: ot ATAKI ili ot samoy PROEKTSII / smeny sostava roya?"""
import numpy as np
from harness import landscapes as LS, swarm as SW, attacks as AT, metrics as ME, budget as BUD
from harness.runner import TUNED

L = LS.make("sphere", d=10)
seeds = range(10)
c0 = SW.Config(T=200, eps_budget=None, **TUNED["sphere"])
tr0 = SW.run(L, "median", c0, 0, attack=None, collect_honest=True)
d_eff = BUD.effective_d(tr0.honest_rounds[:120])
print(f"d_eff = {d_eff:.2f}\n")

hon = {s: SW.run(L, "median", c0, s, attack=None) for s in seeds}

def harm(mk, eps, two_sided=True):
    cfg = SW.Config(T=200, eps_budget=eps, d_eff=d_eff, two_sided=two_sided,
                    **TUNED["sphere"])
    pairs, es = [], []
    for s in seeds:
        tr = SW.run(L, "median", cfg, s, attack=mk())
        pairs.append((hon[s].f_consensus, tr.f_consensus))
        es.append(np.nanmean(tr.eps_spent))
    r = ME.summarize(pairs, f_opt=L.f_opt)
    return r["L_med"], r["d_med"], float(np.nanmean(es))

print(f"{'stsenariy':<46}{'L':>7}{'d':>7}{'eps':>8}")
print("-" * 68)
for label, mk, eps, ts in [
    ("agent 0 CHESTEN, bez byudzheta",        AT.Honest, None, True),
    ("agent 0 CHESTEN, no proektsiya eps=-5.5", AT.Honest, -5.5, True),
    ("agent 0 CHESTEN, no proektsiya eps=0",    AT.Honest, 0.0, True),
    ("agent 0 CHESTEN, no proektsiya eps=+3",   AT.Honest, 3.0, True),
    ("signflip, eps=-5.5",                    lambda: AT.SignFlip(1.0), -5.5, True),
    ("signflip, eps=+3",                      lambda: AT.SignFlip(1.0), 3.0, True),
]:
    l_, d_, e_ = harm(mk, eps, ts)
    print(f"{label:<46}{l_:>7.3f}{d_:>7.3f}{e_:>8.2f}")

print("\nKONTROL: 20 chestnyh protiv 20 chestnyh (raznye sidy shuma NE dolzhny)")
a = SW.run(L, "median", c0, 3, attack=None)
b = SW.run(L, "median", c0, 3, attack=None)
print(f"  identichnost: {np.array_equal(a.f_consensus, b.f_consensus)}")
print(f"\nKONTROL: chto vozvrashchaet L, esli atakovannyy = chestnomu pobitno")
print(f"  L = {ME.L_single(hon[0].f_consensus, hon[0].f_consensus)}")
