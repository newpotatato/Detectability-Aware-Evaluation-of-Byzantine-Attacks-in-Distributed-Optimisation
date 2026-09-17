"""Masshtab smeshcheniya i chto ono otmenyaet."""
import numpy as np
from harness import landscapes as LS, swarm as SW, attacks as AT, metrics as ME, budget as BUD
from harness.runner import TUNED

def L_fixed(fh, fa):
    """Simmetrichnaya versiya: ogibayushchaya u OBOIH."""
    return ME.L_single(fh, np.minimum.accumulate(fa))

def d_fixed(fh, fa, fopt):
    return ME.d_single(fh, np.minimum.accumulate(fa), fopt)

for lname, agg in [("sphere", "median"), ("rastrigin", "krum")]:
    L = LS.make(lname, d=10)
    cfg = SW.Config(T=200, eps_budget=None, **TUNED[lname])
    seeds = range(10)
    hon = {s: SW.run(L, agg, cfg, s, attack=None) for s in seeds}
    print(f"\n=== {lname} / {agg} ===")
    print(f"{'':<34}{'L star':>9}{'L nov':>9}{'d star':>9}{'d nov':>9}")
    for label, mk in [("NET ATAKI (agent 0 chesten)", AT.Honest),
                      ("signflip", lambda: AT.SignFlip(1.0)),
                      ("gaussian", lambda: AT.GaussianNoise(1.0))]:
        po, pn = [], []
        for s in seeds:
            tr = SW.run(L, agg, cfg, s, attack=mk())
            po.append((hon[s].f_consensus, tr.f_consensus))
            pn.append((hon[s].f_consensus, np.minimum.accumulate(tr.f_consensus)))
        ro = ME.summarize(po, f_opt=L.f_opt); rn = ME.summarize(pn, f_opt=L.f_opt)
        print(f"{label:<34}{ro['L_med']:>9.3f}{rn['L_med']:>9.3f}"
              f"{ro['d_med']:>9.3f}{rn['d_med']:>9.3f}")

print("\n\nRANGI: vliyaet li smeshchenie na SRAVNENIE atak mezhdu soboy?")
print("d = log(ga) - log(gh); gh odinakova dlya vseh atak v yacheyke,")
print("znachit RAZNOSTI d mezhdu atakami ot nee ne zavisyat.\n")
L = LS.make("sphere", d=10)
cfg = SW.Config(T=200, eps_budget=None, **TUNED["sphere"])
seeds = range(10)
hon = {s: SW.run(L, "krum", cfg, s, attack=None) for s in seeds}
res = {}
for nm, mk in [("signflip", lambda: AT.SignFlip(1.0)),
               ("gaussian", lambda: AT.GaussianNoise(1.0)),
               ("byzfl", lambda: AT.FixedAmplitude(1.5))]:
    po, pn = [], []
    for s in seeds:
        tr = SW.run(L, "krum", cfg, s, attack=mk())
        po.append((hon[s].f_consensus, tr.f_consensus))
        pn.append((hon[s].f_consensus, np.minimum.accumulate(tr.f_consensus)))
    res[nm] = (ME.summarize(po, f_opt=L.f_opt)["d_med"],
               ME.summarize(pn, f_opt=L.f_opt)["d_med"])
print(f"{'ataka':<12}{'d star':>9}{'d nov':>9}")
for k, (a, b) in res.items():
    print(f"{k:<12}{a:>9.3f}{b:>9.3f}")
o = sorted(res, key=lambda k: -res[k][0]); n = sorted(res, key=lambda k: -res[k][1])
print(f"\nporyadok staryy: {o}\nporyadok novyy:  {n}\nsovpadaet: {o == n}")
