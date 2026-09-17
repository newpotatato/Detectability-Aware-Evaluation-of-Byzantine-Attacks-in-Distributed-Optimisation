"""Proverka klyuchevyh invariantov progona exp4b, ne trogaya sam progon."""
import numpy as np
from harness import landscapes as LS, swarm as SW, attacks as AT, budget as BUD
from harness.exp0 import TUNED
from harness.exp4 import ATTACKS

L = LS.make("sphere", d=10)
c0 = SW.Config(T=120, eps_budget=None, **TUNED["sphere"])
tr0 = SW.run(L, "median", c0, 0, attack=None, collect_honest=True)
d_eff = BUD.effective_d(tr0.honest_rounds[:100])
print(f"d_eff = {d_eff:.3f}\n")

print("1) URAVNIVANIE: fakticheski potrachennaya eps dolzhna RAVNYATSYA tselevoy")
print(f"{'ataka':<11}" + "".join(f"{'cel='+str(b):>12}" for b in [0.3, 1.0, 3.0, 10.0]))
ok = True
for a, mk in ATTACKS.items():
    row = f"{a:<11}"
    for b in [0.3, 1.0, 3.0, 10.0]:
        cfg = SW.Config(T=120, eps_budget=b, d_eff=d_eff, two_sided=True, **TUNED["sphere"])
        e = np.nanmean(SW.run(L, "median", cfg, 0, attack=mk()).eps_spent)
        row += f"{e:>12.4f}"
        if abs(e - b) > 1e-6: ok = False
    print(row)
print(f"   -> {'OK: uravnivanie tochnoe' if ok else 'PROBLEMA: eps ne ravna tseli'}\n")

print("2) NATIVNYY progon: eps dolzhna RAZLICHATSYA mezhdu atakami")
cfgN = SW.Config(T=120, eps_budget=50.0, d_eff=d_eff, two_sided=False, **TUNED["sphere"])
es = {}
for a, mk in ATTACKS.items():
    es[a] = float(np.nanmean(SW.run(L, "median", cfgN, 0, attack=mk()).eps_spent))
    print(f"   {a:<11}{es[a]:>9.2f}")
sp = max(es.values()) - min(es.values())
print(f"   -> razbros {sp:.2f} {'OK' if sp > 1 else 'PROBLEMA: vse odinakovy'}\n")

print("3) CHESTNYY progon ne zavisit ot byudzheta (kesh mozhno perepolzovat)")
h1 = SW.run(L, "median", SW.Config(T=120, eps_budget=0.3, **TUNED["sphere"]), 0)
h2 = SW.run(L, "median", SW.Config(T=120, eps_budget=99., **TUNED["sphere"]), 0)
same = np.array_equal(h1.f_consensus, h2.f_consensus)
print(f"   -> {'OK: sovpadayut pobitno (4x lishney raboty v exp4b)' if same else 'PROBLEMA'}\n")

print("4) NASYSHCHENIE: skolko atak dayut prakticheski odinakovyy vred (svyazki v range)")
from harness import metrics as ME
for agg in ["mean", "median", "krum"]:
    cfg = SW.Config(T=120, eps_budget=1.0, d_eff=d_eff, two_sided=True, **TUNED["sphere"])
    hon = {s: SW.run(L, agg, cfg, s, attack=None) for s in range(6)}
    vals = []
    for a, mk in ATTACKS.items():
        pairs = [(hon[s].f_consensus, SW.run(L, agg, cfg, s, attack=mk()).f_consensus)
                 for s in range(6)]
        vals.append(ME.summarize(pairs, f_opt=L.f_opt)["d_med"])
    vals = np.array(vals)
    rng_ = vals.max() - vals.min()
    print(f"   {agg:<8} razmah vreda = {rng_:.4f}"
          f"   {'MALO -> rangi shumnye' if rng_ < 0.05 else 'ok'}")
