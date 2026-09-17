"""Diagnostika: pri kakom byudzhete metrika vreda PERESTAET byt nasyshchennoy."""
import numpy as np
from harness import landscapes as LS, swarm as SW, attacks as AT, metrics as ME

if __name__ == "__main__":
    L = LS.make("sphere", d=10)
    seeds = range(8)
    GRID = np.logspace(-0.5, 2.0, 10)
    for eps in [None, 3.0, 0.5]:
        cfg = SW.Config(T=200, lr=0.08, eta=0.3, eps_budget=eps)
        honest = {s: SW.run(L, "krum", cfg, s, attack=None) for s in seeds}
        print(f"\n--- eps_budget = {eps} ---")
        print(f"{'kappa':>8}{'L_med':>9}{'d_med':>9}{'clip%':>8}{'eps_fakt':>10}")
        for k in GRID:
            pairs, clips, epss = [], [], []
            for s in seeds:
                tr = SW.run(L, "krum", cfg, s, attack=AT.FixedAmplitude(k))
                pairs.append((honest[s].f_consensus, tr.f_consensus))
                clips.append(tr.clipped.mean())
                epss.append(np.nanmean(tr.eps_spent))
            r = ME.summarize(pairs, f_opt=L.f_opt)
            e = np.nanmean(epss)
            print(f"{k:>8.2f}{r['L_med']:>9.3f}{r['d_med']:>9.2f}{np.mean(clips):>8.0%}"
                  + (f"{e:>10.2f}" if np.isfinite(e) else f"{'-':>10}"))
