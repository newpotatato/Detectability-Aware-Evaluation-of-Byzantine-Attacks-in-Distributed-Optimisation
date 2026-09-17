"""
Perezapusk yacheek Krum (f=4) v ostavshihsya setkah: ADAPT i SWEEP.

Prodolzhenie rerun_krum_f4.py. Te zhe soobrazheniya: Krum rabotal so svoim
umolchaniem f=2, poetomu vse ego yacheyki nado poschitat zanovo -- vklyuchaya
razvertki po dole vrediteley (risunok 5) i setku adaptacii (otricatelnyy
rezultat pro obratnuyu svyaz pervogo poryadka).

V razvertkah f dolzhno ravnyatsya IMENNO doli vrediteley v toy razvertke, a ne
chetyrem: eto delaet run_grid avtomaticheski cherez agg_kwargs_for(n_mal).
"""
from __future__ import annotations
import time
import numpy as np
from harness.runner import run_grid

ATKS = ["signflip", "gaussian", "byzfl_1.5", "alie", "ipm", "minmax", "minsum"]
KAPPA = np.logspace(-1.5, 1.0, 10)
FIXED = [f"fixed@{k:.4f}" for k in KAPPA]

if __name__ == "__main__":
    t0 = time.time()

    print("### 1/3 ADAPT (Krum, f=4)", flush=True)
    run_grid("results/m4_adapt_krum_f4.csv",
             landscapes=["sphere", "ellipsoid", "rastrigin", "gallagher"],
             instances=[0, 1], aggs=["krum"],
             attacks=FIXED + ["first_order", "global_search", "random"],
             budgets=[0.0, 1.0], seeds=list(range(16)),
             T=200, n_agents=20, n_mal=4, tag="m4_adapt")

    print("\n### 2/3 SWEEP po dole vrediteley (Krum, f = m)", flush=True)
    for m in [1, 2, 4, 6]:
        run_grid(f"results/sweep_m{m}_krum_f4.csv",
                 landscapes=["sphere", "rastrigin"], instances=[0, 1],
                 aggs=["krum"], attacks=ATKS,
                 budgets=["native", -3.0, 0.0, 3.0],
                 seeds=list(range(12)), T=200, n_agents=20, n_mal=m, tag=f"m{m}")

    print("\n### 3/3 SWEEP po razmeru roya (Krum, f = m)", flush=True)
    for N, m in [(10, 2), (50, 10), (100, 20)]:
        run_grid(f"results/sweep_N{N}_krum_f4.csv",
                 landscapes=["sphere", "rastrigin"], instances=[0],
                 aggs=["krum"], attacks=ATKS, budgets=["native", 0.0],
                 seeds=list(range(10)), T=200, n_agents=N, n_mal=m, tag=f"N{N}")

    print(f"\n### GOTOVO za {time.time()-t0:.0f} s", flush=True)
