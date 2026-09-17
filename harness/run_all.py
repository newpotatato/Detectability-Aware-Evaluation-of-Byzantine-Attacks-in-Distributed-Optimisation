"""
POLNAYA PROGRAMMA V1 na ispravlennyh metrikah, m=4 (20% vrediteley).

Razvedka 2026-08-28 pokazala: pri m=1 ataka POMOGAET v 43% progonov i mediannyy
vred 0.04 -- sravnivat poryadok atak v takom rezhime bessmyslenno. Pri m=4
pomogaet lish 14.6%, mediannyy vred 0.58, dve treti progonov teryayut polovinu
progressa. m=4 sootvetstvuet standartu literatury (10-20%).

  1. MAIN   -- glavnaya gipoteza: menyaetsya li poryadok atak pri ravnom byudzhete
  2. ADAPT  -- sozhalenie i perepoverka gipotezy pervogo poryadka (E0 na ispravlennoy
               metrike: staryy verdikt byl vynesen na slomannom instrumente)
  3. SWEEP  -- razvertki po dole vrediteley i razmeru roya
"""
import numpy as np
from harness.runner import run_grid

AGGS = ["mean", "median", "trimmed", "krum", "talwar", "tukey"]
ATKS = ["signflip", "gaussian", "byzfl_1.5", "alie", "ipm", "minmax", "minsum"]
BUDG = ["native", -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 3.0, 10.0]
KAPPA = np.logspace(-1.5, 1.0, 10)
FIXED = [f"fixed@{k:.4f}" for k in KAPPA]

if __name__ == "__main__":
    print("### 1/3 MAIN", flush=True)
    run_grid("results/m4_main.csv",
             landscapes=["sphere", "ellipsoid", "rastrigin", "gallagher"],
             instances=[0, 1], aggs=AGGS, attacks=ATKS, budgets=BUDG,
             seeds=list(range(16)), T=200, n_agents=20, n_mal=4, tag="m4_main")

    print("\n### 2/3 ADAPT (sozhalenie + pervyy poryadok protiv globalnogo)", flush=True)
    run_grid("results/m4_adapt.csv",
             landscapes=["sphere", "ellipsoid", "rastrigin", "gallagher"],
             instances=[0, 1], aggs=AGGS,
             attacks=FIXED + ["first_order", "global_search", "random"],
             budgets=[0.0, 1.0], seeds=list(range(16)),
             T=200, n_agents=20, n_mal=4, tag="m4_adapt")

    print("\n### 3/3 SWEEP", flush=True)
    for m in [1, 2, 4, 6]:
        run_grid(f"results/sweep_m{m}.csv",
                 landscapes=["sphere", "rastrigin"], instances=[0, 1],
                 aggs=AGGS, attacks=ATKS, budgets=["native", -3.0, 0.0, 3.0],
                 seeds=list(range(12)), T=200, n_agents=20, n_mal=m, tag=f"m{m}")
    for N, m in [(10, 2), (50, 10), (100, 20)]:
        run_grid(f"results/sweep_N{N}.csv",
                 landscapes=["sphere", "rastrigin"], instances=[0],
                 aggs=AGGS, attacks=ATKS, budgets=["native", 0.0],
                 seeds=list(range(10)), T=200, n_agents=N, n_mal=m, tag=f"N{N}")
    print("\n### VSE PROGONY ZAVERSHENY", flush=True)
