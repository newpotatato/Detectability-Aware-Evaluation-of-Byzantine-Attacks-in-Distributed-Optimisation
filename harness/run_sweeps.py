"""
RAZVERTKI: razmer roya i dolya vrediteley.

Klyuchevoy vopros -- rezhim m=1. Pochti vse raboty berut 10-20% vrediteley;
pri m>=2 i IID ALIE otsenivaet statistiki chestnyh po SVOEY zhe gruppe, i
nablyudenie ey ne nuzhno. Pri m=1 etot priem otkazyvaet.
"""
from harness.runner import run_grid

AGGS = ["mean", "median", "trimmed", "krum", "talwar", "tukey"]
ATKS = ["signflip", "gaussian", "byzfl_1.5", "alie", "ipm", "minmax", "minsum"]

if __name__ == "__main__":
    for n_ag, n_mal in [(20, 1), (20, 2), (20, 4), (10, 1), (50, 1), (50, 5)]:
        run_grid(f"results/sweep_N{n_ag}_m{n_mal}.csv",
                 landscapes=["sphere", "rastrigin"], instances=[0, 1],
                 aggs=AGGS, attacks=ATKS,
                 budgets=["native", 0.3, 1.0, 3.0],
                 seeds=list(range(10)), T=200,
                 n_agents=n_ag, n_mal=n_mal, tag=f"N{n_ag}_m{n_mal}")
