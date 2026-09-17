"""RAZVEDKA po sile vreditelya: pri kakom m vred nachinaet dominirovat?"""
from harness.runner import run_grid
if __name__ == "__main__":
    for m in [1, 2, 4, 6]:
        run_grid(f"results/reconm_{m}.csv",
                 landscapes=["sphere", "rastrigin"], instances=[0],
                 aggs=["median", "trimmed", "krum", "tukey"],
                 attacks=["signflip", "gaussian", "alie", "minmax"],
                 budgets=["native", -3.0, 0.0, 3.0],
                 seeds=list(range(8)), T=200, n_agents=20, n_mal=m, tag=f"m{m}")
