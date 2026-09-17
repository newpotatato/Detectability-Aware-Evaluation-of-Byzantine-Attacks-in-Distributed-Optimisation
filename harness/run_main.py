"""
GLAVNAYA SETKA V1: sravnenie atak pri ravnom byudzhete.

4 landshafta x 3 ekzemplyara x 6 agregatorov x 7 atak x 6 urovney byudzheta
x 12 sidov. Syrye dannye posidno -> results/main.csv
"""
from harness.runner import run_grid

if __name__ == "__main__":
    run_grid(
        "results/main.csv",
        landscapes=["sphere", "ellipsoid", "rastrigin", "gallagher"],
        instances=[0, 1, 2],
        aggs=["mean", "median", "trimmed", "krum", "talwar", "tukey"],
        attacks=["signflip", "gaussian", "byzfl_1.5", "alie", "ipm", "minmax", "minsum"],
        budgets=["native", 0.1, 0.3, 1.0, 3.0, 10.0],
        seeds=list(range(12)),
        T=200, n_agents=20, n_mal=1, d=10, tag="main",
    )
