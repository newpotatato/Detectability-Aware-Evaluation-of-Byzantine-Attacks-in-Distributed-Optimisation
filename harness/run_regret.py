"""
SOZHALENIE onlayn-podbora amplitudy.

Potolok -- nailuchshaya FIKSIROVANNAYA amplituda, naydennaya pereigryvaniem na
tom zhe side. Sravnivayutsya: pervyy poryadok (zakon FedSA), globalnyy poisk,
sluchaynyy vybor (nizhnyaya granitsa obucheniya).
"""
import numpy as np
from harness.runner import run_grid

KAPPA = np.logspace(-1.5, 1.0, 10)
FIXED = [f"fixed@{k:.4f}" for k in KAPPA]

if __name__ == "__main__":
    run_grid("results/regret.csv",
             landscapes=["sphere", "rastrigin", "gallagher"], instances=[0, 1],
             aggs=["mean", "median", "trimmed", "krum", "talwar", "tukey"],
             attacks=FIXED + ["first_order", "global_search", "random"],
             budgets=[1.0], seeds=list(range(12)), T=200, tag="regret")
