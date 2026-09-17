"""Setka na realnoy zadache: MNIST + FedSGD, m=4 iz 20 (20% vrediteley)."""
from harness.runner_fed import run_fed_grid
if __name__ == "__main__":
    run_fed_grid("results/mnist.csv",
                 aggs=["mean", "median", "trimmed", "krum", "talwar", "tukey"],
                 attacks=["signflip", "gaussian", "byzfl_1.5", "alie", "ipm",
                          "minmax", "minsum"],
                 budgets=["native", -4000.0, -2000.0, 0.0, 2000.0, 4000.0],
                 seeds=list(range(8)), T=150, n_clients=20, n_mal=4,
                 # 6 protsessov vmesto 11: kazhdyy derzhit svoyu kopiyu MNIST
                 # (~220 MB) plyus rabochie massivy modeli
                 workers=10)
