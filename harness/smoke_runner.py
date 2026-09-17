"""Bystraya proverka dvizhka pered dlinnymi progonami."""
from harness.runner import run_grid

if __name__ == "__main__":
    run_grid("results/_smoke.csv",
             landscapes=["sphere"], instances=[0], aggs=["median", "krum"],
             attacks=["signflip", "alie"], budgets=["native", 1.0],
             seeds=list(range(4)), T=120, tag="smoke")
