"""
GLAVNAYA SETKA, versiya 2 -- na ISPRAVLENNOY metrike.

Otlichiya ot versii 1:
  * simmetrichnaya metrika vreda (ogibayushchaya u oboih krivyh);
  * diapazon byudzhetov rasshiren v OTRITSATELNUYU oblast: ALIE i IPM po prirode
    sidyat na eps ~ -5, i bez etogo uravnivanie s nimi nevozmozhno;
  * 16 sidov vmesto 12 -- v proshloy versii 3/4 par byli nerazlichimy.
"""
from harness.runner import run_grid

if __name__ == "__main__":
    run_grid(
        "results/main2.csv",
        landscapes=["sphere", "ellipsoid", "rastrigin", "gallagher"],
        instances=[0, 1],
        aggs=["mean", "median", "trimmed", "krum", "talwar", "tukey"],
        attacks=["signflip", "gaussian", "byzfl_1.5", "alie", "ipm", "minmax", "minsum"],
        budgets=["native", -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 3.0, 10.0],
        seeds=list(range(16)),
        T=200, n_agents=20, n_mal=1, d=10, tag="main2",
    )
