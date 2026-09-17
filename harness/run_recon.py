"""
RAZVEDKA: gde nahoditsya GRADUIROVANNAYA oblast po gromkosti?

Glavnaya setka pokazala eps50 = 0.1 (minimum setki) VO VSEH yacheykah, to est
ves eksperiment proshel VYSHE tochki nasyshcheniya. Zdes ishchem, pri kakih
byudzhetah vred voobshche reagiruet na gromkost.

Klyuchevoe: diapazon rasshiren v OTRITSATELNUYU oblast. ALIE i IPM po prirode
sidyat na eps ~ -5, i eto ne krayniy sluchay, a ih shtatnyy rezhim. Byudzhet
osmyslen do -d_eff/2 ~ -5.9, gde radius shara obrashchaetsya v nol.
"""
from harness.runner import run_grid

BUDGETS = ["native", -5.5, -5.0, -4.5, -4.0, -3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 3.0, 10.0]

if __name__ == "__main__":
    run_grid("results/recon.csv",
             landscapes=["sphere", "rastrigin"], instances=[0],
             aggs=["mean", "median", "trimmed", "krum", "talwar", "tukey"],
             attacks=["signflip", "gaussian", "byzfl_1.5", "alie", "ipm", "minmax", "minsum"],
             budgets=BUDGETS, seeds=list(range(6)), T=200, tag="recon")
