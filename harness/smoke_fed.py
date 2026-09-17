"""Proverka zadachi FL do dlinnyh progonov."""
import time, numpy as np
from harness import fedsgd as FS, attacks as AT, metrics as ME

if __name__ == "__main__":
    cfg = FS.FedConfig(T=150, eps_budget=None)
    t0 = time.time()
    tr = FS.run_fed("mean", cfg, 0, attack=None, collect_honest=True)
    print(f"chestnyy progon: {time.time()-t0:.1f} s")
    print(f"  test-loss {tr.f_test[0]:.3f} -> {tr.f_test[-1]:.3f}")
    print(f"  tochnost  {tr.acc[0]:.3f} -> {tr.acc[-1]:.3f}")

    d_eff = FS.diag_effective_d(tr.honest_rounds[:60])
    print(f"\nd_eff = {d_eff:.1f}  (nominalnaya razmernost {len(tr.honest_rounds[0][0])})")

    print(f"\n{'ataka':<12}{'loss':>9}{'tochnost':>10}{'L':>8}{'eps':>9}{'sek':>7}")
    cfgA = FS.FedConfig(T=150, eps_budget=None, d_eff=d_eff)
    for nm, mk in [("net ataki", AT.Honest),
                   ("signflip", lambda: AT.SignFlip(1.0)),
                   ("alie", lambda: AT.ALIE()),
                   ("ipm", lambda: AT.IPM(2.0)),
                   ("minmax", lambda: AT.MinMax())]:
        t1 = time.time()
        a = FS.run_fed("mean", cfgA, 0, attack=mk())
        Lv = ME.L_single(tr.f_test, a.f_test)
        print(f"{nm:<12}{a.f_test[-1]:>9.3f}{a.acc[-1]:>10.3f}"
              f"{(Lv if Lv is ME.HELP else f'{Lv:.3f}'):>8}"
              f"{np.nanmean(a.eps_spent):>9.1f}{time.time()-t1:>7.1f}")
