"""Smoke-test: chestnyy roy shoditsya, prostaya ataka meshaet, parnost soblyudena."""
import numpy as np
from harness import landscapes as LS, swarm as SW, aggregators as AGG


class SignFlip(SW.Attack):
    name = "signflip"
    def __init__(self, kappa=1.0): self.kappa = kappa
    def propose(self, ctx):
        return ctx.x_own + self.kappa * (ctx.honest_proposal - ctx.x_own) * (-1.0)


if __name__ == "__main__":
    cfg = SW.Config(n_agents=20, n_malicious=1, T=300, eps_budget=None)
    print(f"{'landshaft':<12}{'agregator':<12}{'f chestn':>12}{'f ataka':>12}{'otnosh':>9}")
    print("-" * 57)
    for lname in ["sphere", "rastrigin"]:
        L = LS.make(lname, d=10)
        for aname in AGG.CORE:
            h, a = SW.paired_run(L, aname, cfg, seed=0, attack=SignFlip())
            fh = h.f_consensus[-1] - L.f_opt
            fa = a.f_consensus[-1] - L.f_opt
            print(f"{lname:<12}{aname:<12}{fh:>12.4g}{fa:>12.4g}{fa/max(fh,1e-12):>9.2f}")

    print("\nProverka parnosti: dva chestnyh progona na odnom side sovpadayut pobitno")
    L = LS.make("sphere", d=10)
    t1 = SW.run(L, "median", cfg, seed=7)
    t2 = SW.run(L, "median", cfg, seed=7)
    print(f"  max|raznitsa| = {np.max(np.abs(t1.f_consensus - t2.f_consensus)):.2e}")

    print("\nProverka: bez ataki traektoriya ubyvaet")
    print(f"  f(c) v nachale = {t1.f_consensus[0]:.4g}, v kontse = {t1.f_consensus[-1]:.4g}")
