"""
POLNAYA POVERKA pered obyavleniem rezultatov validnymi.

Kazhdaya proverka -- CHISLENNAYA, a ne rassuzhdeniem. Za den bylo naydeno pyat
oshibok, kazhdaya iz kotoryh davala pravdopodobnye i nevernye chisla, poetomu
doverie k rassuzhdeniyu zdes neumestno.
"""
from __future__ import annotations
import itertools
import numpy as np

from . import landscapes as LS, swarm as SW, attacks as AT, metrics as ME
from . import aggregators as AGG, budget as BUD, fedsgd as FS
from .runner import TUNED

OK, FAIL = "OK  ", "FAIL"
res = []


def check(name, cond, detail=""):
    res.append((cond, name))
    print(f"  [{OK if cond else FAIL}] {name}" + (f"   {detail}" if detail else ""))


# ============================================================ A. metriki
print("\nA. METRIKI VREDA")
T = 200
rng = np.random.default_rng(0)
H = 100 * np.exp(np.linspace(0, np.log(1e-3), T)) * (1 + 0.3 * rng.random(T))
# ISPRAVLENO 2026-09-05. Ranshe zdes proveryalos L_single(H, H) ~ 0, i proverka
# PADALA -- no ne iz-za oshibki v konveyere, a potomu chto trebovala nulya ot
# velichiny, u kotoroy est sobstvennyy nenulevoy POL (sm. ME.L_baseline). Nol pri
# nulevom vrede daet ne L, a L_adjusted; imenno ona i vhodit vo vse svodki.
check("L_adj(chestnyy, chestnyy) = 0", abs(ME.L_adjusted(H, H)) < 1e-12,
      f"= {ME.L_adjusted(H, H):.2e}   (sam pol L = {ME.L_baseline(H):.4f})")
check("d(chestnyy, chestnyy) = 0", abs(ME.d_single(H, H, 0.0)) < 1e-12,
      f"= {ME.d_single(H, H, 0.0):.2e}")
A = np.full(T, np.minimum.accumulate(H)[T // 2])
check("L(konchil v seredine) = 0.5", abs(ME.L_single(H, A) - 0.5) < 1e-9,
      f"= {ME.L_single(H, A):.4f}")
check("L(uroven starta) = 1", abs(ME.L_single(H, np.full(T, H[0])) - 1.0) < 0.01,
      f"= {ME.L_single(H, np.full(T, H[0])):.4f}")
span = np.minimum.accumulate(H)[0] - np.minimum.accumulate(H)[-1]
check("L(otbroshen za start na 1 dlinu) = 2",
      abs(ME.L_single(H, np.full(T, H[0] + span)) - 2.0) < 0.02,
      f"= {ME.L_single(H, np.full(T, H[0] + span)):.4f}")
check("ataka POMOGLA raspoznaetsya", ME.L_single(H, H * 0.5) is ME.HELP)
check("simmetriya: ogibayushchaya u OBOIH",
      ME.L_single(H, H * 1.0) == ME.L_single(H, np.minimum.accumulate(H)))

# ============================================================ B. parnost
print("\nB. PARNOST KONTRFAKTICHESKOGO PROGONA")
L10 = LS.make("sphere", d=10)
cfg = SW.Config(T=120, eps_budget=None, **TUNED["sphere"])
a = SW.run(L10, "median", cfg, 5, attack=None)
b = SW.run(L10, "median", cfg, 5, attack=None)
check("dva chestnyh progona na odnom side pobitno ravny",
      np.array_equal(a.f_consensus, b.f_consensus))
c = SW.run(L10, "median", cfg, 5, attack=AT.Honest())
check("progon s CHESTNYM 'vreditelem' pobitno raven chestnomu",
      np.array_equal(a.f_consensus, c.f_consensus),
      "znachit rashozhdenie voznikaet TOLKO ot deystviy ataki")
d1 = SW.run(L10, "median", cfg, 5, attack=AT.SignFlip(1.0))
d2 = SW.run(L10, "median", cfg, 6, attack=AT.SignFlip(1.0))
check("raznye sidy dayut raznye traektorii",
      not np.array_equal(d1.f_consensus, d2.f_consensus))

# ============================================================ C. byudzhet
print("\nC. BYUDZHET NEZAMETNOSTI")
tr0 = SW.run(L10, "median", SW.Config(T=200, eps_budget=None, **TUNED["sphere"]),
             0, attack=None, collect_honest=True)
d_eff = BUD.effective_d(tr0.honest_rounds[:120])
errs = []
for eps in [-5.0, -2.0, 0.0, 1.0, 10.0]:
    cfgb = SW.Config(T=120, eps_budget=eps, d_eff=d_eff, two_sided=True,
                     **TUNED["sphere"])
    for atk in [AT.SignFlip(1.0), AT.ALIE(), AT.MinMax()]:
        e = np.nanmean(SW.run(L10, "median", cfgb, 0, attack=atk).eps_spent)
        errs.append(abs(e - eps))
check("uravnivanie TOCHNOE dlya vseh atak i byudzhetov", max(errs) < 1e-6,
      f"max otklonenie ot tseli = {max(errs):.2e}")
mu, S = BUD.honest_moments(tr0.honest_rounds[0])
z = mu + 100 * np.ones(10)
zp, _ = BUD.project(z, mu, S, 1.0, two_sided=True, d_eff=d_eff)
check("proektsiya vozvrashchaet tochku NA sfere zadannogo radiusa",
      abs(BUD.mahalanobis_sq(zp, mu, np.linalg.inv(S)) - (d_eff + 2.0)) < 1e-8)
zq = mu + 1e-4 * np.ones(10)
zqp, _ = BUD.project(zq, mu, S, 1.0, two_sided=True, d_eff=d_eff)
check("dvuhstoronnost: TIHAYA otpravka podnimaetsya do tseli",
      abs(BUD.mahalanobis_sq(zqp, mu, np.linalg.inv(S)) - (d_eff + 2.0)) < 1e-8,
      "bez etogo ALIE (eps=-4.75) ne uravnivalas by voobshche")

# ============================================================ D. agregatory
print("\nD. AGREGATORY")
X = np.random.default_rng(1).normal(size=(21, 40))
check("trimmed(alpha=0) == mean",
      np.allclose(AGG.trimmed_mean(X, alpha=0.0), AGG.mean(X)))
check("median == pokoordinatnaya mediana numpy",
      np.allclose(AGG.median(X), np.median(X, axis=0)))
check("krum vozvrashchaet ODNU IZ otpravok",
      any(np.allclose(AGG.krum(X), X[i]) for i in range(len(X))))
a_ = np.einsum("ij,ij->i", X, X)
D_fast = np.maximum(a_[:, None] + a_[None, :] - 2 * (X @ X.T), 0)
D_naive = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2) ** 2
check("bystrye rasstoyaniya krum == naivnye",
      np.abs(D_fast - D_naive).max() / D_naive.max() < 1e-12,
      f"otn. rashozhdenie {np.abs(D_fast-D_naive).max()/D_naive.max():.1e}")
Y = np.random.default_rng(2).normal(size=(20, 15))
base = {k: AGG.get(k)(Y) for k in AGG.AGGREGATORS}
u = np.zeros(15); u[0] = 1.0
far = {k: np.linalg.norm(AGG.get(k)(np.vstack([Y, 1e6 * u])) - base[k])
       for k in AGG.AGGREGATORS}
check("mean NE ogranichen (vliyanie rastet s amplitudoy)", far["mean"] > 1e3,
      f"sdvig {far['mean']:.1e}")
check("robastnye OGRANICHENY pri gigantskom vybrose",
      all(far[k] < 10 for k in ["median", "trimmed", "krum", "tukey", "talwar"]),
      f"max sdvig {max(far[k] for k in ['median','trimmed','krum','tukey','talwar']):.3f}")

# ============================================================ E. ataki
print("\nE. ATAKI")
Yh = np.random.default_rng(3).normal(size=(16, 60))
mu_ = Yh.mean(axis=0)
p_ = -AT._unit(Yh.std(axis=0))
Z = Yh - mu_
a2 = np.einsum("ij,ij->i", Z, Z); b2 = Z @ p_; c2 = float(p_ @ p_)
lam = 3.7
lhs = a2 - 2 * lam * b2 + lam * lam * c2
rhs = np.linalg.norm(Yh - (mu_ + lam * p_), axis=1) ** 2
check("razlozhenie rasstoyaniy v MinMax/MinSum tochnoe",
      np.abs(lhs - rhs).max() / rhs.max() < 1e-10,
      f"otn. rashozhdenie {np.abs(lhs-rhs).max()/rhs.max():.1e}")

seen = {}
class Spy(AT.SignFlip):
    name, level = "spy0", "0a"
    def propose(self, ctx):
        seen["oracle0"] = ctx.oracle
        return super().propose(ctx)
class Spy1(AT.ALIE):
    name, level = "spy1", "1"
    def propose(self, ctx):
        seen["oracle1"] = ctx.oracle
        return super().propose(ctx)
SW.run(L10, "median", cfg, 0, attack=Spy())
SW.run(L10, "median", cfg, 0, attack=Spy1())
check("uroven 0 NE poluchaet orakul (zashchita ot utechki)",
      seen.get("oracle0") is None)
check("uroven 1 orakul poluchaet", seen.get("oracle1") is not None)

# ============================================================ F. FL-zadacha
print("\nF. ZADACHA MNIST")
Xm, ym = FS.load_mnist()
check("MNIST zagruzhen i normirovan", Xm.shape == (70000, 784)
      and 0.0 <= Xm.min() and Xm.max() <= 1.0, f"{Xm.shape}, [{Xm.min()}, {Xm.max()}]")
net = FS.MLP(h=32)
w0 = net.init(np.random.default_rng(0))
idx = np.stack([np.random.default_rng(k).choice(len(Xm), 32, replace=False)
                for k in range(20)])
Gl = np.stack([net.loss_grad(w0, Xm[idx[c]], ym[idx[c]])[1] for c in range(20)])
Gb = net.grads_batch(w0, Xm[idx], ym[idx])
check("batchevye gradienty == potsiklovye",
      np.abs(Gl - Gb).max() / np.abs(Gl).max() < 1e-12,
      f"otn. rashozhdenie {np.abs(Gl-Gb).max()/np.abs(Gl).max():.1e}")
gnum = np.zeros(5)
for j in range(5):
    e = np.zeros(net.dim); e[j * 1000] = 1e-5
    gnum[j] = (net.loss_grad(w0 + e, Xm[idx[0]], ym[idx[0]])[0]
               - net.loss_grad(w0 - e, Xm[idx[0]], ym[idx[0]])[0]) / 2e-5
gana = Gl[0][[j * 1000 for j in range(5)]]
check("analiticheskiy gradient == chislennyy",
      np.abs(gnum - gana).max() / (np.abs(gnum).max() + 1e-12) < 1e-4,
      f"otn. rashozhdenie {np.abs(gnum-gana).max()/(np.abs(gnum).max()+1e-12):.1e}")

cfgf = FS.FedConfig(T=25, eps_budget=None)
h0 = FS.run_fed("mean", cfgf, 0, attack=None, collect_honest=True)
de_f = FS.diag_effective_d(h0.honest_rounds[:20])
check("d_eff na MNIST poryadka razmernosti, a ne na poryadki bolshe",
      0.2 * net.dim < de_f < 2.0 * net.dim,
      f"d_eff = {de_f:.0f}, razmernost = {net.dim}")
errs_f = []
for eps in [-3000.0, 0.0, 3000.0]:
    cf = FS.FedConfig(T=25, eps_budget=eps, d_eff=de_f, two_sided=True)
    e = np.nanmean(FS.run_fed("mean", cf, 0, attack=AT.SignFlip(1.0)).eps_spent)
    errs_f.append(abs(e - eps))
check("uravnivanie na MNIST tochnoe", max(errs_f) < 1e-3,
      f"max otklonenie {max(errs_f):.2e}")
hh = FS.run_fed("mean", FS.FedConfig(T=25, eps_budget=None), 0, attack=AT.Honest())
# ISPRAVLENO 2026-09-05: proveryaem L_adj, a ne L (sm. blok A). Traektorii zdes
# sovpadayut POBITNO, i syroe L ravno svoemu polu, a ne nulyu.
check("MNIST: progon s chestnym 'vreditelem' ~ chestnyy",
      abs(ME.L_adjusted(h0.f_test, hh.f_test)) < 1e-12,
      f"L_adj = {ME.L_adjusted(h0.f_test, hh.f_test):.2e}   "
      f"(pol L = {ME.L_baseline(h0.f_test):.3f})")

# ============================================================ G. statistika
print("\nG. STATISTICHESKAYA PROTSEDURA")
r = np.random.default_rng(7)
same = r.normal(size=40)
check("confident_order: odinakovye vyborki -> 'ne razlichimy'",
      ME_ord := AT and __import__("harness.analyze", fromlist=["x"]).confident_order(
          same, same + r.normal(0, 1e-9, 40), r) == 0)
an = __import__("harness.analyze", fromlist=["x"])
check("confident_order: yavno bolshaya -> +1",
      an.confident_order(same + 5.0, same, r) == 1)
check("confident_order: yavno menshaya -> -1",
      an.confident_order(same, same + 5.0, r) == -1)
check("confident_order: shum bez sdviga -> 0 (net lozhnyh perevorotov)",
      an.confident_order(r.normal(size=30), r.normal(size=30), r) == 0)

# ============================================================ itog
print("\n" + "=" * 62)
bad = [n for c, n in res if not c]
print(f"PROVEROK: {len(res)},  PROSHLI: {len(res)-len(bad)},  PROVALENO: {len(bad)}")
for n in bad:
    print(f"  PROVAL: {n}")
print("VERDIKT:", "VSE PROVERKI PROYDENY" if not bad else "EST PROVALY -- REZULTATY NE VALIDNY")
