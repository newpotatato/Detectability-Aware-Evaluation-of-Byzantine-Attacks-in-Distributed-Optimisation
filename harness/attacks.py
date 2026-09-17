"""
Ataki. Kazhdaya pomechena UROVNEM informirovannosti (RELATED_WORK §6):

  0a -- nichego ne nablyudaet, bez obucheniya
  0b -- nichego ne nablyudaet, adaptiruetsya po otkliku agregata
  1  -- vidit statistiki CHESTNYH otpravok
  2  -- vidit chestnye otpravki I ekzemplyar agregatora

Uroven ne dekorativen: swarm.run peredaet orakulnye dannye v ctx.oracle TOLKO
pri level >= 1. Eto strukturnaya zashchita ot utechki informatsii v politiku,
o kotoroy predupredazhdaet METRICS.md §8.1.

Otpravlyaetsya TOCHKA (predlagaemoe sleduyushchee polozhenie), sm. swarm.py.
"""
from __future__ import annotations
import numpy as np
from .swarm import Attack, Ctx


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def _uphill(ctx: Ctx):
    """Napravlenie vreda: protiv sobstvennogo gradienta (variant IPM, POLICY §2.1)."""
    return _unit(ctx.g_own)


def _scale(ctx: Ctx) -> float:
    """
    Lokalnyy masshtab roya: norma sobstvennogo CHESTNOGO shaga agenta.

    Zachem. Esli amplitudu zadavat ABSOLYUTNO, to po mere shodimosti sobstvennyy
    masshtab roya padaet (na sfere -- na chetyre poryadka), a kappa ostaetsya
    prezhney, i OTNOSITELNAYA gromkost ataki neogranichenno rastet. Togda vse
    kappa okazyvayutsya ekvivalentny, vred nasyshchaetsya, i vybor amplitudy
    perestaet chto-libo znachit -- diagnostika 2026-08-28 pokazala rovno eto.

    Poetomu z = c_prev + kappa * s_t * u, gde s_t -- lokalno NABLYUDAEMYY masshtab.
    Velichina dostupna vreditelyu: eto ego sobstvennyy shag, a ne statistika roya.
    """
    s = float(np.linalg.norm(ctx.honest_proposal - ctx.x_own))
    return s if s > 1e-12 else 1.0


# ============================================================ uroven 0a


class Honest(Attack):
    name, level = "honest", "0a"

    def propose(self, ctx):
        return ctx.honest_proposal


class SignFlip(Attack):
    """Shag vverh vmesto vniz, s postoyannoy amplitudoy."""
    name, level = "signflip", "0a"

    def __init__(self, kappa=1.0):
        self.kappa = kappa

    def propose(self, ctx):
        base = ctx.c_prev if ctx.c_prev is not None else ctx.x_own
        return base + self.kappa * _scale(ctx) * _uphill(ctx)


class GaussianNoise(Attack):
    """Kontrol: sluchaynoe napravlenie toy zhe amplitudy. Otlichaet umysel ot dispersii."""
    name, level = "gaussian", "0a"

    def __init__(self, kappa=1.0):
        self.kappa = kappa

    def reset(self, rng, cfg):
        self.rng = rng

    def propose(self, ctx):
        base = ctx.c_prev if ctx.c_prev is not None else ctx.x_own
        return base + self.kappa * _scale(ctx) * _unit(self.rng.normal(size=len(ctx.x_own)))


class FixedAmplitude(Attack):
    """Postoyannaya amplituda vdol napravleniya vreda. Bazovyy uroven dlya sozhaleniya."""
    name, level = "fixed", "0a"

    def __init__(self, kappa=1.0):
        self.kappa = kappa

    def propose(self, ctx):
        base = ctx.c_prev if ctx.c_prev is not None else ctx.x_own
        return base + self.kappa * _scale(ctx) * _uphill(ctx)


class RandomAmplitude(Attack):
    """Amplituda vybiraetsya sluchayno kazhdyy raund. Nizhnyaya granitsa obucheniya."""
    name, level = "random", "0a"

    def __init__(self, grid):
        self.grid = np.asarray(grid, float)

    def reset(self, rng, cfg):
        self.rng = rng

    def propose(self, ctx):
        base = ctx.c_prev if ctx.c_prev is not None else ctx.x_own
        return base + float(self.rng.choice(self.grid)) * _scale(ctx) * _uphill(ctx)


# ============================================================ uroven 0b


class _AmplitudeAttack(Attack):
    """Obshchaya chast: napravlenie fiksirovano, adaptiruetsya tolko amplituda."""

    def reset(self, rng, cfg):
        self.rng = rng
        self.kappa = float(self.k0)
        self.prev_kappa = None
        self.prev_c = None
        self.u = None

    def propose(self, ctx):
        self.u = _uphill(ctx)
        self.base = ctx.c_prev if ctx.c_prev is not None else ctx.x_own
        self.s = _scale(ctx)
        self.kappa = float(np.clip(self._next_kappa(), self.kmin, self.kmax))
        return self.base + self.kappa * self.s * self.u

    #: DIAGNOSTIKA. True -> vmesto zashumlennogo proksi ispolzuetsya ISTINNOE
    #: vliyanie, izmerennoe sredoy. Real'nomu vreditelyu nedostupno; sluzhit
    #: POVERKOY PRIBORA: otlichaet "gipoteza neverna" ot "poisk ploho napisan".
    oracle_influence = False

    def observe(self, c_new, z_sent, oracle_infl=None):
        if self.oracle_influence and oracle_infl is not None:
            self._update(float(oracle_infl))
        elif self.prev_c is not None and self.u is not None:
            # Nablyudaemyy proksi: smeshchenie konsensusa vdol napravleniya ataki.
            # Imenno etu velichinu FedSA otsenivaet konechnymi raznostyami.
            self._update(float((c_new - self.prev_c) @ self.u))
        self.prev_c = c_new.copy()
        self.prev_kappa = self.kappa

    def _next_kappa(self):
        raise NotImplementedError

    def _update(self, infl):
        pass


class FirstOrderAmplitude(_AmplitudeAttack):
    """
    Zakon FedSA v nashey postanovke: proizvodnaya otklika agregatora otsenivaetsya
    KONECHNYMI RAZNOSTYAMI po sobstvennomu vozmushcheniyu, amplituda dvigaetsya
    vverh po etoy proizvodnoy.

    Eto DOBROSOVESTNAYA adaptatsiya ih mehanizma pod nashu tsel (maksimum vreda
    vmesto ustavki), sm. RELATED_WORK §7. Ozhidaemyy otkaz -- v zonah, gde
    proizvodnaya neinformativna: obval Krum v nol, razryv Talwar, shum roya.
    """
    name, level = "first_order", "0b"

    def __init__(self, k0=1.0, step=0.6, probe=0.15, kmin=0.05, kmax=50.0):
        self.k0, self.step, self.probe = k0, step, probe
        self.kmin, self.kmax = kmin, kmax

    def reset(self, rng, cfg):
        super().reset(rng, cfg)
        self.prev_infl = None
        self.pending = self.k0

    def _next_kappa(self):
        # dither, chtoby konechnaya raznost byla opredelena
        return self.pending * (1.0 + self.probe * (1 if self.rng.random() < 0.5 else -1))

    def _update(self, infl):
        if self.prev_infl is not None and self.prev_kappa is not None:
            dk = self.kappa - self.prev_kappa
            if abs(dk) > 1e-9:
                g = (infl - self.prev_infl) / dk          # d(vliyanie)/d(amplituda)
                self.pending = float(np.clip(
                    self.pending + self.step * np.sign(g) * max(abs(g), 1e-3),
                    self.kmin, self.kmax))
        self.prev_infl = infl


class GlobalSearchAmplitude(_AmplitudeAttack):
    """
    Nash poisk: amplituda ne dvigaetsya lokalno, a OTSENIVAETSYA po vsey setke.
    Kazhdaya amplituda probuetsya, nakaplivaetsya srednee nablyudaemoe vliyanie,
    vybor -- argmax s ubyvayushchey doley issledovaniya.

    Klyuchevoe otlichie ot pervogo poryadka: v zone nulevogo vliyaniya (za obvalom
    Krum) gradienta net vovse, i lokalnyy zakon zastrevaet; global'nyy poisk
    prodolzhaet probovat drugie amplitudy i nahodit pik.
    """
    name, level = "global_search", "0b"

    def __init__(self, grid=None, k0=1.0, explore=0.3, kmin=0.05, kmax=50.0):
        self.grid = np.asarray(grid if grid is not None
                               else np.logspace(-1, 1.5, 12), float)
        self.k0, self.explore = k0, explore
        self.kmin, self.kmax = kmin, kmax

    def reset(self, rng, cfg):
        super().reset(rng, cfg)
        self.sum = np.zeros(len(self.grid))
        self.cnt = np.zeros(len(self.grid))
        self.idx = int(np.argmin(np.abs(self.grid - self.k0)))
        self.t = 0

    def _next_kappa(self):
        self.t += 1
        untried = np.nonzero(self.cnt == 0)[0]
        if len(untried):                                   # snachala obyti vsyu setku
            self.idx = int(untried[0])
        elif self.rng.random() < self.explore / np.sqrt(self.t):
            self.idx = int(self.rng.integers(len(self.grid)))
        else:
            self.idx = int(np.argmax(self.sum / np.maximum(self.cnt, 1)))
        return float(self.grid[self.idx])

    def _update(self, infl):
        self.sum[self.idx] += infl
        self.cnt[self.idx] += 1


# ============================================================ uroven 1


class ALIE(Attack):
    """
    z_max iz chisla klientov, sdvig mu - z_max * sigma pokoordinatno.
    Trebuet srednee i s.k.o. CHESTNYH (v originale otsenivayutsya po svoey zhe
    gruppe pri IID, chto pri m=1 nevozmozhno -- sm. RELATED_WORK §0.1).
    """
    name, level = "alie", "1"

    def __init__(self, z=None):
        self.z_fixed = z

    def reset(self, rng, cfg):
        n, f = cfg.n_agents, cfg.n_malicious
        if self.z_fixed is not None:
            self.z = self.z_fixed
        else:
            from scipy.stats import norm
            s = n - int(np.floor(n / 2 + 1))
            self.z = float(norm.ppf(max(min((n - s) / max(n - f, 1), 1 - 1e-9), 1e-9)))

    def propose(self, ctx):
        Y = ctx.oracle["honest"]
        return Y.mean(axis=0) - self.z * Y.std(axis=0)


class IPM(Attack):
    """Vektor protiv srednego chestnyh predlozheniy, masshtab epsilon."""
    name, level = "ipm", "1"

    def __init__(self, eps=1.0):
        self.eps = eps

    def propose(self, ctx):
        Y = ctx.oracle["honest"]
        mu = Y.mean(axis=0)
        return mu - self.eps * (mu - ctx.x_own)


class MinMax(Attack):
    """
    lambda reshaetsya iz usloviya: maks. rasstoyanie ot vredonosnoy otpravki do
    lyuboy drugoy ne prevoshodit maks. rasstoyaniya mezhdu dvumya chestnymi.
    """
    name, level = "minmax", "1"

    def __init__(self, iters=30):
        self.iters = iters

    def propose(self, ctx):
        Y = ctx.oracle["honest"]
        mu = Y.mean(axis=0)
        p = -_unit(Y.std(axis=0))
        # Rasstoyaniya v dvoichnom poiske schitayutsya cherez razlozhenie
        #   ||(Y_i - mu) - lam*p||^2 = a_i - 2*lam*b_i + lam^2*c,
        # gde a, b, c predvychislyayutsya odin raz. Bez etogo kazhdaya iteratsiya
        # stoit O(n*d), chto pri d ~ 25000 (real'naya model) delaet ataku v 13 raz
        # medlennee prochih i blokiruet setku.
        Z = Y - mu
        a = np.einsum("ij,ij->i", Z, Z)
        b = Z @ p
        c = float(p @ p)
        D2 = a[:, None] + a[None, :] - 2.0 * (Z @ Z.T)
        thr2 = float(np.maximum(D2, 0).max())
        lo, hi = 0.0, 100.0
        for _ in range(self.iters):
            mid = 0.5 * (lo + hi)
            if float(np.max(a - 2.0 * mid * b + mid * mid * c)) <= thr2:
                lo = mid
            else:
                hi = mid
        return mu + lo * p


class MinSum(MinMax):
    """To zhe, no ogranichivaetsya SUMMA kvadratov rasstoyaniy."""
    name, level = "minsum", "1"

    def propose(self, ctx):
        Y = ctx.oracle["honest"]
        mu = Y.mean(axis=0)
        p = -_unit(Y.std(axis=0))
        Z = Y - mu
        a = np.einsum("ij,ij->i", Z, Z)
        b = Z @ p
        c = float(p @ p)
        n = len(Y)
        D2 = a[:, None] + a[None, :] - 2.0 * (Z @ Z.T)
        thr = float(np.maximum(D2, 0).sum(axis=1).max())
        lo, hi = 0.0, 100.0
        for _ in range(self.iters):
            mid = 0.5 * (lo + hi)
            if float(np.sum(a) - 2.0 * mid * float(np.sum(b)) + n * mid * mid * c) <= thr:
                lo = mid
            else:
                hi = mid
        return mu + lo * p


REGISTRY = {c.name: c for c in [Honest, SignFlip, GaussianNoise, FixedAmplitude,
                                RandomAmplitude, FirstOrderAmplitude,
                                GlobalSearchAmplitude, ALIE, IPM, MinMax, MinSum]}
LEVEL0 = ["fixed", "random", "gaussian", "signflip", "first_order", "global_search"]
LEVEL1 = ["alie", "ipm", "minmax", "minsum"]
