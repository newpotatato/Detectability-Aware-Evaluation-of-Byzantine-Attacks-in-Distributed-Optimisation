"""
Tsikl roya s PARNYM kontrfaktichnym progonom (METRICS.md §2).

Protokol raunda:
  1. agent i v tochke x_i predlagaet sleduyushchuyu tochku
         u_i = x_i - lr * grad f(x_i) + shum_i
  2. agregator schitaet c = AGG({u_j})
  3. agent chastichno podtyagivaetsya k konsensusu
         x_i <- (1 - eta) * u_i + eta * c

Otpravlyaetsya TOCHKA, a ne shag: v konsensusnoy optimizatsii sostoyanie roya --
eto gde on seychas stoit. Polnoe smeshivanie (eta = 1) zapreshcheno: pri nem vse
agenty shlopyvayutsya v odnu tochku i ponyatie sobstvennoy traektorii ischezaet
(RESEARCH_PLAN §2.1).

KLYUCHEVOE dlya parnosti: shum chestnyh agentov PREDGENERIRUETSYA iz sida, poetomu
v chestnom i atakovannom progone on POBITNO odinakov. Vse rashozhdenie traektoriy
prichinno atribuiruetsya vreditelyu.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from . import aggregators as AGG
from . import budget as BUD


@dataclass
class Config:
    n_agents: int = 20
    n_malicious: int = 1
    T: int = 300
    lr: float = 0.02
    eta: float = 0.3               # dolya podtyagivaniya k konsensusu, < 1
    grad_noise: float = 0.15       # sigma stohasticheskogo gradienta
    eps_budget: float = 1.0        # byudzhet nezametnosti; None = bez ogranicheniya
    d_eff: float | None = None     # effektivnaya razmernost, sm. budget.effective_d
    two_sided: bool = True         # URAVNIVAT byudzhet (i vverh, i vniz), a ne tolko obrezat
    agg_kwargs: dict = field(default_factory=dict)


@dataclass
class Ctx:
    """
    Chto vreditel vidit na raunde t.

    Pole oracle zapolnyaetsya TOLKO dlya atak urovnya >= 1 (attacks.py). Dlya
    urovnya 0 ono ravno None, poetomu popytka podglyadet za chestnymi privedet
    k oshibke, a ne k tihoy utechke. Eto strukturnaya realizatsiya razdeleniya
    metrik eksperimentatora i agenta (METRICS.md §8.1).
    """
    t: int
    x_own: np.ndarray
    g_own: np.ndarray
    honest_proposal: np.ndarray    # chto on otpravil by, bud chestnym
    c_prev: np.ndarray | None      # agregat proshlogo raunda
    rho_prev: float | None         # sila otkata proshlogo raunda
    oracle: dict | None = None     # {"honest": (n_h, d)} -- tolko pri level >= 1


class Attack:
    """Bazovyy interfeys. propose vozvrashchaet otpravku DO proektsii na byudzhet."""
    name = "none"
    level = "0a"                   # "0a" | "0b" | "1" | "2"

    def reset(self, rng, cfg):
        pass

    def propose(self, ctx: Ctx) -> np.ndarray:
        return ctx.honest_proposal

    def observe(self, c_new: np.ndarray, z_sent: np.ndarray, oracle_infl=None):
        """oracle_infl -- ISTINNOE vliyanie ||AGG(U) - AGG(U bez vreditelya)||.
        Real'nomu vreditelyu NEDOSTUPNO; peredaetsya tolko dlya poverki pribora."""
        pass


@dataclass
class Trace:
    f_consensus: np.ndarray        # f(c_t), t = 0..T-1
    f_mean_pos: np.ndarray         # f v srednem polozhenii roya
    eps_spent: np.ndarray          # stoimost raunda dlya vreditelya (NaN v chestnom)
    clipped: np.ndarray            # byla li otpravka obrezana proektsiey
    rho: np.ndarray                # sila otkata ||c - z|| / ||z - c_prev||
    honest_rounds: list            # otpravki chestnyh po raundam (dlya kalibrovki)


def _pregen_noise(cfg: Config, d: int, seed: int) -> np.ndarray:
    """Shum (T, n, d), odinakovyy v oboih progonah pary."""
    return np.random.default_rng(10_000 + seed).normal(
        0.0, cfg.grad_noise, (cfg.T, cfg.n_agents, d))


def run(landscape, agg_name: str, cfg: Config, seed: int,
        attack: Attack | None = None, collect_honest: bool = False) -> Trace:
    """
    Odin progon. attack=None -> chestnyy progon (kontrfaktual).
    Vreditelyami schitayutsya agenty s indeksami 0..n_malicious-1.
    """
    d = landscape.d
    agg = AGG.get(agg_name)
    rng = np.random.default_rng(seed)
    X = landscape.sample_start(cfg.n_agents, rng)
    noise = _pregen_noise(cfg, d, seed)

    n_mal = 0 if attack is None else cfg.n_malicious
    mal = list(range(n_mal))
    hon = list(range(n_mal, cfg.n_agents))

    if attack is not None:
        attack.reset(np.random.default_rng(20_000 + seed), cfg)

    f_c = np.empty(cfg.T); f_m = np.empty(cfg.T)
    eps = np.full(cfg.T, np.nan); clip = np.zeros(cfg.T, bool)
    rho = np.full(cfg.T, np.nan)
    honest_rounds = []
    c_prev, rho_prev = None, None

    for t in range(cfg.T):
        # --- chestnye predlozheniya vseh agentov
        U = np.empty((cfg.n_agents, d))
        for i in range(cfg.n_agents):
            U[i] = X[i] - cfg.lr * landscape.grad(X[i]) + noise[t, i]

        # --- podmena otpravok vreditelyami
        if attack is not None:
            Yh = U[hon]                                  # orakulnye, tolko dlya byudzheta
            mu, S = BUD.honest_moments(Yh)
            # orakul otdaetsya atake TOLKO esli ee uroven eto pozvolyaet
            orc = {"honest": Yh.copy()} if str(attack.level) >= "1" else None
            for i in mal:
                ctx = Ctx(t=t, x_own=X[i], g_own=landscape.grad(X[i]),
                          honest_proposal=U[i], c_prev=c_prev, rho_prev=rho_prev,
                          oracle=orc)
                z = np.asarray(attack.propose(ctx), float)
                if cfg.eps_budget is not None:
                    # dvuhstoronnee URAVNIVANIE i tot zhe d_eff, chto v round_cost:
                    # inache radius shara stroitsya po nominalnoy d, a stoimost --
                    # po kalibrovannoy, i tselevoy uroven okazyvaetsya ne tem
                    z, was_clipped = BUD.project(z, mu, S, cfg.eps_budget,
                                                 two_sided=cfg.two_sided,
                                                 d_eff=cfg.d_eff)
                    clip[t] |= was_clipped
                if i == 0:
                    eps[t] = BUD.round_cost(z, mu, S, cfg.d_eff)
                U[i] = z

        if collect_honest:
            honest_rounds.append(U[hon].copy())

        # --- agregatsiya i chastichnoe smeshivanie
        c = agg(U, **cfg.agg_kwargs)
        if attack is not None:
            z0 = U[0]
            denom = np.linalg.norm(z0 - c_prev) if c_prev is not None else np.linalg.norm(z0)
            rho[t] = np.linalg.norm(c - z0) / max(denom, 1e-12)
            # istinnoe vliyanie schitaem TOLKO esli ataka ego prosit (diagnostika):
            # lishnyaya agregatsiya na kazhdom raunde udvaivaet stoimost progona
            oi = None
            if getattr(attack, "oracle_influence", False):
                oi = float(np.linalg.norm(c - agg(U[hon], **cfg.agg_kwargs)))
            attack.observe(c, z0, oracle_infl=oi)
        rho_prev = rho[t] if attack is not None else None

        X = (1.0 - cfg.eta) * U + cfg.eta * c[None, :]
        f_c[t] = landscape(c)
        f_m[t] = landscape(X.mean(axis=0))
        c_prev = c

    return Trace(f_c, f_m, eps, clip, rho, honest_rounds)


def paired_run(landscape, agg_name, cfg, seed, attack):
    """Chestnyy i atakovannyy progony pri odinakovoy initsializatsii i shume."""
    return (run(landscape, agg_name, cfg, seed, attack=None),
            run(landscape, agg_name, cfg, seed, attack=attack))


def calibrate_d_eff(landscape, agg_name, cfg, seeds=(0, 1, 2), max_rounds=200):
    """Kalibrovochnyy progon: d_eff po chestnym otpravkam, FIKSIRUETSYA na seriyu."""
    rounds = []
    for s in seeds:
        tr = run(landscape, agg_name, cfg, s, attack=None, collect_honest=True)
        rounds.extend(tr.honest_rounds[:max_rounds])
    return BUD.effective_d(rounds)
