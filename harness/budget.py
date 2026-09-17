"""
Byudzhet nezametnosti (shared/BUDGET.md).

eps = (1/2T) * sum_t ( ||z_t - mu_t||^2_{Sigma_t^-1} - d )

mu_t, Sigma_t schitayutsya po CHESTNYM otpravkam raunda -- eto ORAKULNYE
velichiny, dostupnye eksperimentatoru, no NE agentu. Proektsiya na shar --
vneshnee ogranichenie sredy, a ne chast politiki (BUDGET.md §5).
"""
from __future__ import annotations
import numpy as np


def honest_moments(Y: np.ndarray, shrink: float | None = None):
    """
    Srednee i kovariatsiya chestnyh otpravok so szhatiem k diagonali.

    Pri n ~ 20 i d ~ 10 vyborochnaya kovariatsiya ploho obuslovlena, poetomu
    szhatie obyazatelno (BUDGET.md §7.2). Pri shrink=None koeffitsient
    vybiraetsya avtomaticheski iz otnosheniya d/n.
    """
    n, d = Y.shape
    mu = Y.mean(axis=0)
    S = np.cov(Y, rowvar=False, bias=False) if n > 1 else np.eye(d)
    S = np.atleast_2d(S)
    if shrink is None:
        shrink = float(np.clip(d / max(n - 1, 1), 0.05, 0.9))
    target = (np.trace(S) / d) * np.eye(d)
    S = (1.0 - shrink) * S + shrink * target
    S += 1e-12 * np.eye(d)
    return mu, S


def mahalanobis_sq(z, mu, S_inv):
    v = z - mu
    return float(v @ S_inv @ v)


def effective_d(Y, shrink: float | None = None) -> float:
    """
    Effektivnaya razmernost: srednee maha^2 CHESTNOY otpravki pri NASHIH
    otsenkah mu, Sigma, poschitannoe s isklyucheniem samoy tochki.

    Zachem. Teoriya daet E[maha^2] = d pri IZVESTNYH mu, Sigma. My zhe
    otsenivaem ih po n ~ 20 tochkam, i obratnaya vyborochnaya kovariatsiya
    smeshchena vverh mnozhitelem poryadka n/(n-d-2): pri n=19, d=10 eto ~2.7.
    Podstanovka d v kachestve nulya dala by chestnomu agentu byudzhet ~1.5
    vmesto 0 i zavysila by izmerennuyu gromkost vsem.

    Reshenie -- kalibrovat nol empiricheski po samim chestnym, s ischerpyvayushchim
    ucheto smeshcheniya lyuboy prirody. Eto ne evristika: velichina imeet tochnyy
    smysl "skolko maha^2 nabiraet TIPICHNYY chestnyy uchastnik".

    VAZHNO: raspredelenie maha^2 pri otsenennoy kovariatsii tyazhelohvostoe
    (otnoshenie kvadratichnyh form), poetomu srednee po odnomu raundu iz ~20
    tochek beznadezhno shumno. Y dolzhen byt SPISKOM massivov po mnogim raundam
    -- kalibrovochnyy progon, posle chego d_eff FIKSIRUETSYA na vsyu seriyu,
    kak i f_ref v METRICS.md.

    Y -- massiv (n, d) libo posledovatelnost takih massivov.
    """
    rounds = [Y] if isinstance(Y, np.ndarray) and Y.ndim == 2 else list(Y)
    vals = []
    for Yr in rounds:
        n = len(Yr)
        if n < 3:
            continue
        for i in range(n):
            mu, S = honest_moments(np.delete(Yr, i, axis=0), shrink)
            vals.append(mahalanobis_sq(Yr[i], mu, np.linalg.inv(S)))
    if not vals:
        return float(np.atleast_2d(rounds[0]).shape[1])
    return float(np.mean(vals))


def round_cost(z, mu, S, d_eff=None):
    """
    Vklad odnogo raunda v byudzhet: (maha^2 - d_eff) / 2.

    d_eff -- effektivnaya razmernost (sm. effective_d). Esli ne zadana,
    beretsya nominalnaya d, chto korrektno lish pri izvestnyh mu, Sigma.
    Mozhet byt otritsatelnym: dlya nesmeshchennoy otsenki eto normalno.
    """
    d_eff = d_eff if d_eff is not None else len(mu)
    return 0.5 * (mahalanobis_sq(z, mu, np.linalg.inv(S)) - d_eff)


def project(z, mu, S, eps_budget, two_sided=True, d_eff=None):
    """
    Privedenie otpravki k zadannomu byudzhetu.

    two_sided=True (po umolchaniyu) -- RAVNENIE: otpravka masshtabiruetsya na
    POVERHNOST shara, i vverh, i vniz. Bez etogo sravnenie ne yavlyaetsya
    sravneniem pri ravnom byudzhete: ataka, ot prirody bolee tihaya, chem
    tselevoy uroven, tak i ostaetsya tihoy. Naprimer ALIE po konstruktsii
    derzhitsya vnutri oblaka chestnyh i imeet eps ~ -4.8; pri odnostoronney
    proektsii ona sravnivalas by na -4.8 protiv signflip na +0.3, chto delaet
    ves protokol bessmyslennym (obnaruzheno 2026-08-28 na progone exp4).

    two_sided=False -- staroe povedenie, tolko potolok. Ostavleno dlya sverki.

    Vozvrashchaet (z_proj, bylo_li_izmeneno).
    """
    d = float(d_eff) if d_eff is not None else float(len(mu))
    S_inv = np.linalg.inv(S)
    m2 = mahalanobis_sq(z, mu, S_inv)
    R2 = d + 2.0 * eps_budget
    if R2 <= 0:
        return mu.copy(), True
    if m2 <= 1e-18:                       # otpravka sovpala s tsentrom oblaka
        return z, False
    if not two_sided and m2 <= R2:
        return z, False
    if abs(m2 - R2) < 1e-12:
        return z, False
    return mu + (z - mu) * np.sqrt(R2 / m2), True


def calibrate_grid(scale_hint=1.0, n=9):
    """
    Logarifmicheskaya setka byudzhetov ot pochti-nezametnogo do uverenno
    otsekaemogo. scale_hint podbiraetsya kalibrovochnym progonom: nuzhno,
    chtoby verhniy konets uzhe uverenno obrezalsya (BUDGET.md §7.5).
    """
    return scale_hint * np.logspace(-2, 2, n)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 19, 10
    Y = rng.normal(0, 1, (n, d))
    mu, S = honest_moments(Y)
    S_inv = np.linalg.inv(S)

    print(f"n={n}, d={d}, cond(Sigma) = {np.linalg.cond(S):.1f}\n")
    print("Skhodimost d_eff po chislu kalibrovochnyh raundov:")
    for R in [1, 10, 100, 1000, 4000]:
        rounds = [rng.normal(0, 1, (n, d)) for _ in range(R)]
        print(f"  R={R:>5}:  d_eff = {effective_d(rounds):.3f}")

    rounds = [rng.normal(0, 1, (n, d)) for _ in range(4000)]
    d_eff = effective_d(rounds)
    print(f"\nprinyato d_eff = {d_eff:.3f}  (nominalnaya d = {d}, smeshchenie x{d_eff/d:.2f})\n")

    print("Proverka: u CHESTNOY otpravki srednyaya stoimost raunda ~ 0")
    naive, cal = [], []
    for _ in range(4000):
        Yr = rng.normal(0, 1, (n, d))
        m_, S_ = honest_moments(Yr[:-1])
        naive.append(round_cost(Yr[-1], m_, S_))
        cal.append(round_cost(Yr[-1], m_, S_, d_eff))
    print(f"  s nominalnoy d : srednee = {np.mean(naive):+.3f}  <- SMESHCHENO")
    print(f"  s d_eff        : srednee = {np.mean(cal):+.3f}  <- OK\n")

    print("Stoimost otpravki, smeshchennoy na k sigm po odnoy osi:")
    print(f"{'sdvig':>7}{'maha^2':>10}{'eps za raund':>14}")
    u = np.zeros(d); u[0] = 1.0
    for k in [0, 1, 2, 3, 5, 10]:
        z = mu + k * np.sqrt(S[0, 0]) * u
        print(f"{k:>7}{mahalanobis_sq(z, mu, S_inv):>10.2f}"
              f"{round_cost(z, mu, S, d_eff):>14.2f}")

    print("\nProverka proektsii pri eps=1.0 (radius^2 = d + 2 = 12):")
    for k in [1, 3, 10]:
        z = mu + k * np.sqrt(S[0, 0]) * u
        zp, clipped = project(z, mu, S, 1.0)
        print(f"  sdvig {k:>2}: maha^2 {mahalanobis_sq(z, mu, S_inv):>7.2f}"
              f" -> {mahalanobis_sq(zp, mu, S_inv):>6.2f}   obrezan={clipped}")
