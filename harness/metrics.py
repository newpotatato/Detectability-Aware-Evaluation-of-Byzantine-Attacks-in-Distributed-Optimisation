"""
Metriki vreda (shared/METRICS.md).

L    -- dolya vychislitelnogo progressa, otnyataya atakoy (§6)
sat  -- dolya progonov, obnulennyh polnostyu (L >= 1)
help -- dolya progonov, gde ataka uluchshila ishod
d    -- logarifm otnosheniya zazorov do optimuma (§5), trebuet f*
eps50-- naimenshiy byudzhet, pri kotorom mediana L dostigaet tselevogo urovnya (§8a)
"""
from __future__ import annotations
import numpy as np

HELP = "help"          # A < H_T: ataka uluchshila, L ne opredelena


def running_min(v: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(v)


def L_single(f_honest: np.ndarray, f_attacked: np.ndarray):
    """
    Poterya vychislitelnogo progressa dlya odnogo sida.

    OBE krivye beryutsya v vide monotonnoy ogibayushchey (luchshee na tekushchiy
    moment).

    ISPRAVLENO 2026-08-28. Ranshe ogibayushchaya bralas TOLKO u chestnoy krivoy,
    a u atakovannoy bralos syroe konechnoe znachenie. Konsensus roya kolebletsya,
    poetomu ego posledneye znachenie VSEGDA vyshe sobstvennogo minimuma, i eta
    fluktuatsiya zaschityvalas kak vred: sravnenie chestnogo progona S SAMIM
    SOBOY davalo L = 0.91 pri nulevom vrede. Smeshchenie vshestero prevoshodilo
    nastoyashchiy signal i portilo ne tolko urovni, no i RANGI atak (poryadok
    menyalsya na protivopolozhnyy posle ispravleniya).

    Opasenie, iz-za kotorogo asimmetriya byla vvedena, -- chto ogibayushchaya
    skroet uvod roya vverh -- zakryvaetsya otdelnym pokazatelem uphill_single,
    a ne narusheniem simmetrii.
    """
    H = running_min(f_honest)
    H0, HT = float(H[0]), float(H[-1])
    A = float(running_min(f_attacked)[-1])
    T = len(f_honest)

    if A < HT:
        return HELP
    if A > H0:
        span = H0 - HT
        return 1.0 + (A - H0) / span if span > 1e-15 else 1.0
    idx = np.nonzero(H <= A)[0]
    t_h = int(idx[0]) if len(idx) else 0
    return (T - t_h) / T


def d_single(f_honest, f_attacked, f_opt):
    """
    Logarifm otnosheniya zazorov. None, esli zazor ne polozhitelen.
    ISPRAVLENO 2026-08-28: ogibayushchaya u OBOIH krivyh (sm. L_single).
    """
    gh = float(running_min(f_honest)[-1]) - f_opt
    ga = float(running_min(f_attacked)[-1]) - f_opt
    if gh <= 1e-15 or ga <= 1e-15:
        return None
    return float(np.log(ga) - np.log(gh))


def L_baseline(f_honest):
    """
    Sobstvennyy POL velichiny L: znachenie, kotoroe ona prinimaet pri POLNOM
    otsutstvii vreda (atakovannaya krivaya sovpadaet s chestnoy).

    Otkuda beretsya. L = (T - t_h)/T, gde t_h -- moment, kogda ogibayushchaya
    chestnogo progona vpervye dostigla urovnya A. Esli roy k kontsu vyshel na
    plato, ego ogibayushchaya dostigaet finalnogo urovnya RANSHE T, i dazhe
    tochnoe sovpadenie krivyh daet L > 0.

    Izmereno 2026-08-28: na MNIST pri T=150 pol raven 0.031 +- 0.012, na sfere
    pri T=200 -- 0.379 +- 0.223 (roy vyhodit na shumovoe plato k 38% progona).

    Velichina L_adjusted = L - L_baseline ravna nulyu pri nulevom vrede.
    Na RANGI vychet ne vliyaet: vnutri yacheyki chestnyy progon obshchiy dlya
    vseh atak. Odnako bez vycheta slabye ataki prizhimayutsya k polu i stanovyatsya
    NERAZLICHIMY -- effekt konservativnyy (rastet dolya neyasnyh par), no
    absolyutnye znacheniya L bez vycheta zavysheny.
    """
    v = L_single(f_honest, f_honest)
    return 0.0 if v is HELP else float(v)


def L_adjusted(f_honest, f_attacked):
    """L s vychetom sobstvennogo pola chestnogo progona. HELP prohodit naskvoz."""
    v = L_single(f_honest, f_attacked)
    if v is HELP:
        return HELP
    return float(v) - L_baseline(f_honest)


def uphill_single(f_honest, f_attacked):
    """
    Diagnostika: naskolko roy zakonchil VYSHE sobstvennogo luchshego, v edinitsah
    puti chestnogo progona.

    Ogibayushchaya po postroeniyu ne ukhudshaetsya, poetomu sluchay "atakuyushchiy
    uvel roy vverh posle togo, kak tot uzhe nashel horoshuyu tochku" ona ne vidit.
    Zdes on izmeryaetsya yavno. Velichina 0 znachit, chto roy zakonchil v svoem
    zhe luchshem sostoyanii; bolshie znacheniya -- chto ego ottashchili naverh.
    """
    H = running_min(f_honest)
    span = float(H[0]) - float(H[-1])
    if span <= 1e-15:
        return np.nan
    return (float(f_attacked[-1]) - float(running_min(f_attacked)[-1])) / span


def summarize(pairs, f_opt=None):
    """
    pairs -- posledovatelnost (f_honest, f_attacked) po sidam.
    Vozvrashchaet slovar so vsemi velichinami §4-§7.
    """
    raw = [L_single(h, a) for h, a in pairs]
    n = len(raw)
    n_help = sum(1 for v in raw if v is HELP)
    Ls = np.array([v for v in raw if v is not HELP], float)

    out = {
        "n_seeds": n,
        "help": n_help / n if n else np.nan,
        "sat": float(np.mean(Ls >= 1.0)) if len(Ls) else np.nan,
        "L_med": float(np.median(Ls)) if len(Ls) else np.nan,
        "L_q10": float(np.quantile(Ls, 0.10)) if len(Ls) else np.nan,
        "L_q90": float(np.quantile(Ls, 0.90)) if len(Ls) else np.nan,
    }
    up = np.array([uphill_single(h, a) for h, a in pairs], float)
    up = up[np.isfinite(up)]
    out["uphill_med"] = float(np.median(up)) if len(up) else np.nan

    if f_opt is not None:
        ds = [d_single(h, a, f_opt) for h, a in pairs]
        ds = np.array([v for v in ds if v is not None], float)
        out["d_med"] = float(np.median(ds)) if len(ds) else np.nan
        out["d_q90"] = float(np.quantile(ds, 0.90)) if len(ds) else np.nan
    return out


def eps_at_target(eps_grid, L_med_by_eps, q=0.5):
    """
    eps_q (§8a): naimenshiy byudzhet, pri kotorom mediana L dostigaet q.

    Vozvrashchaet (eps_q, eps_bar):
      eps_q   -- None, esli tsel ne dostignuta ni pri odnom byudzhete (TSENZURIROVANO;
                 podstavlyat max setki NELZYA -- eto zanizit stoykost agregatora)
      eps_bar -- naibolshiy byudzhet, pri kotorom vred eshche >= q. U SPADAYUSHCHIH
                 agregatorov vred nemonotonen po gromkosti, i verhnyaya granitsa
                 poleznoy gromkosti sama po sebe rezultat.
    """
    eps_grid = np.asarray(eps_grid, float)
    v = np.asarray(L_med_by_eps, float)
    ok = np.nonzero(v >= q)[0]
    if len(ok) == 0:
        return None, None
    return float(eps_grid[ok[0]]), float(eps_grid[ok[-1]])


if __name__ == "__main__":
    T = 100
    # chestnaya krivaya: geometricheskaya shodimost 100 -> 0.1
    H = 100 * np.exp(np.linspace(0, np.log(1e-3), T))

    print("Proverka L na sinteticheskih sluchayah\n")
    cases = [
        ("ataka nichego ne otnyala   ", H.copy()),
        ("konchil tam, gde chestnyy v seredine", np.full(T, H[T // 2])),
        ("konchil na urovne starta   ", np.full(T, H[0])),
        ("otbroshen za start vdvoe   ", np.full(T, H[0] + (H[0] - H[-1]))),
        ("ataka POMOGLA              ", H.copy() * 0.5),
    ]
    for label, A in cases:
        v = L_single(H, A)
        print(f"  {label}  L = {v if v is HELP else f'{v:.3f}'}")

    print("\nProverka svodki po 5 sidam")
    pairs = [(H, A) for _, A in cases]
    for k, val in summarize(pairs, f_opt=0.0).items():
        print(f"  {k:<8} = {val:.3f}" if isinstance(val, float) else f"  {k:<8} = {val}")

    print("\nProverka eps50 na nemonotonnoy krivoy (spadayushchiy agregator)")
    grid = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    Lmed = [0.05, 0.20, 0.62, 0.71, 0.10, 0.02]     # pik vnutri, dalshe obval
    e50, ebar = eps_at_target(grid, Lmed, 0.5)
    print(f"  eps50 = {e50}   eps_bar = {ebar}   (poleznyy diapazon gromkosti)")
