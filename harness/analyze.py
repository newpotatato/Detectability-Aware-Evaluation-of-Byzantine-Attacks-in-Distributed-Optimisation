"""
Analiz syryh dannyh progonov (results/*.csv).

Klyuchevaya statistika -- NE Spirmen. Rangovaya korrelyatsiya schitaetsya po
semi atakam, i ee standartnaya oshibka ~1/sqrt(6) = 0.41, to est otdelnoe
znachenie pochti neinformativno. Vmesto etogo -- DOLYA PEREVERNUVSHIHSYA PAR,
poolennaya po vsem blokam: 4 landshafta x 3 ekz x 6 agregatorov x C(7,2)=21
para = do 1512 par, chto na poryadok luchshe po moshchnosti.

Perevorot zaschityvaetsya TOLKO esli poryadok pary uveren V OBOIH rezhimah:
bootstrap-interval raznosti po sidam ne nakryvaet nol. Inache my by schitali
shum za rezultat.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

METRIC = "d"          # osnovnaya mera vreda; L vyrozhdena na obryvistyh krivyh


def load(path):
    df = pd.read_csv(path)
    df["eps_target"] = df["eps_target"].astype(str)
    return df


def _boot_ci(x, n=2000, q=(2.5, 97.5), rng=None):
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    bs = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(bs, q))


def add_harm(df, baseline, keys):
    """
    KANONICHESKIY L_adj -- edinstvennoe opredelenie na ves proekt.

        L_adj = max(0, L - L_baseline),
        prichem progony, gde ataka POMOGLA (helped == 1, L zapisan kak NaN),
        vhodyat kak L_adj = 0.

    ISPRAVLENO 2026-09-05. Do etogo velichina schitalas TREMYA raznymi sposobami:
    reversal_stats brala metriku napryamuyu, eps50_table stavila L=0 u helped no
    ne vychitala pol, a figures.load vychitala pol, obnulyala helped I obrezala
    snizu. Tablitsa 2 i risunok 2 v rukopisi rashodilis do chetyreh raz pri
    odinakovoy podpisi. Teper vse berut etu funktsiyu.

    Obnulenie -- ne kosmetika: ono cenzuriruet okolo 22% progonov na sinteticheskoy
    setke, i eto obyazano byt skazano v tekste raboty ryadom s opredeleniem.
    """
    d = df.merge(baseline, on=keys)
    la = (d["L"] - d["L_baseline"]).where(d["helped"] == 0, 0.0)
    d["L_adj"] = la.clip(lower=0).fillna(0.0)
    return d


def confident_order(a, b, rng, n=3000):
    """
    Uveren li poryadok mezhdu dvumya atakami po parnym (po sidam) nablyudeniyam.
    Vozvrashchaet +1 (a > b), -1 (a < b) libo 0 (ne razlichimy).
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return 0
    bs = rng.choice(d, (n, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    if lo > 0:
        return 1
    if hi < 0:
        return -1
    return 0


def _block_ci(cells, n=3000, q=(2.5, 97.5), rng=None):
    """
    BLOCHNYY bootstrap po YACHEYKAM (landshaft, ekzemplyar, agregator).

    ISPRAVLENO 2026-09-05. Ranshe interval schitalsya resemplingom PULA par, kak
    esli by pary byli nezavisimy. Oni ne nezavisimy: 1008 par poluchayutsya iz 48
    yacheek po 21 pare, pary vnutri yacheyki delyat te zhe 16 sidov i te zhe
    chestnye progony, a pary s obshchey atakoy korrelirovany cherez nee. Naivnyy
    bootstrap v takom sluchae ZANIZHAET shirinu intervala.

    cells -- spisok par (rev, tot) po yacheykam. Resempliruem yacheyki celikom.
    """
    rng = rng or np.random.default_rng(0)
    cells = [(r, t) for r, t in cells if t > 0]
    if not cells:
        return np.nan, np.nan
    arr = np.asarray(cells, float)
    idx = rng.integers(0, len(arr), (n, len(arr)))
    rev = arr[idx, 0].sum(axis=1)
    tot = arr[idx, 1].sum(axis=1)
    fr = np.divide(rev, tot, out=np.full_like(rev, np.nan), where=tot > 0)
    fr = fr[np.isfinite(fr)]
    if len(fr) < 2:
        return np.nan, np.nan
    return tuple(np.percentile(fr, q))


def _pair_counts(g_a, g_b, metric, rng):
    """
    Sravnivaet uporyadochennost vseh par atak v dvuh vyborkah odnoy yacheyki.
    Vozvrashchaet (perevernulos, uverenno_uporyadocheno, neyasno).
    """
    atks = sorted(set(g_a.attack) & set(g_b.attack))
    pa = {k: g_a[g_a.attack == k].sort_values("seed")[metric].values for k in atks}
    pb = {k: g_b[g_b.attack == k].sort_values("seed")[metric].values for k in atks}
    rev = tot = amb = 0
    for x, y in itertools.combinations(atks, 2):
        sa = confident_order(pa[x], pa[y], rng)
        sb = confident_order(pb[x], pb[y], rng)
        if sa == 0 or sb == 0:
            amb += 1
            continue
        tot += 1
        rev += int(sa != sb)
    return rev, tot, amb


CELL = ["landscape", "instance", "agg"]


def reversal_stats(df, metric=METRIC, seed=0):
    """Dolya par, poryadok kotoryh perevernulsya pri uravnivanii byudzheta."""
    rng = np.random.default_rng(seed)
    budgets = sorted([b for b in df["eps_target"].unique() if b != "native"],
                     key=float)
    out = []
    for b in budgets:
        cells, amb = [], 0
        for _, g in df.groupby(CELL):
            r, t, a = _pair_counts(g[g.eps_target == "native"],
                                   g[g.eps_target == b], metric, rng)
            cells.append((r, t)); amb += a
        rev = sum(r for r, _ in cells); tot = sum(t for _, t in cells)
        lo, hi = _block_ci(cells, rng=rng)
        out.append(dict(budget=float(b), n_cells=len(cells), n_pairs=tot,
                        n_ambiguous=amb, reversed=rev,
                        frac=rev / tot if tot else np.nan, lo=lo, hi=hi))
    return pd.DataFrame(out)


def null_calibration(df, metric=METRIC, seed=0):
    """
    NULEVAYA KALIBROVKA: ta zhe protsedura na NEPERESEKAYUSHCHIHSYA polovinah
    sidov PRI NEIZMENNOM byudzhete. Lyuboy perevorot, kotoryy ona pokazhet,
    porozhden isklyuchitelno vyborochnym shumom.

    DOBAVLENO 2026-09-05: ranshe eti chisla poluchalis vruchnuyu i v kode ih ne
    bylo, hotya oni -- yakor glavnogo utverzhdeniya raboty.

    Vazhnaya ogovorka, kotoraya obyazana popast v tekst: kazhdaya polovina neset
    VDVOE menshe sidov, chem osnovnoe izmerenie, poetomu uverenno uporyadochennyh
    par zdes menshe i procedura po postroeniyu konservativna.
    """
    rng = np.random.default_rng(seed)
    seeds = sorted(df["seed"].unique())
    h1, h2 = set(seeds[::2]), set(seeds[1::2])
    out = []
    for b in sorted(df["eps_target"].unique(), key=lambda x: (x == "native", x)):
        cells, amb = [], 0
        for _, g in df.groupby(CELL):
            gb = g[g.eps_target == b]
            r, t, a = _pair_counts(gb[gb.seed.isin(h1)], gb[gb.seed.isin(h2)],
                                   metric, rng)
            cells.append((r, t)); amb += a
        rev = sum(r for r, _ in cells); tot = sum(t for _, t in cells)
        lo, hi = _block_ci(cells, rng=rng)
        out.append(dict(budget=b, n_pairs=tot, n_ambiguous=amb, reversed=rev,
                        frac=rev / tot if tot else np.nan, lo=lo, hi=hi))
    res = pd.DataFrame(out)
    tr, tt = int(res["reversed"].sum()), int(res["n_pairs"].sum())
    print(f"  nulevaya kalibrovka, pool po vsem rezhimam: {tr}/{tt} = "
          f"{tr / tt:.2%}" if tt else "  net par")
    return res


def native_eps_table(df):
    """Sobstvennaya gromkost kazhdoy ataki -- podtverzhdenie posylki §2."""
    nat = df[df.eps_target == "native"]
    return (nat.groupby("attack")["eps_spent"]
            .agg(["median", "min", "max"]).sort_values("median"))


def harm_table(df, metric=METRIC):
    """Mediannyy vred po (agregator, ataka, byudzhet)."""
    return df.pivot_table(index=["agg", "attack"], columns="eps_target",
                          values=metric, aggfunc="median")


def eps50_table(df, q=0.5):
    """
    eps_q (METRICS §8a): naimenshiy byudzhet, pri kotorom mediana L dostigaet q.
    Vozvrashchaet takzhe eps_bar -- naibolshiy takoy byudzhet (u spadayushchih
    agregatorov vred nemonotonen po gromkosti).
    """
    rows = []
    # ISPRAVLENO 2026-09-05: schitaem po kanonicheskomu L_adj (add_harm), a ne po
    # syromu L. U L est pol okolo 0.31 na sinteticheskoy setke, poetomu porog
    # "L >= 0.5" byl radikalno slabee zayavlennogo v rukopisi "L_adj >= 0.5",
    # i eps50 iz etoy funktsii ne sovpadal s risunkom 4.
    if "L_adj" not in df.columns:
        raise ValueError("nuzhen kadr posle add_harm(): net kolonki L_adj")
    d2 = df[df.eps_target != "native"].copy()
    for (l, a, atk), g in d2.groupby(["landscape", "agg", "attack"]):
        m = g.groupby("eps_target")["L_adj"].median()
        m.index = m.index.astype(float)
        m = m.sort_index()
        hit = m.values >= q
        ok = m.index[hit]
        # SVYAZNOST. eps50 i eps_bar -- pervoe i poslednee prevyshenie poroga, a
        # ne granitsy nepreryvnogo intervala. Esli mezhdu nimi est provaly, eto
        # ne "polosa", i nazyvat ee tak v tekste nelzya. Flag dobavlen 2026-09-05.
        i = np.nonzero(hit)[0]
        contiguous = bool(len(i)) and bool(hit[i[0]:i[-1] + 1].all())
        rows.append(dict(landscape=l, agg=a, attack=atk,
                         eps50=(float(ok[0]) if len(ok) else np.nan),
                         eps_bar=(float(ok[-1]) if len(ok) else np.nan),
                         contiguous=int(contiguous),
                         censored=int(len(ok) == 0)))
    return pd.DataFrame(rows)


def report(path, metric=METRIC, baseline="results/baseline_syn.csv"):
    df = load(path)
    df = add_harm(df, pd.read_csv(baseline),
                  ["landscape", "instance", "agg", "seed"])
    print(f"strok: {len(df)}   landshaftov: {df.landscape.nunique()}   "
          f"ekzemplyarov: {df.instance.nunique()}   sidov: {df.seed.nunique()}")
    print(f"L_adj prinuditelno obnulen u {(df.L_adj <= 0).mean():.1%} progonov "
          f"(helped {df.helped.mean():.1%} + otricatelnye)\n")

    print("=" * 70)
    print("1. SOBSTVENNAYA GROMKOST ATAK (posylka: oni nesopostavimy)\n")
    print(native_eps_table(df).round(2).to_string())

    print("\n" + "=" * 70)
    print("2. PEREVOROTY PORYADKA pri uravnivanii byudzheta\n")
    rs = reversal_stats(df, metric)
    print(f"{'byudzhet':>9}{'par':>7}{'neyasnyh':>10}{'perevernulos':>14}"
          f"{'dolya':>8}{'95% CI':>18}")
    for _, r in rs.iterrows():
        print(f"{r.budget:>9}{int(r.n_pairs):>7}{int(r.n_ambiguous):>10}"
              f"{int(r['reversed']):>14}{r.frac:>8.1%}"
              f"{f'[{r.lo:.1%}, {r.hi:.1%}]':>18}")

    print("\n" + "=" * 70)
    print("2a. NULEVAYA KALIBROVKA (te zhe pary, nepersekayushchiesya poloviny sidov,\n"
          "    byudzhet NE menyaetsya -- vse perevoroty zdes ot shuma)\n")
    nc = null_calibration(df, metric)
    print(f"{'rezhim':>9}{'par':>7}{'perevernulos':>14}{'dolya':>8}")
    for _, r in nc.iterrows():
        print(f"{str(r.budget):>9}{int(r.n_pairs):>7}"
              f"{int(r['reversed']):>14}{r.frac:>8.1%}")

    print("\n" + "=" * 70)
    print("3. DOLYA PROGONOV, GDE ATAKA POMOGLA\n")
    print((df.groupby("attack")["helped"].mean() * 100).round(1).to_string())

    print("\n" + "=" * 70)
    print("4. eps50 -- tsena vreda v edinitsah gromkosti (mediana po ekzemplyaram)\n")
    e = eps50_table(df)
    piv = e.pivot_table(index="agg", columns="attack", values="eps50", aggfunc="median")
    print(piv.to_string())
    print(f"\ntsenzurirovano (tsel ne dostignuta ni pri odnom byudzhete): "
          f"{e.censored.mean():.0%} yacheek")
    return df, rs, e


if __name__ == "__main__":
    import sys
    report(sys.argv[1] if len(sys.argv) > 1 else "results/main.csv")
