"""
Figures for V1. All labels in English (submission language).

Palette: categorical set from the visualization guide, validated (all six checks
passed; worst adjacent CVD pair dE 9.1). Hue follows the entity, never its rank,
and the assignment is fixed across figures.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analyze import confident_order, _pair_counts, _block_ci

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
AGG_COLOR = {"mean": PAL[0], "trimmed": PAL[1], "median": PAL[2],
             "krum": PAL[3], "talwar": PAL[4], "tukey": PAL[6]}
ATK_COLOR = {"alie": PAL[0], "ipm": PAL[6], "gaussian": PAL[3], "signflip": PAL[2],
             "minsum": PAL[4], "minmax": PAL[1], "byzfl_1.5": PAL[5]}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

#: measured empirically, see v1/RESULTS.md §9
CLASS = {"krum": "redescending", "talwar": "redescending", "tukey": "redescending",
         "mean": "monotone", "trimmed": "monotone", "median": "flat"}
STYLE = {"redescending": ("-", 2.4), "monotone": ((0, (5, 2)), 1.8),
         "flat": ((0, (1, 2)), 1.8)}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "figure.dpi": 170,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})


def _grid(ax, axis="y"):
    ax.grid(axis=axis, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def load(path, bpath, keys):
    # L_adj schitaetsya v odnom meste na ves proekt -- analyze.add_harm.
    # Ranshe formula dublirovalas zdes i rashodilas s analyze.eps50_table i s
    # tablitsey 2 rukopisi (sm. kommentariy v add_harm).
    from .analyze import add_harm
    d = add_harm(pd.read_csv(path), pd.read_csv(bpath), keys)
    d["La"] = d["L_adj"]
    return d


# ------------------------------------------------------------------ 1


def fig_reversals(syn, mn, out):
    """Main result: fraction of attack pairs whose order flips at matched budget."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
    for ax, (d, keys, title, xl, nz) in zip(axes, [
            (syn, ["landscape", "instance", "agg"], "(a) Synthetic landscapes",
             "budget $\\varepsilon$", 0.003),
            (mn, ["agg"], "(b) MNIST + FedSGD",
             "budget $\\varepsilon$ ($\\times10^3$)", 0.0)]):
        rng = np.random.default_rng(0)
        bs = sorted([x for x in d.eps_target.unique() if x != "native"], key=float)
        xs, ys, los, his = [], [], [], []
        for b in bs:
            # BLOCHNYY bootstrap po yacheykam, kak v analyze.reversal_stats:
            # pary vnutri yacheyki ne nezavisimy, i naivnyy pool zanizhaet interval
            cells = []
            for _, g in d.groupby(keys):
                r, t, _ = _pair_counts(g[g.eps_target == "native"],
                                       g[g.eps_target == b], "La", rng)
                cells.append((r, t))
            rev = sum(r for r, _ in cells); tot = sum(t for _, t in cells)
            lo, hi = _block_ci(cells, rng=rng)
            sc = 1e-3 if "MNIST" in title else 1.0
            xs.append(float(b) * sc); ys.append(rev / tot); los.append(lo); his.append(hi)
        ax.fill_between(xs, los, his, color=PAL[0], alpha=0.15, lw=0, zorder=1)
        ax.plot(xs, ys, "-o", color=PAL[0], lw=2, ms=5, zorder=3,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.axhline(nz, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.text(xs[0], nz + max(his) * 0.035,
                f"null {nz:.1%}", fontsize=7.5, color=INK2)
        ax.set_title(title, loc="left"); ax.set_xlabel(xl)
        ax.set_ylim(0, max(his) * 1.32); _grid(ax)
        ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    axes[0].set_ylabel("attack pairs that reverse")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


# ------------------------------------------------------------------ 2


def fig_harm_curves(syn, mn, out):
    """
    Key finding: harm as a function of allowed loudness runs in opposite
    directions for two classes of aggregators.

    Hue carries the aggregator, LINE STYLE carries the class. End labels are
    omitted: with six series they collide, so identity is carried by the legend.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, (d, title, xl, sc) in zip(axes, [
            (syn, "(a) Synthetic landscapes", "budget $\\varepsilon$", 1.0),
            (mn, "(b) MNIST + FedSGD", "budget $\\varepsilon$ ($\\times10^3$)", 1e-3)]):
        q = d[d.eps_target != "native"].copy(); q["eps"] = q.eps_target.astype(float)
        piv = q.pivot_table(index="eps", columns="agg", values="La", aggfunc="median")
        for a in [x for x in ["krum", "talwar", "tukey", "median", "mean", "trimmed"]
                  if x in piv.columns]:
            ls, lw = STYLE[CLASS[a]]
            ax.plot(piv.index * sc, piv[a], ls=ls, lw=lw, color=AGG_COLOR[a],
                    label=f"{a} — {CLASS[a]}", zorder=3, marker="o", ms=3.5,
                    markeredgecolor="white", markeredgewidth=0.8)
        ax.set_title(title, loc="left"); ax.set_xlabel(xl); _grid(ax)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("harm $L_{adj}$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


# ------------------------------------------------------------------ 3


def fig_loudness_slope(syn, mn, out):
    """
    Premise: attacks differ in loudness by orders of magnitude, and the ORDER
    does not carry over between tasks.

    A slopegraph on ranks rather than paired bar charts: absolute scales differ
    by four orders of magnitude, and what matters is that the lines cross.
    """
    ns = syn[syn.eps_target == "native"].groupby("attack")["eps_spent"].median()
    nm = mn[mn.eps_target == "native"].groupby("attack")["eps_spent"].median()
    atks = sorted(set(ns.index) & set(nm.index))
    rs = ns[atks].rank(ascending=False)
    rm = nm[atks].rank(ascending=False)

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for a in atks:
        moved = abs(rs[a] - rm[a]) >= 3
        ax.plot([0, 1], [rs[a], rm[a]], "-o", color=ATK_COLOR[a],
                lw=2.8 if moved else 1.3, ms=6, zorder=4 if moved else 3,
                alpha=1.0 if moved else 0.7,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.annotate(f"{a}  {ns[a]:+.1f}", (0, rs[a]), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=8, color=ATK_COLOR[a])
        ax.annotate(f"{nm[a]/1000:+.1f}k  {a}", (1, rm[a]), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8, color=ATK_COLOR[a])
    ax.set_xlim(-0.62, 1.62); ax.set_ylim(len(atks) + 0.6, 0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Synthetic landscapes", "MNIST + FedSGD"], fontsize=9)
    ax.set_yticks(range(1, len(atks) + 1))
    ax.set_ylabel("loudness rank  (1 = loudest)")
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", alpha=0.5, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


# ------------------------------------------------------------------ 4


def fig_useful_window(syn, out, q=0.5):
    """
    The band of loudness in which an attack still does serious damage.

    Redescending defenses admit only a narrow window; averaging ones have no
    upper bound at all. Directly actionable for defense design.
    """
    d = syn[syn.eps_target != "native"].copy(); d["eps"] = d.eps_target.astype(float)
    rows = []
    for (a, atk), g in d.groupby(["agg", "attack"]):
        m = g.groupby("eps")["La"].median().sort_index()
        ok = m.index[m.values >= q]
        if len(ok):
            rows.append((a, atk, float(ok[0]), float(ok[-1])))
    W = pd.DataFrame(rows, columns=["agg", "attack", "lo", "hi"])
    # ascending i draws upward, so this list is bottom-to-top: redescending on top
    order = [a for a in ["mean", "trimmed", "median", "tukey", "talwar", "krum"]
             if a in set(W["agg"])]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    top = d.eps.max()
    for i, a in enumerate(order):
        sub = W[W["agg"] == a]
        n = len(sub)
        for j, (_, r) in enumerate(sub.iterrows()):
            y = i + (j - (n - 1) / 2) * 0.17
            # a window of zero width is still a window: give it a visible stub
            ax.plot([r.lo, max(r.hi, r.lo + 0.3)], [y, y], lw=3.4,
                    color=AGG_COLOR[a], solid_capstyle="butt", zorder=3, alpha=0.9)
            if r.hi >= top:
                ax.annotate("", xy=(top + 1.2, y), xytext=(r.hi, y), zorder=3,
                            arrowprops=dict(arrowstyle="-|>", color=AGG_COLOR[a],
                                            lw=1.5))
    for i, a in enumerate(order):
        ax.text(-6.3, i, a, ha="right", va="center", fontsize=9,
                color=AGG_COLOR[a], fontweight="bold")
        ax.text(-6.3, i + 0.33, f"{CLASS[a]} · {len(W[W['agg'] == a])} of 7 attacks",
                ha="right", va="center", fontsize=7, color=INK2)
    ax.set_yticks([]); ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlim(-6.5, top + 2.0)
    ax.set_xlabel("detectability budget $\\varepsilon$")
    ax.axvline(0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(0.2, -0.7, "honest", fontsize=7, color=INK2)
    _grid(ax, axis="x")
    for sp in ["left"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


# ------------------------------------------------------------------ 5


def fig_malicious_fraction(out):
    """
    Context: the attack's sign depends on how many clients run it. With one
    attacker in twenty it helps the swarm in 43% of runs.
    """
    ms, help_, harm = [1, 2, 4, 6], [], []
    for m in ms:
        s = pd.read_csv(f"results/sweep_m{m}.csv")
        help_.append(s.helped.mean() * 100)
        harm.append(s["d"].median())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    x = [m / 20 * 100 for m in ms]
    for ax, (v, lab, col, fmt, lo) in zip(axes, [
            (help_, "(a) runs where the attack helped the swarm", PAL[2], "{:.0f}%", 0),
            (harm, "(b) median harm $\\Delta$", PAL[1], "{:.2f}", 0)]):
        ax.plot(x, v, "-o", color=col, lw=2.2, ms=6,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        for xi, vi in zip(x, v):
            ax.annotate(fmt.format(vi), (xi, vi), xytext=(0, 8),
                        textcoords="offset points", ha="center",
                        fontsize=7.5, color=INK2)
        ax.set_xlabel("malicious clients, %")
        ax.set_title(lab, loc="left", fontsize=9, pad=8)
        _grid(ax); ax.set_ylim(lo, max(v) * 1.25)
    axes[0].yaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


if __name__ == "__main__":
    syn = load("results/m4_main.csv", "results/baseline_syn.csv",
               ["landscape", "instance", "agg", "seed"])
    mn = load("results/mnist.csv", "results/baseline_mnist.csv", ["agg", "seed"])
    print("Figures:")
    fig_reversals(syn, mn, "figures/fig1_reversals.png")
    fig_harm_curves(syn, mn, "figures/fig2_harm_curves.png")
    fig_loudness_slope(syn, mn, "figures/fig3_loudness_slope.png")
    fig_useful_window(syn, "figures/fig4_useful_window.png")
    fig_malicious_fraction("figures/fig5_malicious.png")
