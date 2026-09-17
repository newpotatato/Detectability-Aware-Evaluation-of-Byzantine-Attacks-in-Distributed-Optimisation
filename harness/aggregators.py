"""
Robastnye agregatory v R^d.

Vse prinimayut massiv (n, d) i vozvrashchayut vektor (d,).
Flag `redescending` pomechaet operatory s NEMONOTONNOY krivoy vliyaniya --
imenno na nih dolzhna lomatsya obratnaya svyaz pervogo poryadka (V1 §5.2).
"""
from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------- prostye


def mean(X, **kw):
    return X.mean(axis=0)


def median(X, **kw):
    """Pokoordinatnaya mediana."""
    return np.median(X, axis=0)


def trimmed_mean(X, alpha=0.2, **kw):
    """Pokoordinatnoe otsechenie alpha s kazhdoy storony."""
    n = len(X)
    k = int(np.floor(alpha * n))
    if n - 2 * k <= 0:
        return np.median(X, axis=0)
    S = np.sort(X, axis=0)
    return S[k:n - k].mean(axis=0)


def geometric_median(X, iters=200, tol=1e-10, **kw):
    """Weiszfeld."""
    v = np.median(X, axis=0)
    for _ in range(iters):
        dist = np.linalg.norm(X - v, axis=1)
        dist = np.maximum(dist, 1e-12)
        w = 1.0 / dist
        new = (w[:, None] * X).sum(axis=0) / w.sum()
        if np.linalg.norm(new - v) < tol:
            return new
        v = new
    return v


# ---------------------------------------------------------------- distantsionnye


def _krum_scores(X, f):
    """
    Popairnye kvadraty rasstoyaniy cherez razlozhenie ||xi-xj||^2 = a_i + a_j - 2 xi.xj.

    Pryamoe vychislenie X[:,None,:] - X[None,:,:] sozdaet massiv (n, n, d): pri
    n=20 i d=25450 (real'naya model) eto 81 MB na KAZHDYY vyzov v KAZHDOM raunde,
    chto v mnogoprotsessnom progone privodit k ischerpaniyu pamyati (2026-08-28).
    Razlozhenie daet matritsu (n, n) i ne zavisit ot d po pamyati.
    """
    n = len(X)
    a = np.einsum("ij,ij->i", X, X)
    D = np.maximum(a[:, None] + a[None, :] - 2.0 * (X @ X.T), 0.0)
    m = max(n - f - 2, 1)
    return np.sort(D, axis=1)[:, 1:m + 1].sum(axis=1)


def krum(X, f=2, **kw):
    return X[int(np.argmin(_krum_scores(X, f)))]


def multi_krum(X, f=2, m=None, **kw):
    n = len(X)
    m = m if m is not None else max(n - 2 * f, 1)
    idx = np.argsort(_krum_scores(X, f))[:m]
    return X[idx].mean(axis=0)


def bulyan(X, f=2, **kw):
    """Krum-otbor theta kandidatov, zatem pokoordinatnyy usechennyy srednim."""
    n = len(X)
    theta = max(n - 2 * f, 1)
    pool, sel = list(range(n)), []
    for _ in range(theta):
        sub = X[pool]
        i = int(np.argmin(_krum_scores(sub, f))) if len(sub) > 2 else 0
        sel.append(pool[i]); pool.pop(i)
        if not pool:
            break
    S = X[sel]
    beta = max(len(S) - 2 * f, 1)
    med = np.median(S, axis=0)
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        col = S[:, j]
        out[j] = col[np.argsort(np.abs(col - med[j]))[:beta]].mean()
    return out


# ---------------------------------------------------------------- otsekayushchie


def centered_clipping(X, tau=1.0, iters=3, **kw):
    """
    CCLIP (Karimireddy i dr.): v <- v + mean_i (x_i - v) * min(1, tau/||x_i - v||).
    Nachalnoe v -- pokoordinatnaya mediana (v originale eto moment proshlogo raunda).
    """
    v = np.median(X, axis=0)
    for _ in range(iters):
        diff = X - v
        nrm = np.maximum(np.linalg.norm(diff, axis=1), 1e-12)
        scale = np.minimum(1.0, tau / nrm)
        v = v + (diff * scale[:, None]).mean(axis=0)
    return v


# ---------------------------------------------------------------- spadayushchie


def _robust_scale(X, v):
    """MAD-podobnyy masshtab rasstoyaniy do tekushchey otsenki."""
    dist = np.linalg.norm(X - v, axis=1)
    return 1.4826 * np.median(np.abs(dist - np.median(dist))) + 1e-12


def tukey(X, c=4.685, iters=50, tol=1e-12, **kw):
    """Bikvadratnaya M-otsenka. Ves plavno padaet do nulya na |u|=1."""
    v = np.median(X, axis=0)
    for _ in range(iters):
        s = _robust_scale(X, v)
        u = np.linalg.norm(X - v, axis=1) / (c * s)
        w = np.where(u <= 1.0, (1.0 - u ** 2) ** 2, 0.0)
        if w.sum() < 1e-12:
            break
        new = (w[:, None] * X).sum(axis=0) / w.sum()
        if np.linalg.norm(new - v) < tol:
            return new
        v = new
    return v


def talwar(X, c=2.795, iters=50, tol=1e-12, **kw):
    """Zhestkoe otsechenie: ves 1 vnutri poroga, 0 snaruzhi. Razryvnaya."""
    v = np.median(X, axis=0)
    for _ in range(iters):
        s = _robust_scale(X, v)
        w = (np.linalg.norm(X - v, axis=1) <= c * s).astype(float)
        if w.sum() < 1e-12:
            break
        new = (w[:, None] * X).sum(axis=0) / w.sum()
        if np.linalg.norm(new - v) < tol:
            return new
        v = new
    return v


# ---------------------------------------------------------------- reestr

AGGREGATORS = {
    "mean":        dict(fn=mean,               redescending=False, bounded=False),
    "median":      dict(fn=median,             redescending=False, bounded=True),
    "trimmed":     dict(fn=trimmed_mean,       redescending=False, bounded=True),
    "geomedian":   dict(fn=geometric_median,   redescending=False, bounded=True),
    "krum":        dict(fn=krum,               redescending=True,  bounded=True),
    "multikrum":   dict(fn=multi_krum,         redescending=True,  bounded=True),
    "bulyan":      dict(fn=bulyan,             redescending=False, bounded=True),
    "cclip":       dict(fn=centered_clipping,  redescending=False, bounded=True),
    "tukey":       dict(fn=tukey,              redescending=True,  bounded=True),
    "talwar":      dict(fn=talwar,             redescending=True,  bounded=True),
}

#: Yadro nabora dlya V1 -- pokryvaet vse kachestvenno razlichnye formy vliyaniya
CORE = ["mean", "median", "trimmed", "krum", "tukey"]
#: Operatory, na kotoryh ozhidaetsya otkaz obratnoy svyazi pervogo poryadka
REDESCENDING = [k for k, v in AGGREGATORS.items() if v["redescending"]]


def get(name):
    return AGGREGATORS[name]["fn"]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 21, 10
    Y = rng.normal(0.0, 1.0, (n - 1, d))
    base = {k: get(k)(Y) for k in AGGREGATORS}

    GRID = [0.5, 1, 2, 3, 5, 8, 15, 40]

    def classify(v):
        """
        rastet   -- maksimum na krayu setki, rost sushchestvennyy
        ploskoe  -- krivaya pochti postoyanna: amplituda bezrazlichna
        SPADAET  -- maksimum vnutri, hvost zametno nizhe (pervyy poryadok slomaetsya)
        """
        v = np.asarray(v)
        lo, hi = v.min(), v.max()
        if hi <= 1e-12 or hi / max(lo, 1e-12) < 1.05:
            return "ploskoe", 0.0
        drop = 1.0 - v[-1] / hi
        if int(np.argmax(v)) < len(v) - 1 and drop > 0.3:
            return "SPADAET", drop
        return "rastet", drop

    print("Krivaya vliyaniya: ||AGG(Y u z) - AGG(Y)|| kak funktsiya amplitudy ||z||\n")
    print(f"{'agregator':<12}" + "".join(f"{m:>8.1f}" for m in GRID)
          + f"{'k*':>7}{'forma':>10}{'spad':>7}")
    print("-" * 100)
    u = np.zeros(d); u[0] = 1.0
    for k in AGGREGATORS:
        f = get(k)
        vals = [np.linalg.norm(f(np.vstack([Y, kappa * u])) - base[k]) for kappa in GRID]
        form, drop = classify(vals)
        kstar = GRID[int(np.argmax(vals))]
        print(f"{k:<12}" + "".join(f"{x:>8.3f}" for x in vals)
              + f"{kstar:>7.1f}{form:>10}{drop:>7.0%}")

    print("\nSPADAET => u ataki est VNUTRENNIY optimum amplitudy;")
    print("           zakon, sleduyushchiy za proizvodnoy, proskochit ego v zonu nulevogo vliyaniya.")
