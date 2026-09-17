"""
Realnaya zadacha federativnogo obucheniya: MNIST + FedSGD + perceptron 784-32-10.

Kanonicheskiy protokol po SoK 2025: tsentralnyy server, polnoe uchastie klientov,
FedSGD (bolshinstvo atak i zashchit na otravlenie modeli postroeny imenno na nem),
odin skrytyy sloy poryadka 32 neyronov, IID-razbienie kak osnovnoy rezhim.

DVE VAZHNYE OTLICHIYA OT SINTETICHESKIH LANDSHAFTOV
---------------------------------------------------
1. f* NEIZVESTNA, poetomu metrika d (logarifm otnosheniya zazorov) neprimenima.
   Osnovnoy stanovitsya L -- poterya vychislitelnogo progressa, ona f* ne trebuet.

2. RAZMERNOST >> CHISLO KLIENTOV: 784*32+32+32*10+10 = 25482 protiv n ~ 20.
   Polnaya kovariatsionnaya matritsa vyrozhdena beznadezhno, i mahalanobisov
   byudzhet v ishodnom vide NEPRIMENIM. Ispolzuetsya DIAGONALNAYA kovariatsiya.
   Eto ne kostyl: ALIE po postroeniyu ispolzuet POKOORDINATNOE s.k.o., to est
   diagonalnoe priblizhenie -- de-fakto standart oblasti. Odnako eto oznachaet,
   chto byudzhet perestaet videt korrelyatsii mezhdu koordinatami, i ataka,
   anomalnaya imenno po sochetaniyu koordinat, dlya nego nevidima.
   [?] Ogranichenie, podlezhashchee yavnoy ogovorke v tekste raboty.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from . import aggregators as AGG
from .swarm import Attack, Ctx

_DATA = {}


def load_mnist(path_x="results/_mnist_X.npy", path_y="results/_mnist_y.npy"):
    if "X" not in _DATA:
        _DATA["X"] = np.load(path_x)
        _DATA["y"] = np.load(path_y)
    return _DATA["X"], _DATA["y"]


# ------------------------------------------------------------------ model


class MLP:
    """784 -> H -> 10, softmax + kross-entropiya. Realizatsiya na numpy:
    polnyy kontrol nad gradientom i nikakih nakladnyh rashodov freymvorka."""

    def __init__(self, d_in=784, h=32, k=10):
        self.d_in, self.h, self.k = d_in, h, k
        self.shapes = [(d_in, h), (h,), (h, k), (k,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.dim = sum(self.sizes)

    def init(self, rng):
        w1 = rng.normal(0, np.sqrt(2.0 / self.d_in), (self.d_in, self.h))
        b1 = np.zeros(self.h)
        w2 = rng.normal(0, np.sqrt(2.0 / self.h), (self.h, self.k))
        b2 = np.zeros(self.k)
        return np.concatenate([w1.ravel(), b1, w2.ravel(), b2])

    def unpack(self, w):
        out, i = [], 0
        for sh, sz in zip(self.shapes, self.sizes):
            out.append(w[i:i + sz].reshape(sh)); i += sz
        return out

    def loss_grad(self, w, X, y):
        W1, b1, W2, b2 = self.unpack(w)
        n = len(X)
        h = np.maximum(X @ W1 + b1, 0.0)                      # ReLU
        logits = h @ W2 + b2
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits); p = e / e.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.maximum(p[np.arange(n), y], 1e-12)).mean())
        dl = p.copy(); dl[np.arange(n), y] -= 1.0; dl /= n
        gW2 = h.T @ dl; gb2 = dl.sum(0)
        dh = (dl @ W2.T) * (h > 0)
        gW1 = X.T @ dh; gb1 = dh.sum(0)
        return loss, np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])

    def grads_batch(self, w, Xs, ys):
        """
        Gradienty VSEH klientov za odin prohod.

        Xs -- (C, B, 784), ys -- (C, B). Vozvrashchaet (C, dim).

        Zachem. Tsikl po klientam delaet C otdelnyh umnozheniy matrits 32x784,
        chto sliskom malo dlya effektivnoy raboty numpy: progon upiraetsya v
        nakladnye rashody, a ne v vychisleniya (po otsenke FLOP -- v desyat raz).
        Zdes vse klienty schitayutsya odnim tenzornym vyrazheniem.
        """
        W1, b1, W2, b2 = self.unpack(w)
        C, B, _ = Xs.shape
        h = np.maximum(Xs @ W1 + b1, 0.0)                    # (C, B, H)
        logits = h @ W2 + b2                                 # (C, B, K)
        logits -= logits.max(axis=2, keepdims=True)
        e = np.exp(logits); p = e / e.sum(axis=2, keepdims=True)
        dl = p
        ci = np.arange(C)[:, None]; bi = np.arange(B)[None, :]
        dl[ci, bi, ys] -= 1.0
        dl /= B
        # Batchevyy matmul, a NE einsum: dlya svertok vida "cbd,cbh->cdh" einsum
        # ne uhodit v BLAS i schitaet naivno -- izmereno vtroe medlennee tsikla.
        gW2 = np.matmul(h.transpose(0, 2, 1), dl)            # (C, H, K)
        gb2 = dl.sum(axis=1)                                 # (C, K)
        dh = np.matmul(dl, W2.T) * (h > 0)                   # (C, B, H)
        gW1 = np.matmul(Xs.transpose(0, 2, 1), dh)           # (C, D, H)
        gb1 = dh.sum(axis=1)                                 # (C, H)
        return np.concatenate([gW1.reshape(C, -1), gb1,
                               gW2.reshape(C, -1), gb2], axis=1)

    def loss_acc(self, w, X, y):
        W1, b1, W2, b2 = self.unpack(w)
        h = np.maximum(X @ W1 + b1, 0.0)
        logits = h @ W2 + b2
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits); p = e / e.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())
        return loss, float((p.argmax(1) == y).mean())


# ------------------------------------------------------------------ byudzhet (diagonalnyy)


def diag_moments(Y, shrink=0.1):
    """
    Srednee i pokoordinatnaya dispersiya so SZHATIEM k srednej dispersii.

    Zhestkiy pol vida max(var, 1e-12) zdes NEDOPUSTIM: u MNIST kraevye pikseli
    vsegda nulevye, poetomu vesa pri nih imeyut TOCHNO nulevuyu dispersiyu
    gradienta po vsem klientam. Delenie na pol prevrashchaet chislennuyu pyl v
    gigantskiy vklad -- pri pervom progone d_eff vyshla 7.6e8 protiv nominalnoy
    razmernosti 25450, t.e. zavyshenie v 30 tysyach raz (2026-08-28).

    Szhatie k srednej dispersii lechit eto ustoychivo i yavlyaetsya pryamym
    analogom szhatiya k diagonali v polnomernom sluchae.
    """
    mu = Y.mean(axis=0)
    var = Y.var(axis=0, ddof=1) if len(Y) > 1 else np.ones(Y.shape[1])
    m = float(np.mean(var)) + 1e-30
    return mu, (1.0 - shrink) * var + shrink * m


def diag_maha(z, mu, var):
    d = z - mu
    return float(np.sum(d * d / var))


def diag_project(z, mu, var, eps_budget, d_eff, two_sided=True):
    R2 = d_eff + 2.0 * eps_budget
    if R2 <= 0:
        return mu.copy(), True
    m2 = diag_maha(z, mu, var)
    if m2 <= 1e-18:
        return z, False
    if not two_sided and m2 <= R2:
        return z, False
    return mu + (z - mu) * np.sqrt(R2 / m2), True


def diag_effective_d(rounds):
    """d_eff po chestnym gradientam s isklyucheniem samoy tochki, pool po raundam."""
    vals = []
    for Y in rounds:
        n = len(Y)
        if n < 3:
            continue
        for i in range(n):
            mu, var = diag_moments(np.delete(Y, i, axis=0))
            vals.append(diag_maha(Y[i], mu, var))
    return float(np.mean(vals)) if vals else float(rounds[0].shape[1])


# ------------------------------------------------------------------ protokol


@dataclass
class FedConfig:
    n_clients: int = 20
    n_malicious: int = 4
    T: int = 150
    lr: float = 0.5
    batch: int = 32
    hidden: int = 32
    eps_budget: float | None = None
    d_eff: float | None = None
    two_sided: bool = True
    n_test: int = 4000
    agg_kwargs: dict = field(default_factory=dict)

    @property
    def n_agents(self):        # sovmestimost s atakami, napisannymi pod Config roya
        return self.n_clients


@dataclass
class FedTrace:
    f_test: np.ndarray          # test-loss po raundam -- igraet rol f(x_t)
    acc: np.ndarray
    eps_spent: np.ndarray
    honest_rounds: list


def run_fed(agg_name, cfg: FedConfig, seed: int, attack: Attack | None = None,
            collect_honest=False) -> FedTrace:
    X, y = load_mnist()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    te = idx[:cfg.n_test]
    tr = idx[cfg.n_test:]
    shards = np.array_split(tr, cfg.n_clients)          # IID-razbienie
    Xte, yte = X[te], y[te]

    net = MLP(h=cfg.hidden)
    w = net.init(np.random.default_rng(10_000 + seed))
    agg = AGG.get(agg_name)

    n_mal = 0 if attack is None else cfg.n_malicious
    mal, hon = list(range(n_mal)), list(range(n_mal, cfg.n_clients))
    if attack is not None:
        attack.reset(np.random.default_rng(20_000 + seed), cfg)

    brng = np.random.default_rng(30_000 + seed)         # obshchiy dlya pary progonov
    f = np.empty(cfg.T); acc = np.empty(cfg.T)
    eps = np.full(cfg.T, np.nan)
    hrounds = []
    c_prev = np.zeros(net.dim)

    for t in range(cfg.T):
        bs = np.stack([brng.choice(shards[i], size=cfg.batch, replace=False)
                       for i in range(cfg.n_clients)])
        G = net.grads_batch(w, X[bs], y[bs])

        if attack is not None:
            Yh = G[hon]
            mu, var = diag_moments(Yh)
            orc = {"honest": Yh.copy()} if str(attack.level) >= "1" else None
            for i in mal:
                # V PROSTRANSTVE GRADIENTOV "vverh" -- eto MINUS gradient:
                # server delaet w <- w - lr*AGG, poetomu chtoby podnyat poteryu,
                # nado otpravit -g. I yakor -- nol (net obnovleniya), a ne
                # predydushchiy agregat: v gradientnom prostranstve on ne igraet
                # roli tekushchego polozheniya.
                ctx = Ctx(t=t, x_own=np.zeros(net.dim), g_own=-G[i],
                          honest_proposal=G[i], c_prev=None, rho_prev=None,
                          oracle=orc)
                z = np.asarray(attack.propose(ctx), float)
                if cfg.eps_budget is not None:
                    z, _ = diag_project(z, mu, var, cfg.eps_budget,
                                        cfg.d_eff if cfg.d_eff else net.dim,
                                        cfg.two_sided)
                if i == 0:
                    eps[t] = 0.5 * (diag_maha(z, mu, var)
                                    - (cfg.d_eff if cfg.d_eff else net.dim))
                G[i] = z

        if collect_honest:
            hrounds.append(G[hon].copy())

        c = agg(G, **cfg.agg_kwargs)
        if attack is not None:
            attack.observe(c, G[0])
        w = w - cfg.lr * c
        c_prev = c
        f[t], acc[t] = net.loss_acc(w, Xte, yte)

    return FedTrace(f, acc, eps, hrounds)
