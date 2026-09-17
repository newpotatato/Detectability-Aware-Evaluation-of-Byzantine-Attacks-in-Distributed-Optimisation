"""
Landshafty dlya V1. Vdohnovleny BBOB: izvestnyy optimum, sluchaynyy sdvig i
povorot na kazhdyy ekzemplyar, chetyre klassa trudnosti.

Kazhdaya funktsiya vozvrashchaet znachenie i analiticheskiy gradient.
Optimum f* izvesten po postroeniyu (zdes vsegda 0 posle sdviga).
"""
from __future__ import annotations
import numpy as np


def _rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """Sluchaynaya ortogonalnaya matritsa (QR ot gaussovoy)."""
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    return q * np.sign(np.diag(r))


class Landscape:
    """Bazovyy klass. x_opt -- polozhenie minimuma, f_opt -- ego znachenie."""

    name = "base"

    def __init__(self, d: int, instance: int = 0, domain: float = 5.0):
        self.d = d
        self.domain = domain
        rng = np.random.default_rng(1000 + instance)
        self.x_opt = rng.uniform(-domain * 0.8, domain * 0.8, d)
        self.R = _rotation(d, rng)
        self.f_opt = 0.0

    def __call__(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample_start(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """n nachalnyh polozheniy, ravnomerno po oblasti."""
        return rng.uniform(-self.domain, self.domain, (n, self.d))


class Sphere(Landscape):
    """BBOB f1. Vypuklaya, horosho obuslovlennaya. Kontrolnyy sluchay."""

    name = "sphere"

    def __call__(self, x):
        z = x - self.x_opt
        return float(z @ z)

    def grad(self, x):
        return 2.0 * (x - self.x_opt)


class Ellipsoid(Landscape):
    """BBOB f10. Silno vytyanutaya, povernutaya. Chislo obuslovlennosti 1e6."""

    name = "ellipsoid"

    def __init__(self, d, instance=0, domain=5.0, cond=1e3):
        """
        cond -- chislo obuslovlennosti. BBOB f10 ispolzuet 1e6, no pri prostom
        gradientnom spuske eto nerabotosposobno: max sobstvennoe znachenie
        gessiana 2*cond, ustoychivyy shag trebuet lr < 1/cond, i roy rashoditsya.
        Po umolchaniyu vzyato 1e3 -- landshaft ostaetsya sushchestvenno
        anizotropnym, no razreshim shagom poryadka 1e-3.
        [?] Otklonenie ot BBOB, ogovorit v tekste raboty.
        """
        super().__init__(d, instance, domain)
        p = np.arange(d) / max(d - 1, 1)
        self.w = cond ** p                       # vesa po osyam

    def __call__(self, x):
        z = self.R @ (x - self.x_opt)
        return float(self.w @ (z * z))

    def grad(self, x):
        z = self.R @ (x - self.x_opt)
        return 2.0 * (self.R.T @ (self.w * z))


class Rastrigin(Landscape):
    """BBOB f15. Multimodalnaya s reshetkoy minimumov i globalnoy strukturoy."""

    name = "rastrigin"

    def __init__(self, d, instance=0, domain=5.0, amp=10.0):
        super().__init__(d, instance, domain)
        self.amp = amp

    def __call__(self, x):
        z = x - self.x_opt
        return float(self.amp * (self.d - np.sum(np.cos(2 * np.pi * z))) + z @ z)

    def grad(self, x):
        z = x - self.x_opt
        return 2.0 * np.pi * self.amp * np.sin(2 * np.pi * z) + 2.0 * z


class Gallagher(Landscape):
    """
    BBOB f21-podobnaya. Summa gaussovyh pikov so sluchaynymi polozheniyami i
    vysotami -- multimodalnaya BEZ globalnoy struktury. Samyy trudnyy klass.
    Realizovana kak minus-smes: minimum tam, gde samyy vysokiy pik.
    """

    name = "gallagher"

    def __init__(self, d, instance=0, domain=5.0, n_peaks=21, width=2.0):
        super().__init__(d, instance, domain)
        rng = np.random.default_rng(2000 + instance)
        self.centers = rng.uniform(-domain * 0.8, domain * 0.8, (n_peaks, d))
        self.centers[0] = self.x_opt                    # samyy vysokiy pik
        self.heights = np.concatenate([[10.0], rng.uniform(1.0, 9.0, n_peaks - 1)])
        self.inv_w = 1.0 / (width ** 2)
        self._refine_optimum()

    def _terms(self, x):
        diff = x[None, :] - self.centers                # (K, d)
        q = np.einsum("kd,kd->k", diff, diff) * self.inv_w / self.d
        return diff, self.heights * np.exp(-0.5 * q)

    def __call__(self, x):
        _, t = self._terms(x)
        return float(-np.sum(t))

    def grad(self, x):
        diff, t = self._terms(x)
        # d/dx [-sum h_k exp(-q_k/2)] = sum h_k exp(-q_k/2) * diff_k * inv_w / d
        return np.einsum("k,kd->d", t, diff) * (self.inv_w / self.d)

    def _refine_optimum(self, steps=4000, lr=0.05):
        """
        Tsentr samogo vysokogo pika NE yavlyaetsya minimumom: sosednie piki ego
        smeshchayut. Utochnyaem lokalnym spuskom i fiksiruem f_opt.
        [?] Predpolagaetsya, chto globalnyy minimum lezhit v basseyne samogo
        vysokogo pika. Pri h_0=10 protiv ostalnyh <=9 i razdelennyh tsentrah
        eto pochti vsegda tak, no garantii net -- proveryat pri smene parametrov.
        """
        x = self.centers[0].copy()
        for _ in range(steps):
            g = self.grad(x)
            x -= lr * g
            if np.linalg.norm(g) < 1e-12:
                break
        self.x_opt = x
        self.f_opt = self(x)


REGISTRY = {
    "sphere": Sphere,
    "ellipsoid": Ellipsoid,
    "rastrigin": Rastrigin,
    "gallagher": Gallagher,
}


def make(name: str, d: int, instance: int = 0, **kw) -> Landscape:
    return REGISTRY[name](d, instance, **kw)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"{'landscape':<12} {'f(x_opt)':>12} {'|grad| pri opt':>16} {'f(sluch)':>14}")
    print("-" * 58)
    for nm in REGISTRY:
        L = make(nm, d=10)
        x = rng.uniform(-5, 5, 10)
        # chislennaya proverka gradienta
        h, g_num = 1e-6, np.zeros(10)
        for i in range(10):
            e = np.zeros(10); e[i] = h
            g_num[i] = (L(x + e) - L(x - e)) / (2 * h)
        err = np.max(np.abs(g_num - L.grad(x))) / (np.max(np.abs(g_num)) + 1e-12)
        print(f"{nm:<12} {L(L.x_opt):>12.4g} {np.linalg.norm(L.grad(L.x_opt)):>16.3e} "
              f"{L(x):>14.4g}   otn.osh.grad={err:.2e}")
