# Comparing Byzantine attacks at matched detectability

Code for the paper *Comparing Byzantine attacks at matched detectability*
(K. Chernikov, E. Tarasova).

Byzantine attacks on distributed optimisation are usually ranked by the harm
they cause, with each attack run at whatever magnitude its reference
implementation happens to ship. Magnitude is free in such a comparison, so the
ranking confounds how clever an attack is with how loudly it shouts. This code
equalises attacks by a detectability budget — a bound on the Kullback–Leibler
rate between the observed and the honest report distributions — and compares
them at matched loudness.

## Installation

Python 3.11 with NumPy, SciPy, pandas and Matplotlib:

    pip install -r requirements.txt

`harness.fetch_mnist` additionally needs scikit-learn; nothing else does.

## Reproducing the results

Run from the repository root. Every script writes to `results/`, which is not
tracked here.

    python -m harness.audit_full                  # 32 numerical self-checks
    python -m harness.run_all                     # full V1 programme, ~2 h on 11 cores
    python -m harness.fetch_mnist                 # MNIST cache, ~200 MB
    OMP_NUM_THREADS=1 python -m harness.run_mnist # federated task, ~4 h on 10 cores
    python -m harness.run_baselines               # floors of the harm metric L
    python -m harness.analyze results/m4_main.csv # ranks, intervals, reversals
    python -m harness.paper_numbers               # every number quoted in the paper
    python -m harness.figures                     # the five figures

`run_mnist` needs `OMP_NUM_THREADS=1`. Without it the worker processes contend
for BLAS threads and the run takes about six times longer.

Outcomes are stored per seed, so ranks, confidence intervals and $\varepsilon_{50}$
are recomputed from the CSVs without repeating the runs.

## Layout

| | |
|---|---|
| `harness/landscapes.py` | BBOB/COCO function classes: sphere, ellipsoid, Rastrigin, Gallagher |
| `harness/swarm.py` | the optimisation loop the agents share |
| `harness/aggregators.py` | mean, coordinate-wise median, trimmed mean, Krum, Talwar, Tukey |
| `harness/attacks.py` | sign flipping, Gaussian, ALIE, IPM, Min-Max, Min-Sum, adaptive policies |
| `harness/budget.py` | the detectability budget: KL rate under a Mahalanobis model |
| `harness/metrics.py` | harm $L$, saturation, useful window, $d$, $\varepsilon_{50}$ |
| `harness/fedsgd.py` | federated task: FedSGD, 784–32–10 perceptron on MNIST |
| `harness/runner.py`, `runner_fed.py` | parallel grid engines |
| `harness/analyze.py`, `paper_numbers.py` | analysis and the paper's numbers |
| `harness/figures.py` | the five figures |
| `harness/audit_full.py`, `audit_run.py` | verification suite |
| `figures/` | the figures as published |

## Figures

| | |
|---|---|
| `fig1_reversals.png` | attack pairs whose order reverses once loudness is matched |
| `fig2_harm_curves.png` | harm against budget, by aggregation rule |
| `fig3_loudness_slope.png` | harm per unit of loudness |
| `fig4_useful_window.png` | the budget range in which an attack is worth running |
| `fig5_malicious.png` | dependence on the malicious fraction |

## Citation

    @article{chernikov_matched_detectability,
      author  = {Chernikov, Kirill and Tarasova, Elizaveta},
      title   = {Comparing Byzantine Attacks at Matched Detectability},
      year    = {2026}
    }

## License

MIT, see [LICENSE](LICENSE).
