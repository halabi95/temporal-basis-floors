# temporal-basis-floors

Code and results for

> H. El Halabi and Y. Takahashi, **How Accurately Must the Frequency Be Known? Temporal Basis Selection for Reduced-Order Reconstruction of Periodic Fluid-Structure Interaction**, submitted to *Turbomachinery* (ISROMAC19 special issue), 2026.

Frequency-domain methods represent a time-periodic flow on a small basis of temporal functions. This repository computes the **projection floor** of three such bases (harmonic balance, proper orthogonal decomposition and dynamic mode decomposition), the smallest error any method restricted to a given basis can reach. It quantifies how the harmonic-balance floor degrades when the assumed fundamental frequency is wrong, and checks the closed-form phase-slip model of that degradation (Eq. 9 of the paper) against the measured floors.

## Repository layout

| Path | Contents |
|---|---|
| `src/floors.py` | All projection floors, the frequency sensitivity, the region floors and the Eq. (9) check; writes `results/floors.json` |
| `src/make_fig1_setup.py` | Reproduces Fig. 1: the network's subdomains and the 753 measurement points (needs the reference mesh) |
| `src/make_figures.py` | Reproduces Figs. 2, 3 and 4 from `results/floors.json` |
| `src/aggregate_trained.py` | Reproduces Table 3 (trained reconstruction) from the result CSVs |
| `results/floors.json` | Output of `floors.py` on the reference data used in the paper |
| `results/trained_*.csv` | Evaluated errors and floors of the 54 trained networks (3 seeds each) |
| `figures/` | Fig. 1 as PNG; Figs. 2 to 4 as PDF and PNG |
| `data/` | Place the reference `.npz` files here (not tracked by git) |

## Reproducing the paper

```bash
pip install -r requirements.txt
python src/make_figures.py        # Figs. 2-4 from results/floors.json
python src/aggregate_trained.py   # Table 3 from results/trained_*.csv
python src/floors.py              # recompute results/floors.json (needs the reference data in data/)
python src/make_fig1_setup.py     # Fig. 1 (needs the reference mesh in data/)
```

The first two commands need nothing beyond this repository. `floors.py` takes a few minutes, most of it the DMD of the cylinder field.

## Where each result comes from

| Paper | Code |
|---|---|
| Eq. (3) harmonic-balance basis | `hb()` |
| Eq. (5) DMD frequencies and basis | `dmd_freqs()`, `dmd_basis()` |
| Eq. (6) projection floor | `floor()` |
| Eq. (7) optimal floor | `opt_floor()` |
| Refined fundamental f0* | `refine()` |
| Eq. (9) phase-slip model | `energy_fractions()`, `predicted()` |
| Fig. 1 | `make_fig1_setup.py` |
| Fig. 2, Fig. 3, Fig. 4 | keys `fig2`, `fig3a` and `mech`, `fig4` of `floors.json` |
| Table 3 | `aggregate_trained.py` |

All three test cases agree with Eq. (9) to within 0.3 percentage points at every frequency error from 0 to 3 % (`mech` entries of `floors.json`).

## Reference data

`floors.py` expects three files in `data/`:

* `turek_hron_fsi2_reference_dense.npz` and `turek_hron_fsi3_reference_dense.npz`: Turek-Hron FSI2 and FSI3. Arrays `coordinates (n, 2)`, `topology (n_e, 3)`, `domain_markers (n_e,)` with 2 marking solid elements, `times (n_t,)`, `velocity (n_t, n, 2)` in m/s, `pressure (n_t, n)` in Pa and `displacement (n_t, n, 2)` in m.
* `cylinder_re100_vortex_reference.npz`: rigid cylinder at Re = 100. Arrays `times (n_t,)`, `U`, `V`, `P (n_t, n_y, n_x)` on a uniform grid with NaN inside the cylinder, `P_surf (n_t, 72)` and `theta_surf (72,)`.

The reference solutions are available from the corresponding author on request.

## Trained reconstruction

The physics-informed reconstruction behind Table 3 is part of the SPHINX framework and is not included in this release. Its evaluated results are provided in `results/trained_pod.csv` and `results/trained_hb_dmd.csv`. Each row gives one case, basis, number of temporal functions K, seed, region and field, with the global relative L2 error (%) and the floor of the basis the network was trained with.

## Citation

Please cite the paper above; `CITATION.cff` gives the metadata. Citation details will be updated on publication.

## Licence

MIT; see `LICENSE`.
