# Kepler-1627 b with jaxoplanet2: transit + starspot GP from a fit directory

This is the directory-driven version of
`docs/tutorials/lc-gp-transit_using_jaxoplanet.ipynb`: the same data (Kepler
long cadence of KIC 6184894 = Kepler-1627) and the same idea (fit the transit
and the starspot variability jointly, no detrending), but written as
allesfitter-style `settings.csv` + `params.csv` and run from the command line.

| notebook | fit directory |
|---|---|
| numpyro model in Python | `settings.csv` + `params.csv` |
| quasi-periodic rotation kernel (tinygp) | `baseline_flux_kepler,sample_GP_SHO` (allesfitter's SHO GP) |
| `(duration, t0, ror, b)` sampled | allesfitter's `(rr, rsuma, cosi, epoch, period)` |
| long-cadence integration | `t_exp_kepler`, `t_exp_n_int_kepler` |
| MAP with `numpyro_ext` | `jaxoplanet optimize` |
| NUTS + arviz summary | `jaxoplanet mcmc-fit` (tables and plots written automatically) |

## Run it

```bash
cd examples/jaxoplanet2/kepler1627
python prepare_data.py               # needs lightkurve + network; writes kepler.csv
jaxoplanet validate .                 # settings, params, data, initial likelihood
jaxoplanet show-initial-guess .       # results/initial_guess_kepler.pdf
jaxoplanet optimize . -n 3            # posterior maximum, rewrites params.csv
jaxoplanet mcmc-fit .                 # NUTS; then mcmc_table.csv, corner, fit plots
```

`fast_fit` keeps only the +-0.25 d around each of the ~200 transits: 3194 of the
46055 cadences. The SHO GP then models the starspot signal inside those windows,
exactly as allesfitter would with the same settings.

## What to expect (measured when this example was written)

- `validate`: 3194 flux points, 12 free parameters, `OK`.
- `optimize -n 3`: ln posterior 16673.2 -> 19546.5 in 93 evaluations (6.3 s of
  optimisation, ~48 s wall time including JIT on a 128-core Linux box); the
  optimum has Rp/R* = 0.0378, P = 7.202804 d and an SHO frequency
  omega0 = e^0.92, i.e. a ~2.5 d variability period, consistent with the star's
  ~2.6 d rotation.

- `mcmc-fit` with 4 parallel chains of only 150 warmup + 150 draws took
  **2 h 46 min** and did **not** converge for the transit shape: r_hat 1.58-1.76
  (ESS ~3.5) for rr, rsuma and cosi, while epoch, period, the white noise and all
  four GP parameters converged (r_hat <= 1.004, ESS 390-715). Medians:
  Rp/R* = 0.0398 (-0.0014 / +0.0029), P = 7.2028039 +- 0.0000050 d,
  ln omega0 = 0.922 (-0.020 / +0.025), b = 0.55 +- 0.27, T14 = 2.86 h.

  Why: each gradient costs ~72 ms here (10-point exposure integration of 3194
  cadences ~37 ms; the SHO GP roughly doubles it), and the near-central transit
  has the classic rr-b-rsuma degeneracy, so NUTS takes ~500 leapfrog steps per
  draw while its mass matrix is still adapting. Budget a long run (the shipped
  `mcmc_burn_steps,500` / `mcmc_total_steps,1000`) on a many-core machine.

## A note on the notebook's ephemeris

The notebook downloads KIC 6184894 (Kepler-1627) but uses P = 15.335381 d and
calls the planet Kepler-27b. On these data, a BLS search (after a 25-cadence
flatten) finds the transit at **P = 7.202826 d, T0 = 120.79163 BKJD
(BJD 2454953.79163), depth 910 ppm**, while the 15.2-15.5 d range has no
comparable signal (its peak sits at the edge of the range, at 15.219 d). This
example uses the ephemeris measured from the data.
