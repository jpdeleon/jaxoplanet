# Parameters

## Transit parameters

Each companion `<c>` (`b`, `c`, …) is sampled in jaxoplanet's native
`TransitOrbit` parameters, which the transit constrains directly:

| Name | Meaning |
|---|---|
| `<c>_period` | orbital period P (days) |
| `<c>_time_transit` | mid-transit time T0: time of minimum projected separation (BJD) |
| `<c>_duration` | total transit duration T14, first to fourth contact (days) |
| `<c>_impact_param` | impact parameter b |
| `<c>_radius_ratio` | radius ratio k = Rp/R⋆ |
| `<c>_f_c`, `<c>_f_s` | √e cos ω, √e sin ω (optional; default circular) |
| `<c>_K` | RV semi-amplitude (RV fits) |

The model maps them exactly onto a Keplerian orbit by inverting Winn (2010),
eqs. 7 and 14:

```text
b   = (a/R*) cos i · (1 - e²) / (1 + e sin ω)
T14 = P/π · asin( sqrt((1 + k)² - b²) / ((a/R*) sin i) ) · sqrt(1 - e²) / (1 + e sin ω)
```

so eccentric, curved and RV-consistent orbits are modelled exactly, not with a
straight-line approximation. allesfitter's quantities (rsuma, cos i, a/R⋆, i)
are still reported in the derived table.

### Why not allesfitter's rr, rsuma, cosi?

They are strongly correlated for a near-central transit, and cos i sits on a
boundary at 0. NUTS then needs very long trajectories. On the
[Kepler-1627 example](examples/kepler1627.md), the same short run
(4 chains, 150 warmup + 150 draws) gave:

| | rr, rsuma, cosi | radius_ratio, duration, impact_param |
|---|---|---|
| transit-shape r_hat | 1.58–1.76 | 1.012–1.035 |
| transit-shape ESS | ~3.5 | 57–75 |

## Converting an allesfitter params.csv

Files using `<c>_rr`, `<c>_rsuma`, `<c>_cosi` or `<c>_epoch` are rejected by
`validate`, which points to the converter:

```bash
uv run jaxoplanet convert-params my_fit/
```

It rewrites `params.csv` in place and keeps the original as `params.csv.orig`.

- Values and truths convert exactly.
- `rr` and `epoch` are renamed and keep their priors.
- A prior on rsuma or cosi has no equivalent in (duration, b), so the converted
  rows get broad uniform priors: duration in [T14/3, 3·T14] and impact_param in
  [0, 1 + k_max]. The command prints them. **Review them before fitting.**

## Other parameters

| Name | Meaning |
|---|---|
| `host_ldc_q1_<inst>`, `host_ldc_q2_<inst>` | quadratic limb darkening, Kipping (2013) q1, q2 |
| `ln_err_flux_<inst>`, `ln_jitter_rv_<inst>` | white noise (with `error_*_<inst>,sample`) |
| `baseline_*_<inst>` | offsets, polynomials and GP hyperparameters of the baseline set in `settings.csv` |
| `<c>_ttv_transit_<N>` | per-transit timing offsets (with `fit_ttvs,True`) |
