# Plan: jaxoplanet2, an allesfitter-style fitter on a jaxoplanet hard fork

Branch: `feat/cli-fitter`

## Decisions

| Question | Decision |
|---|---|
| Relationship to upstream | **Hard fork.** This repo is renamed `jaxoplanet` → `jaxoplanet2`. It keeps the vendored `jaxoplanet` library source, which is still synced from `upstream/main`, and adds a new `jaxoplanet2` package that holds the fitter. |
| Distribution name | `jaxoplanet2` (on PyPI and in `pyproject.toml`) |
| Import names | `jaxoplanet` for the vendored library (unchanged), `jaxoplanet2` for the fitter |
| CLI command | `jaxoplanet` (`uv run jaxoplanet show-initial-guess .`) |
| `mcmc_nwalkers` | Mapped to the NUTS chain count, capped by the device count. `jx_num_chains` overrides it. |

## Goal

Run a whole transit/RV fit from a directory of plain files, the way
[allesfitter](https://github.com/MNGuenther/allesfitter) does, but with
jaxoplanet + numpyro (JAX, autodiff, NUTS) as the engine:

```bash
uv run jaxoplanet init my_fit/                 # scaffold params.csv + settings.csv
uv run jaxoplanet validate my_fit/             # parse & sanity-check inputs
uv run jaxoplanet show-initial-guess my_fit/   # plot model at params.csv values
uv run jaxoplanet optimize my_fit/             # MAP fit -> params_optimized.csv
uv run jaxoplanet mcmc-fit my_fit/             # NUTS sampling -> results/
uv run jaxoplanet mcmc-output my_fit/          # tables, corner, fit plots
uv run jaxoplanet ns-fit my_fit/               # nested sampling (phase 4)
uv run jaxoplanet ns-output my_fit/
```

Each command also has a Python equivalent that mirrors allesfitter's
module-level functions, so existing `run.py` scripts port by changing one import:

```python
import jaxoplanet2 as jf
jf.show_initial_guess(".")
jf.optimize(".")
jf.mcmc_fit(".")
jf.mcmc_output(".")
```

### Non-goals (for now)

- Full feature parity with allesfitter (flares, spots, phase curves,
  secondary eclipses, stellar grids, `ellc` eclipsing binaries).
- Changing the vendored `jaxoplanet` package. The fitter only *uses* it, so
  upstream merges stay conflict-free outside `pyproject.toml` and `README.md`.
  Any fix needed in `jaxoplanet` itself goes upstream as a PR first.

## Design principles

1. **Drop-in input compatibility.** An existing allesfitter directory
   (`params.csv`, `settings.csv`, `<inst>.csv`) should parse unchanged.
   Settings and parameters we don't support fail loudly with a list of the
   offending keys (`--allow-unsupported` downgrades to warnings), never silently.
2. **Fork hygiene.** All new code lives in `src/jaxoplanet2/` and `tests/jaxoplanet2/`.
   The only upstream-owned files touched are `pyproject.toml` and `README.md`, so a
   `git merge upstream/main` conflicts in at most those two.
3. **Pure core, thin shell.** The parsing, then model building, then inference
   steps are pure functions on frozen dataclasses. The CLI only does argument
   parsing and file I/O.
4. **Small modules.** Aim for 200 to 400 lines per file, as the layout below
   shows.

## Package layout

```
src/jaxoplanet/          # vendored upstream library, untouched
src/jaxoplanet2/
├── __init__.py          # public API: show_initial_guess, optimize, mcmc_fit, ...
├── cli.py               # argparse subcommands -> public API
├── io/
│   ├── settings.py      # settings.csv -> Settings (frozen dataclass)
│   ├── params.py        # params.csv  -> ParamTable (frozen)
│   ├── priors.py        # "uniform a b" / "normal mu sd" / "trunc_normal ..." -> numpyro dist
│   ├── data.py          # <inst>.csv -> Dataset(time, y, yerr) per instrument
│   └── writers.py       # params_optimized.csv, summary tables, LaTeX
├── model/
│   ├── parameterization.py  # allesfitter params -> jaxoplanet Central/Body
│   ├── photometry.py        # limb_dark_light_curve, exposure integration, TTVs
│   ├── rv.py                # Keplerian RV via radial_velocity_semiamplitude
│   ├── baseline.py          # offset / polynomial / GP (tinygp) baselines
│   ├── errors.py            # ln_err_flux_*, ln_jitter_rv_* handling
│   └── numpyro_model.py     # assembles the full numpyro model from Settings+ParamTable
├── infer/
│   ├── optimize.py      # MAP via numpyro_ext.optim (fallback: jaxopt/optax)
│   ├── mcmc.py          # numpyro NUTS, chains, convergence diagnostics
│   └── nested.py        # numpyro.contrib.nested_sampling (jaxns), phase 4
├── plots/
│   ├── initial_guess.py # full LC/RV + phase-folded per companion per instrument
│   ├── fit.py           # posterior model draws over data
│   └── corner.py
└── templates/
    ├── params.csv
    └── settings.csv
```

`pyproject.toml` changes:

```toml
[project]
name = "jaxoplanet2"
dependencies = ["jax", "jaxlib", "equinox",            # upstream's
                "numpy", "numpyro>=0.21", "numpyro-ext", "tinygp",
                "matplotlib", "corner", "arviz>=1.0", "pandas"]

[project.scripts]
jaxoplanet = "jaxoplanet2.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/jaxoplanet", "src/jaxoplanet2"]

[tool.hatch.version]
source = "vcs"
raw-options = { tag_regex = "^jaxoplanet2-v(?P<version>.*)$", git_describe_command = "git describe --tags --match 'jaxoplanet2-v*'" }
```

The fitter is the whole point of the distribution, so its dependencies are core
rather than an extra. Use `argparse` (stdlib) rather than adding a CLI dependency.

**Version tags:** the fork inherits upstream tags such as `v0.1.0`. jaxoplanet2
releases are tagged `jaxoplanet2-vX.Y.Z` so the two version lines never mix.

**Installation conflict:** `jaxoplanet2` provides the `jaxoplanet` module.
Installing it alongside `pip install jaxoplanet` in one environment makes the two
overwrite each other. The README states this, and `jaxoplanet2.__init__` warns if
`importlib.metadata` reports that the `jaxoplanet` distribution is also installed.

**Upstream sync runbook** (to `docs/plans/` → later `CONTRIBUTING.md`):

```bash
git fetch upstream
git switch main && git merge upstream/main   # conflicts only in pyproject.toml/README.md
uv run pytest tests/jaxoplanet2               # the fitter still works on the new core
```

## File formats

### `settings.csv`: allesfitter-compatible `name,value`

Lines starting with `#` are comments. The keys supported per phase:

| Key | Phase | Notes |
|---|---|---|
| `companions_phot`, `companions_rv` | 1 | space-separated letters |
| `inst_phot`, `inst_rv` | 1 | each maps to `<inst>.csv` |
| `host_ld_law_<inst>` | 1 | `quad` → jaxoplanet quadratic LD (q1,q2 Kipping) |
| `error_flux_<inst>`, `error_rv_<inst>` | 1 | `sample` → fit `ln_err_*`, `hybrid` (use file errors) |
| `fast_fit`, `fast_fit_width` | 1 | mask data to windows around transits |
| `t_exp_<inst>`, `t_exp_n_int_<inst>` | 1 | `jaxoplanet.light_curves.transforms.integrate` |
| `baseline_flux_<inst>`, `baseline_rv_<inst>` | 2 | `sample_offset`, `sample_linear`, `hybrid_poly_N`, `sample_GP_Matern32`, `sample_GP_SHO`, `sample_GP_real` (tinygp) |
| `use_host_density_prior` | 2 | adds ρ⋆ prior from `params_star.csv` |
| `shift_epoch`, `inst_for_<c>_epoch` | 2 | re-reference T0 to the data midpoint |
| `fit_ttvs` | 3 | `jaxoplanet.orbits.TTVOrbit` |
| `mcmc_nwalkers`, `mcmc_total_steps`, `mcmc_burn_steps`, `mcmc_thin_by` | 2 | mapped to NUTS: `num_chains` (capped by devices), `num_samples = total-burn`, `num_warmup = burn`, `thinning` |
| `ns_*` | 4 | mapped to jaxns where possible |
| `multiprocess*` | — | accepted and ignored (JAX handles parallelism) |

New jaxoplanet-only keys, prefixed so they never collide:

| Key | Default | Meaning |
|---|---|---|
| `jx_backend` | `cpu` | `cpu` / `gpu` |
| `jx_x64` | `True` | `jax_enable_x64` |
| `jx_nuts_target_accept` | `0.9` | |
| `jx_nuts_max_tree_depth` | `10` | |
| `jx_num_chains` | `2` | overrides the mapping from `mcmc_nwalkers` |
| `jx_seed` | `42` | |
| `jx_optimizer` | `numpyro_ext` | `numpyro_ext` / `jaxopt_lbfgs` / `optax_adam` |

### `params.csv`: allesfitter columns

`#name,value,fit,bounds,label,unit,truth`

- `fit=1`: becomes a sampled numpyro site. Its prior comes from `bounds`.
- `fit=0`: a fixed constant.
- `bounds` grammar: `uniform lo hi`, `normal mu sd`,
  `trunc_normal lo hi mu sd`. Phase 3 adds `loguniform lo hi`.
- Supported names in phase 1 and 2: `<c>_rr`, `<c>_rsuma`, `<c>_cosi`,
  `<c>_epoch`, `<c>_period`, `<c>_K`, `<c>_f_c`, `<c>_f_s`,
  `host_ldc_q1_<inst>`, `host_ldc_q2_<inst>`, `ln_err_flux_<inst>`,
  `ln_jitter_rv_<inst>`, `baseline_*_<inst>`, `<c>_ttv_transit_N`.

### Data files: `<inst>.csv`

Three columns with no header: `time,y,yerr`. Lines starting with `#` are
ignored. This is identical to allesfitter.

## Parameterization mapping (the risky part)

> **Superseded:** the fitter now samples jaxoplanet's native `TransitOrbit`
> parameters (`<c>_radius_ratio`, `<c>_duration`, `<c>_impact_param`,
> `<c>_time_transit`, `<c>_period`) and maps them exactly to a Keplerian orbit by
> inverting Winn (2010) eqs. 7 and 14 (see `jaxoplanet2.model.parameterization`).
> Legacy files are converted with `jaxoplanet convert-params <dir>`. The
> original allesfitter mapping is kept below for reference.

allesfitter's parameters are converted to jaxoplanet's `Central` and `Body` inside
the numpyro model, so gradients flow through:

```
a/R⋆     = (1 + rr) / rsuma
e        = f_c² + f_s²
ω        = atan2(f_s, f_c)
cos i    = cosi
ρ⋆       = 3π (a/R⋆)³ / (G P²)          -> Central(radius=1, density=ρ⋆)
Body(period=P, time_transit=epoch, inclination=arccos(cosi),
     eccentricity=e, omega_peri=ω, radius=rr,
     radial_velocity_semiamplitude=K)
```

In the circular, photometry-only case, use the cheaper `TransitOrbit`, with
duration and impact parameter derived from the values above.

**Validation gate:** run a golden test against allesfitter's own model for the
same parameter vector (for example the TOI-2427 example in `DONE/`), requiring
the light curve to agree within 1 ppm and RV within 1 cm/s. Nothing downstream
starts until this passes.

## Output layout (mirrors allesfitter's `results/`)

```
results/
├── initial_guess_<c>.pdf, initial_guess_<inst>_<c>.pdf
├── params_optimized.csv        # optimize: same schema as params.csv
├── optimize_summary.txt        # loss, grad norm, n_iter
├── mcmc_samples.nc             # arviz InferenceData (netCDF)
├── mcmc_table.csv / .tex       # median ±1σ per param + derived params
├── mcmc_corner.pdf, mcmc_fit_<inst>.pdf
├── mcmc_diagnostics.txt        # r_hat, ESS, divergences
└── logfile_<timestamp>.log
```

The optional flag `optimize --update-params` writes the MAP values back into
`params.csv`. It keeps a timestamped `params.csv.bak` and never overwrites
silently.

Derived parameters to report (phase 2): Rp in R⊕/RJ (from `params_star.csv`),
a in au, i in degrees, b, T14, T23, ρ⋆, e, ω, Teq, Mp from K.

## Phases

### Phase 0: scaffolding (about 0.5 day)
- Rename the distribution to `jaxoplanet2` in `pyproject.toml`, add the `src/jaxoplanet2/` package, and make wheel packaging and version tags follow the Package layout section.
- Add the `jaxoplanet` entry point, `jaxoplanet --help`, and `init` with templates.
- Tests: running the CLI and `init` creates valid files that round-trip through the parsers.

### Phase 1: IO + show-initial-guess (about 2 days)
- `settings.py`, `params.py`, `priors.py`, `data.py` with a strict validation
  that reports unsupported keys.
- `parameterization.py`, `photometry.py`, `rv.py`, and the white-noise `errors.py`.
- `show-initial-guess`, and `validate`.
- Tests (TDD, write first): parser unit tests including malformed rows,
  prior grammar tests, and the **allesfitter golden-model test**.

### Phase 2: optimize + MCMC (about 3 days)
- `numpyro_model.py`, `optimize.py`, `mcmc.py`, and the offset/poly/GP baselines,
  host density prior and `shift_epoch`.
- `mcmc-output`: tables, corner, fit plots, diagnostics.
- Tests: an injection-recovery on synthetic single-planet TESS+RV data
  (truth inside the 2σ interval, r_hat < 1.01), and an end-to-end CLI run on a
  tiny fixture directory marked `@pytest.mark.slow`.

### Phase 3: TTVs + extras (about 2 days)
- `fit_ttvs` via `TTVOrbit`, `loguniform` priors, and more GP kernels.
- Porting at least one of the local `docs/tutorials/*_using_jaxoplanet.ipynb`
  notebooks to the directory-based workflow as a docs example.

### Phase 4: nested sampling (optional, about 2 days)
- `ns-fit` / `ns-output` through `numpyro.contrib.nested_sampling` (jaxns),
  reporting ln Z for model comparison.

## Testing strategy

- `tests/jaxoplanet2/`, pytest, aiming for at least 80% coverage on `jaxoplanet2`.
  The upstream `tests/` suite keeps running unchanged to catch regressions from merges.
- Unit tests: parsers, priors, the parameterization math (compare against
  analytic formulas).
- Integration tests: the numpyro model's log density at a known point matches a
  hand-computed value.
- Golden test: allesfitter parity (skipped if `allesfitter` is not installed).
- End-to-end test: the CLI subcommands on a fixture directory (slow marker).
- Add a `fit` session to `noxfile.py`.

## Risks and open questions

1. **Parameterization parity.** Mismatches in allesfitter's conventions for
   rsuma, the ω sign, epoch reference and LD q→u are the likeliest source of
   subtle bias. The golden test is there to catch this.
2. **Walkers vs NUTS (decided).** `mcmc_nwalkers` maps to the chain count, capped
   by devices. The log and `mcmc_diagnostics.txt` record the mapping so users of
   old settings files aren't surprised.
3. **Fork drift.** If the fork falls behind, merges get painful. Sync with upstream
   at least monthly, and stay off the vendored `jaxoplanet/` code (design principle 2).
4. **Module shadowing.** See the installation conflict above. Revisit a
   dependency-based layout if users need both installed side by side.
5. **GP cost.** Full-cadence TESS with a GP is O(N) with tinygp's quasisep
   kernels. Require quasisep kernels (Matern32, SHO, Celerite) and reject dense
   ones for N > 10⁴.
6. **Float64.** Enable x64 by default. Epochs around 2.46e6 lose precision
   in float32. Also subtract a reference time internally.
