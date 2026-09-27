# Getting started

## Install

```bash
git clone https://github.com/jpdeleon/jaxoplanet2
cd jaxoplanet2
uv sync                    # or: pip install -e .
uv run jaxoplanet --help
```

Optional extras: `ns` for nested sampling (`jaxoplanet ns-fit`, via jaxns) and
`mkdocs` to build this site.

## Set up a fit directory

```bash
uv run jaxoplanet init my_fit/
```

writes two commented templates. Then add one data file per instrument named in
`settings.csv`.

### `<inst>.csv`: data

Three columns without a header, `time,y,yerr` (BJD, relative flux or RV,
uncertainty). Lines starting with `#` are comments. This is allesfitter's
format.

### `settings.csv`: what to fit and how

The template, as written by `jaxoplanet init`:

```text
--8<-- "src/jaxoplanet2/templates/settings.csv"
```

allesfitter keys keep their meaning. `mcmc_nwalkers` sets the number of NUTS
chains and `mcmc_burn_steps` the warmup. Keys starting with `jx_` are
jaxoplanet2-only. Unsupported keys are an error (listed by `validate`), never a
silent skip; `--allow-unsupported` turns them into warnings.

### `params.csv`: parameters and priors

```text
--8<-- "src/jaxoplanet2/templates/params.csv"
```

Columns: `name,value,fit,bounds,label,unit,truth`. Rows with `fit=1` are
sampled with the prior in `bounds`: `uniform lo hi`, `normal mu sd`,
`trunc_normal lo hi mu sd` or `loguniform lo hi`. Rows with `fit=0` are fixed.
See [Parameters](parameters.md) for the transit parameters.

## Run it

```bash
uv run jaxoplanet validate my_fit/
uv run jaxoplanet show-initial-guess my_fit/
uv run jaxoplanet optimize my_fit/
uv run jaxoplanet mcmc-fit my_fit/
```

- `validate` checks the settings, the parameters and the data, and evaluates
  the likelihood once at the starting values.
- `optimize` maximises the posterior and, if the fit improves, writes the result
  back into `params.csv` (the original is kept as a backup). `--no-update` leaves
  `params.csv` alone.
- `mcmc-fit` runs NUTS and then writes the tables and plots. Run
  `mcmc-output` to regenerate them from saved samples.

## Outputs

Everything goes to `my_fit/results/`:

| File | Written by |
|---|---|
| `initial_guess_<inst>.pdf` | `show-initial-guess` |
| `params_optimized.csv`, `optimize_save.json` | `optimize` |
| `mcmc_samples.npz`, `mcmc_diagnostics.txt` (r_hat, ESS, divergences) | `mcmc-fit` |
| `mcmc_table.csv`, `mcmc_derived_table.csv`, `mcmc_latex_table.txt` | `mcmc-fit` / `mcmc-output` |
| `mcmc_corner.pdf`, `mcmc_fit_<inst>.pdf` | `mcmc-fit` / `mcmc-output` |
| `ns_samples.npz`, `ns_table.csv`, `ns_diagnostics.txt` (ln Z) | `ns-fit` / `ns-output` |

The derived table reports the geometry (a/R⋆, i, rsuma, cos i, T14, T23, ρ⋆)
and, with a `params_star.csv`, physical radii, masses, a and Teq.
