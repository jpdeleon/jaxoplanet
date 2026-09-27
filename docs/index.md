# jaxoplanet2

_[allesfitter](https://github.com/MNGuenther/allesfitter)-style transit and RV
fitting, powered by [jaxoplanet](https://jax.exoplanet.codes) + NumPyro._

A fit is a directory of plain files: `settings.csv`, `params.csv` and one
`<inst>.csv` per instrument. The `jaxoplanet` command runs the whole analysis
from them:

```bash
jaxoplanet init my_fit/                 # template settings.csv + params.csv
jaxoplanet validate my_fit/             # check settings, params and data
jaxoplanet show-initial-guess my_fit/   # plot the model at the params.csv values
jaxoplanet optimize my_fit/             # posterior maximum, warm start for MCMC
jaxoplanet mcmc-fit my_fit/             # NUTS; tables, corner and fit plots
```

The model is differentiable end to end (JAX), so sampling uses NUTS with a
dense mass matrix instead of affine-invariant walkers, and it runs on CPU or
GPU.

## Where to go next

- [Getting started](getting-started.md): install, set up a fit directory, run
  it, and find the outputs.
- [Parameters](parameters.md): the sampled transit parameters, and how to
  convert an allesfitter `params.csv`.
- [Kepler-1627 b](examples/kepler1627.md): a worked example on real Kepler data
  (transit + starspot GP), with measured run times and convergence.
- [CLI reference](cli.md) and [API reference](api/index.md): generated from the
  code at every build.
- [Design plan](plans/cli-fitter.md): the original plan and its decisions.

## Relationship to jaxoplanet

This repository is a hard fork of
[exoplanet-dev/jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet). It
ships the upstream `jaxoplanet` library unchanged, plus the `jaxoplanet2`
fitter. The library's own documentation (orbits, light curves, limb darkening,
tutorials) is at [jax.exoplanet.codes](https://jax.exoplanet.codes).

!!! warning "Do not install alongside upstream jaxoplanet"
    The `jaxoplanet2` distribution provides the `jaxoplanet` module. Installing
    it in the same environment as the upstream `jaxoplanet` distribution makes
    the two overwrite each other's files.
