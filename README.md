# jaxoplanet2

_[allesfitter](https://github.com/MNGuenther/allesfitter)-style transit and RV fitting, powered by jaxoplanet_

---

**jaxoplanet2** is a hard fork of
[exoplanet-dev/jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet). It ships
two import packages:

- `jaxoplanet`: the upstream library, vendored unchanged and synced regularly
  from upstream.
- `jaxoplanet2`: a directory-driven fitter that reads allesfitter's
  `params.csv`, `settings.csv` and `<inst>.csv` files and runs them with
  jaxoplanet + NumPyro:

```bash
uv run jaxoplanet show-initial-guess my_fit/
uv run jaxoplanet optimize my_fit/
uv run jaxoplanet mcmc-fit my_fit/
```

> [!WARNING]
> The `jaxoplanet2` distribution provides the `jaxoplanet` module. Do not install
> it alongside the upstream `jaxoplanet` distribution in the same environment,
> because the two would overwrite each other's files.

See [issue #1](https://github.com/jpdeleon/jaxoplanet2/issues/1) for the roadmap.
The upstream jaxoplanet README follows.

---

# jaxoplanet

_Astronomical time series analysis with JAX_

---

*jaxoplanet* is a
[functional-programming](https://en.wikipedia.org/wiki/Functional_programming)-forward
implementation of many features from the
[exoplanet](https://docs.exoplanet.codes/en/latest/) and
[starry](https://starry.readthedocs.io/en/latest/) packages built on top of
[JAX](https://jax.readthedocs.io/en/latest/).

*jaxoplanet* includes fast and robust implementations of many exoplanet-specific
operations, including solving Kepler's equation, and computing limb-darkened
light curves. Since *jaxoplanet* is built on top of JAX it has first-class
support for hardware acceleration using GPUs and TPUs, and it also integrates
seamlessly with modeling tools like
[NumPyro](https://numpyro.readthedocs.io/en/latest/), and
[Flax](https://flax.readthedocs.io/en/latest/).

**For the most complete documentation, check out the documentation page at
[jax.exoplanet.codes](https://jax.exoplanet.codes).**

## Installation

You'll first need to install JAX following [the instructions in the JAX
docs](https://jax.readthedocs.io/en/latest/#installation). For example, to
install the CPU version of JAX, you can run:

```bash
python -m pip install "jax[cpu]"
```

Then install `jaxoplanet` with:

```bash
python -m pip install jaxoplanet
```

If you run into issues with installing `jaxoplanet`, specifically on ARM (i.e., M series chips) Macs, take a look at [the
installation instructions](https://jax.exoplanet.codes/en/latest/install).

```{admonition} Navigating the docs
:class: tip

💽 After [installing](doc:install) `jaxoplanet`, head over to the [Quickstart](quickstart) page to see some of its features.

🚩 If you're running into some problems with `jaxoplanet`, check out the [common issues](doc:commonissues) page for some general tips and tricks.

⚡ In the [Tutorials](tutorials) section we've added a few tutorials showcasing some common astronomy problems where `jaxoplanet` might come in handy!

🖥️ Take a look [here](doc:api) to see a user-friendly API reference for commonly used objects. The full API reference is also available!

```

## Attribution

While we don't yet have a citation for `jaxoplanet`, please reference the GitHub
repository if you find this code useful in your research. The BibTeX entry for
the repo is:

```
@software{jaxoplanet,
  author       = {Soichiro Hattori and
                  Lionel Garcia and
                  Catriona Murray and
                  Jiayin Dong and
                  Shashank Dholakia and
                  David Degen and
                  Daniel Foreman-Mackey},
  title        = {{exoplanet-dev/jaxoplanet: Astronomical time series analysis with JAX}},
  month        = mar,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.0.2},
  doi          = {10.5281/zenodo.10736936},
  url          = {https://doi.org/10.5281/zenodo.10736936}
}
```

## License
Copyright (c) 2021-2024 Simons Foundation, Inc.

`jaxoplanet` is free software made available under the MIT License. For details
see the `LICENSE` file.
