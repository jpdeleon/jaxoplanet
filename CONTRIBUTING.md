# Contributor Guide

Thank you for your interest in improving this project. This project is
open-source and welcomes contributions in the form of bug reports, feature
requests, and pull requests.

Here is a list of important resources for contributors:

- [Source Code](https://github.com/exoplanet-dev/jaxoplanet)
- [Documentation](https://jax.exoplanet.codes)
- [Issue Tracker](https://github.com/exoplanet-dev/jaxoplanet/issues)

## How to report a bug

Report bugs on the [Issue Tracker](https://github.com/exoplanet-dev/jaxoplanet/issues).

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case, and/or steps to
reproduce the issue. In particular, please include a [Minimal, Reproducible
Example](https://stackoverflow.com/help/minimal-reproducible-example).

## How to request a feature

Feel free to request features on the [Issue
Tracker](https://github.com/exoplanet-dev/jaxoplanet/issues).

## How to test the project

```bash
python -m pip install nox
python -m nox -s test
```

## How to submit changes

Open a [Pull Request](https://github.com/exoplanet-dev/jaxoplanet/pulls).

## jaxoplanet2 (this fork)

All jaxoplanet2 code lives in `src/jaxoplanet2/` and `tests/jaxoplanet2/`. The
vendored `src/jaxoplanet/` is never edited here. Fixes to it go upstream as PRs
first and arrive through the sync below.

### Testing the fitter

```bash
python -m nox -s jaxoplanet2        # tests + 80% coverage gate
JAXOPLANET2_SLOW=1 python -m nox -s jaxoplanet2 -- -m slow   # end-to-end fits
```

### Syncing with upstream jaxoplanet

Do this at least monthly so the fork doesn't drift:

```bash
git remote add upstream https://github.com/exoplanet-dev/jaxoplanet  # once
git fetch upstream
git switch main
git merge upstream/main
```

Conflicts are expected only in files the fork deliberately changed:
`pyproject.toml`, `README.md`, `CONTRIBUTING.md` (this section) and `noxfile.py`
(the `jaxoplanet2` session). In each case keep both sides: upstream's changes
plus the jaxoplanet2 additions. Then verify and push:

```bash
python -m nox -s jaxoplanet2
python -m nox -s test-3.13          # upstream suite on the merged core
git push origin main
```

If an upstream API change breaks `jaxoplanet2`, fix `src/jaxoplanet2` in the same
merge PR. Never patch `src/jaxoplanet`.

### Releasing

Tag releases as `jaxoplanet2-vX.Y.Z`. Only those tags drive the version, and the
upstream `vX.Y.Z` tags inherited by the fork are ignored.
