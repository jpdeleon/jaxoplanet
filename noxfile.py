# mypy: ignore-errors

import nox

ALL_PYTHON_VS = ["3.11", "3.12", "3.13", "3.14"]


@nox.session(python=ALL_PYTHON_VS)
def test(session):
    session.install(".[test,test-math]")
    session.run("pip", "freeze")
    session.run("pytest", "-n", "auto", *session.posargs)


@nox.session(python=ALL_PYTHON_VS)
def test_x64(session):
    session.install(".[test,test-math]")
    session.run("pip", "freeze")
    env = {"JAX_ENABLE_X64": "1"}
    session.run("pytest", "-n", "auto", *session.posargs, env=env)


@nox.session
def docs(session):
    session.install(".[docs]")
    with session.chdir("docs"):
        session.run(
            "python",
            "-m",
            "sphinx",
            "-T",
            "-E",
            "-W",
            "--keep-going",
            "-b",
            "dirhtml",
            "-d",
            "_build/doctrees",
            "-D",
            "language=en",
            ".",
            "_build/dirhtml",
        )


@nox.session(python=["3.11", "3.13"])
def jaxoplanet2(session):
    """jaxoplanet2 fitter tests with the 80% coverage gate."""
    session.install(".[test]", "pytest-cov")
    session.run(
        "pytest",
        "tests/jaxoplanet2",
        "--cov=jaxoplanet2",
        "--cov-report=term-missing",
        "--cov-fail-under=80",
        *session.posargs,
        env={"JAX_ENABLE_X64": "1", "MPLBACKEND": "Agg"},
    )
