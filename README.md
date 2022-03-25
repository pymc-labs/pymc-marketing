# pymmmc

![Build](https://github.com/pymc-labs/pymmmc/workflows/ci/badge.svg)

## Bayesian MMMs in PyMC

...

---

## Local Development

1. Create conda environment. For example:

```shell
conda create -n pymmmc_env python=3.8
```

2. Activate environment.

```shell
conda activate pymmmc_env
```

3. Install `pymmmc` package:

```shell
make init
```

4. To run tests:

```shell
make test
```

5. To check code style:

```shell
make check_lint
```

6. Set [pre-commit hooks](https://pre-commit.com/) (Optional):

```shell
pre-commit install
```
