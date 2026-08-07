# Installation

## Installing PyMC-Marketing

PyMC-Marketing requires **Python 3.12 or greater**.

PyMC-Marketing is built on top of **PyMC >= 6.0** and **ArviZ >= 1.2**.

Install the `pymc-marketing` package with pip:

```bash
pip install pymc-marketing
```

Some features are available as optional extras:

```bash
pip install pymc-marketing[dag]  # causal identification tooling
pip install pymc-marketing[pie]  # Predicted Incrementality by Experimentation (PIE), requires pymc-bart
```

You can also install the development version of PyMC-Marketing with:

```bash
pip install git+https://github.com/pymc-labs/pymc-marketing.git
```

Next, you can create a new Jupyter notebook with either JupyterLab or VS Code.

### JupyterLab Notebook

After installing the `pymc-marketing` package, install JupyterLab and launch it:

```bash
pip install jupyterlab
jupyter lab
```

### VS Code Notebook

After installing the `pymc-marketing` package, install ipykernel:

```bash
pip install ipykernel
```

Start VS Code and ensure that the "Jupyter" extension is installed. Press Ctrl + Shift + P and type "Python: Select Interpreter". Press Ctrl + Shift + P and type "Create: New Jupyter Notebook".

## Installation for developers

If you are a developer of pymc-marketing, or want to start contributing, [refer to the contributing guide](https://github.com/pymc-labs/pymc-marketing/blob/main/CONTRIBUTING.md) to get started.

See the official [PyMC installation guide](https://www.pymc.io/projects/docs/en/latest/installation.html) if more detail is needed.
