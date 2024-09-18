## Installing PyMC-Marketing

PyMC-Marketing requires **Python 3.10 or greater**.

Install and activate an environment (e.g. `marketing_env`) with the `pymc-marketing` package from [conda-forge](https://conda-forge.org). It may look something like the following:

```bash
conda create -c conda-forge -n marketing_env pymc-marketing
conda activate marketing_env
```

You can also install the development version of PyMC-Marketing with:

```bash
pip install git+https://github.com/pymc-labs/pymc-marketing.git
```

Next, we you can create a new Jupyter notebook with either JupyterLab or VS Code.

### JupyterLab Notebook

After installing the `pymc-marketing` package (see above), run the following with `marketing_env` activated:

```bash
conda install -c conda-forge jupyterlab
jupyter lab
```

### VS Code Notebook

After installing the `pymc-marketing` package (see above), run the following with `marketing_env` activated:

```bash
conda install -c conda-forge ipykernel
```

Start VS Code and ensure that the "Jupyter" extension is installed. Press Ctrl + Shift + P and type "Python: Select Interpreter". Ensure that `marketing_env` is selected. Press Ctrl + Shift + P and type "Create: New Jupyter Notebook".

## Installation for developers

If you are a developer of pymc-marketing, or want to start contributing, [refer to the contributing guide](https://github.com/pymc-labs/pymc-marketing/blob/main/CONTRIBUTING.md) to get started.

See the official [PyMC installation guide](https://www.pymc.io/projects/docs/en/latest/installation.html) if more detail is needed.
