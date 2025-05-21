FROM jupyter/base-notebook:python-3.11.6

LABEL name="pymc-marketing"
LABEL description="Environment for PyMC-Marketing"

# Install conda environment
COPY environment-dev.yml /tmp/

RUN mamba env update -n base -f /tmp/environment-dev.yml && \
    conda clean --all -f -y && \
    fix-permissions $CONDA_DIR
