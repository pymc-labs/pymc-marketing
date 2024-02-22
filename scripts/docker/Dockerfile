FROM jupyter/base-notebook:python-3.11.6

LABEL name="pymc-marketing"
LABEL description="Environment for PyMC Marketing"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Switch to jovyan to avoid container runs as root
USER $NB_UID

# Install conda environment
COPY /scripts/docker/environment-dev.yml .
RUN mamba env create -f environment-dev.yml && \
    /bin/bash -c ". activate pymc-marketing-dev" && \
    conda clean --all -f -y

# Create kernel
RUN $CONDA_DIR/envs/pymc-marketing-dev/bin/python -m ipykernel install --user --name=pymc-marketing-dev

# Prepend conda environment to path
ENV PATH $CONDA_DIR/envs/pymc-marketing-dev/bin:$PATH

# If you want this environment to be the default one, uncomment the following line:
ENV CONDA_DEFAULT_ENV pymc-marketing-dev

# Setup working folder
WORKDIR /home/jovyan/work

# For running from bash
SHELL ["/bin/bash","-c"]
RUN echo "conda activate pymc-marketing-dev" >> ~/.bashrc && \
    source ~/.bashrc

# For running from jupyter notebook
EXPOSE 8888
CMD ["conda", "run", "--no-capture-output", "-n", "pymc-marketing-dev", "jupyter","notebook","--ip=0.0.0.0","--port=8888","--no-browser"]
