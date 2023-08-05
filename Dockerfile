# Start with the official Debian base image
FROM jupyter/minimal-notebook:latest
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Update the system and install necessary packages
RUN apt-get update && apt-get install -y \
    curl \
    bzip2 \
    build-essential

# Set the working directory
WORKDIR /app

# Download and install Miniconda (ARM64 version)
RUN curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
RUN bash Miniforge3-Linux-aarch64.sh -p /miniconda -b
RUN rm Miniforge3-Linux-aarch64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# Create a new Conda environment and install a package
RUN conda create -n pymc_marketing python=3.10 -y && \
    conda init bash && \
    echo "conda activate pymc_marketing" >> ~/.bashrc && \
    /bin/bash -c ". /root/.bashrc; \
                  conda install -c conda-forge pymc-marketing ipykernel -y; \
                  python -m ipykernel install --user --name=pymc-marketing;"

# Create the directory with correct permissions
RUN mkdir -p /home/jovyan/.local/share/jupyter/runtime && \
    chown -R jovyan:users /home/jovyan/.local/share/jupyter

# Switch back to jovyan user
USER jovyan

WORKDIR /home/jovyan/work

# Set the default command to start the notebook server
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]
