# Start with the official Debian base image
FROM jupyter/minimal-notebook:latest
RUN mamba install pymc_marketing -c conda-forge -y
