FROM jupyter/base-notebook:python-3.12

LABEL name="pymc-marketing"
LABEL description="Environment for PyMC-Marketing"

# Install uv and dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/
COPY pyproject.toml uv.lock /tmp/
WORKDIR /tmp
RUN uv sync --frozen
