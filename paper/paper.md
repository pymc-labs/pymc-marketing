---
title: 'PyMC-Marketing: Bayesian Marketing Mix Models and Customer Analytics in Python'
tags:
  - Python
  - Bayesian statistics
  - marketing mix modeling
  - customer lifetime value
  - PyMC
  - probabilistic programming
  - causal inference
  - marketing analytics
authors:
  - name: William Dean
    orcid: 0009-0003-6510-3545
    affiliation: 1
  - name: Juan Orduz
    orcid:
    affiliation: 1
  - name: Colt Allen
    orcid:
  - name: Carlos Trujillo
    orcid:
    affiliation: 1
  - name: Ricardo Vieira
    orcid:
    affiliation: 1
  - name: Benjamin T. Vincent
    orcid:
    affiliation: 1
affiliations:
 - name: PyMC Labs
   index: 1
date: 4 January 2026
bibliography: paper.bib
---

# Summary

PyMC-Marketing is a comprehensive Python library implementing Bayesian marketing analytics, built on PyMC [@salvatier2016probabilistic]. Unlike existing commercial tools (which provide limited transparency) or open-source alternatives like Meta's Robyn [@facebook2022robyn] and Google's Meridian [@google2023meridian] (which focus primarily on media mix modeling), PyMC-Marketing provides a unified framework spanning four marketing domains with full uncertainty quantification and experimental calibration capabilities.

The library implements: (1) **Media Mix Modeling (MMM)** with customizable transformations, time-varying parameters, and lift test calibration; (2) **Customer Lifetime Value (CLV)** models including BG/NBD and Pareto/NBD frameworks; (3) **Bass Diffusion Models** for adoption forecasting; and (4) **Customer Choice Models** based on random utility theory. All models provide full posterior distributions rather than point estimates, enabling explicit risk assessment in business decisions.

# Statement of Need

Marketing organizations struggle with attribution across multiple touchpoints and delayed conversion effects. Existing solutions suffer from: (1) black-box proprietary models with limited customization; (2) oversimplified approaches failing to capture marketing dynamics; and (3) lack of uncertainty quantification for high-stakes decisions.

PyMC-Marketing addresses these gaps by operationalizing advanced Bayesian methods—hierarchical modeling, experimental calibration, and uncertainty quantification—within a user-friendly, scikit-learn compatible API. Unlike point-estimate approaches in Robyn or commercial tools, all outputs include credible intervals enabling decision-makers to assess risk explicitly.

# Installation

Install via conda-forge for optimal dependency management:
```bash
conda install -c conda-forge pymc-marketing
```

Or via pip:
```bash
pip install pymc-marketing
```

# Dependencies

Core dependencies include PyMC (≥5.0), NumPy, Pandas, ArviZ [@arviz2019], and scikit-learn. Optional dependencies enable GPU acceleration (JAX), advanced samplers (NumPyro, Nutpie), and production deployment (MLflow [@zaharia2018mlflow], Docker).

# Key Features

PyMC-Marketing provides four distinct modules addressing comprehensive marketing analytics:

**1. Media Mix Modeling (MMM)**: Multiple adstock functions, saturation curves, time-varying parameters via HSGP [@solin2020hilbert], and experimental calibration for causal inference.

**2. Customer Lifetime Value (CLV)**: BTYD models [@fader2020customer] including BG/NBD, Pareto/NBD, and Gamma-Gamma frameworks with hierarchical extensions and individual-level uncertainty.

**3. Bass Diffusion Models**: Product adoption forecasting [@bass1969new] with flexible parameterization for innovation and imitation effects across multiple products.

**4. Customer Choice Models**: Discrete choice analysis [@train2009discrete] based on random utility theory, including multinomial logit and multivariate interrupted time series models.

**Production Ready**: All modules feature MLflow [@zaharia2018mlflow] integration, Docker containerization, multiple MCMC backends, and comprehensive diagnostics via ArviZ.

# Key Advantages

PyMC-Marketing provides uncertainty quantification through full posterior distributions, experimental calibration anchoring observational models to causal ground truth, and flexible budget optimization with business constraints. The scikit-learn compatible API ensures seamless integration into existing data science workflows. Comprehensive tutorials and example notebooks demonstrating all model types are available in the online documentation at https://www.pymc-marketing.io/en/stable/, including translations in Spanish.

# Community Guidelines

- **Issues**: Report bugs and feature requests on [GitHub](https://github.com/pymc-labs/pymc-marketing/issues)
- **Discussions**: Join community discussions on [GitHub Discussions](https://github.com/pymc-labs/pymc-marketing/discussions)
- **Documentation**: Comprehensive guides and tutorials available at [pymc-marketing.io](https://www.pymc-marketing.io/en/stable/)
- **Contributing**: See [CONTRIBUTING.md](https://github.com/pymc-labs/pymc-marketing/blob/main/CONTRIBUTING.md) for development guidelines
- **Support**: Professional consulting available through [PyMC Labs](https://www.pymc-labs.com)

# Acknowledgments

We acknowledge the PyMC development team for the foundational probabilistic programming framework and the broader PyData ecosystem contributors. Special recognition goes to the marketing science research community for developing the theoretical frameworks that PyMC-Marketing operationalizes [@jin2017bayesian; @fader2005counting; @bass1969new; @train2009discrete].

# References
