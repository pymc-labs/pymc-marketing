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
    orcid: 0000-0002-1097-6125
    affiliation: 1
  - name: Colt Allen
    orcid: 0009-0006-8055-6560
    affiliation: 1
  - name: Carlos Eduardo Trujillo Agostini
    orcid: 0009-0009-5926-5701
    affiliation: 1
  - name: Ricardo Vieira
    orcid: 0000-0003-4690-7110
    affiliation: 1
  - name: Benjamin T. Vincent
    orcid: 0000-0002-8801-2430
    affiliation: 2
  - name: Thomas V. Wiecki
    orcid: "0009-0000-6015-101X"
    affiliation: 1
  - name: Nathaniel Forde
    orcid: 0009-0005-7585-0987
    affiliation: 1
  - name: Luciano Paz
    orcid: 0000-0002-6255-3888
    affiliation: 3
  - name: Pablo de Roque
    orcid: 0000-0002-0751-9126
    affiliation: 1
  - name: Imri Sofer
    orcid: 0009-0002-5367-3850
    affiliation: 1
  - name: Erik J. Ringen
    orcid: 0000-0002-3565-6961
    affiliation: 4
affiliations:
 - name: PyMC Labs
   index: 1
 - name: InferenceWorks Ltd
   index: 2
 - name: PLC Tech d.o.o.
   index: 3
 - name: Independent Researcher
   index: 4
date: 14 May 2026
bibliography: paper.bib
---

# Summary

PyMC-Marketing is a comprehensive Python library implementing Bayesian marketing analytics, built on PyMC [@pymc2023]. Commercial marketing analytics tools typically provide limited transparency into their models, while open-source alternatives like Meta's Robyn and Google's Meridian focus primarily on media mix modeling [@facebook2022robyn; @google2023meridian]. PyMC-Marketing provides a unified framework spanning multiple marketing domains, including: Marketing Mix Models (MMMs), Customer Lifetime Value (CLV) analysis, Bass Diffusion Models, and Customer Choice Models. All outputs include full posterior distributions rather than point estimates, enabling explicit risk assessment in business decisions. Key capabilities include time-varying media coefficients via Hilbert space Gaussian process approximations [@solin2020hilbert], experimental calibration integrating lift tests into model likelihood, hierarchical BTYD models [@fader2020customer] for CLV, and budget optimization with business constraints. The library is available on PyPI and conda-forge, with comprehensive documentation and example notebooks at https://www.pymc-marketing.io.

# Statement of Need

Marketing organizations struggle to attribute sales outcomes to specific marketing activities across multiple touchpoints and delayed conversion effects. Existing solutions suffer from: (1) black-box proprietary models with limited customization; (2) oversimplified approaches failing to capture marketing dynamics; and (3) lack of uncertainty quantification for high-stakes decisions. The target audience includes marketing data scientists and analysts in industry, academic researchers in causal inference and marketing science, and practitioners building production marketing measurement systems.

PyMC-Marketing addresses these gaps by bridging marketing science research and practical applications. It operationalizes advanced Bayesian methods—hierarchical modeling, experimental calibration, and uncertainty quantification—within a user-friendly, scikit-learn compatible API. Key innovations include time-varying coefficients using modern Gaussian process approximations optimized for marketing contexts, and a novel experimental calibration framework that integrates lift test results directly into model likelihood. While frequentist approaches like Robyn provide bootstrap-based intervals, all PyMC-Marketing outputs include full Bayesian posterior distributions, enabling decision-makers to assess risk explicitly.

# State of the Field

Existing marketing mix modeling tools include Meta's Robyn [@facebook2022robyn] and Google's Meridian [@google2023meridian], which focus primarily on MMM with limited Bayesian inference capabilities. While Robyn provides bootstrap-based uncertainty intervals, it lacks full posterior distributions. Meridian offers Bayesian inference but is limited to MMM without extending to customer lifetime value, choice analysis, or product diffusion modeling. Benchmarks demonstrate that PyMC-Marketing achieves more efficient sampling and more accurate channel contribution recovery than Meridian, with explicit Fourier-based seasonality providing clearer separation of trend, seasonality, and media effects [@pymclabs2025meridian].

PyMC-Marketing fills this gap by providing a unified Bayesian framework across multiple marketing domains (MMM, CLV, Bass diffusion, choice modeling) with full uncertainty quantification. Contributing to existing tools was not appropriate for several reasons: Robyn is implemented in R with bootstrap-based inference, making Python-native full Bayesian methods architecturally incompatible; Meridian is a closed-source commercial product; and neither tool extends beyond MMM to CLV, choice modeling, or product diffusion. We therefore created a standalone Python library integrating advanced Bayesian methods (hierarchical modeling, experimental calibration, time-varying parameters via modern GP approximations) within a scikit-learn compatible API. This design enables both methodological research and production applications while maintaining computational efficiency.

# Software Design

PyMC-Marketing follows a modular component architecture built on the PyMC probabilistic programming framework [@pymc2023]. The design prioritizes flexibility and extensibility through pluggable transformation components (adstock, saturation functions) and a builder pattern for model construction.

Key architectural decisions include: (1) separation of data transformation from model specification enabling custom function implementation; (2) a familiar `fit`/`predict` interface pattern lowering the barrier to adoption for practitioners; (3) PyMC backend providing automatic differentiation and multiple MCMC samplers; (4) standardized serialization for production deployment via MLflow [@zaharia2018mlflow]. This architecture enables both methodological research and production applications while maintaining computational efficiency through GPU acceleration and modern sampling algorithms including NumPyro [@bingham2019pyro] and Nutpie [@Seyboldt_nutpie]. Quality is maintained through a comprehensive automated test suite integrated with GitHub Actions CI, covering core functionality across multiple Python versions.

# Research Impact Statement

PyMC-Marketing provides uncertainty quantification through full posterior distributions, experimental calibration anchoring observational models to causal ground truth, and flexible budget optimization with business constraints. The scikit-learn compatible API ensures seamless integration into existing data science workflows. The library has been deployed in production by companies including HelloFresh [@pymclabs2023hellofresh] and Bolt [@pymclabs2023bolt]. It is also featured in peer-reviewed surveys of open-source MMM tools [@runge2026opensource].

Novel methodological contributions include: (1) time-varying coefficients using modern Gaussian process approximations specifically optimized for marketing applications; (2) experimental calibration framework integrating lift test results directly into model likelihood—a novel approach in the marketing science literature; (3) comprehensive marginal effects analysis for marketing sensitivity studies [@arelbundock2024marginaleffects]. Comprehensive tutorials, example notebooks, and video resources are available in the online documentation at https://www.pymc-marketing.io/en/stable/, with community support from over 70 contributors and translations in Spanish.

# AI Usage Disclosure

Generative AI tools were used during paper preparation. OpenCode (v1.0.220) with Claude Opus 4.5 and OpenCode (v1.14.48) with Claude Sonnet 4.6 via GitHub Copilot assisted with gathering information from existing documentation and codebase, drafting text, and incorporating peer reviewer feedback. The PyMC-Marketing software itself was developed by human contributors. All paper content was reviewed, edited, and validated by the human authors.

# Acknowledgments

We acknowledge the PyMC development team for the foundational probabilistic programming framework and the broader PyData ecosystem contributors. Special recognition goes to the marketing science research community for developing the theoretical frameworks that PyMC-Marketing operationalizes [@jin2017bayesian; @fader2005counting; @bass1969new; @train2009discrete].

PyMC-Marketing is a community-driven project built primarily through volunteer contributions from over 70 developers. Some contributions by PyMC Labs affiliates have received partial funding from both consulting engagements on marketing analytics and its internal budget.

# References
