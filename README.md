<div align="center">

![PyMC-Marketing Logo](docs/source/_static/marketing-logo-light.jpg)

</div>

----

![Build](https://github.com/pymc-labs/pymc-marketing/workflows/ci/badge.svg)
[![codecov](https://codecov.io/gh/pymc-labs/pymc-marketing/branch/main/graph/badge.svg?token=OBV3BS5TYE)](https://codecov.io/gh/pymc-labs/pymc-marketing)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![docs](https://readthedocs.org/projects/pymc-marketing/badge/?version=latest)](https://docs.readthedocs.io/en/latest/)
[![PyPI Version](https://img.shields.io/pypi/v/pymc-marketing.svg)](https://pypi.python.org/pypi/pymc-marketing)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# <span style="color:limegreen">PyMC-Marketing</span>: Bayesian Marketing Mix Modeling (MMM) & Customer Lifetime Value (CLV)

## Marketing Analytics Tools from [PyMC Labs](https://www.pymc-labs.com)

Unlock the power of **Marketing Mix Modeling (MMM)** and **Customer Lifetime Value (CLV)** analytics with PyMC-Marketing. This open-source marketing analytics tool empowers businesses to make smarter, data-driven decisions for maximizing ROI in marketing campaigns.

## Quick Installation Guide for Marketing Mix Modeling (MMM) & CLV

To dive into MMM and CLV analytics, set up a specialized environment, `marketing_env`, via conda-forge:

```bash
conda create -c conda-forge -n marketing_env pymc-marketing
conda activate marketing_env
```

For a comprehensive installation guide, refer to the [official PyMC installation documentation](https://www.pymc.io/projects/docs/en/latest/installation.html).

### Docker

We provide a `Dockerfile` to build a Docker image for PyMC-Marketing so that is accessible from a Jupyter Notebook. See [here](/scripts/docker/README.md) for more details.

## In-depth Bayesian Marketing Mix Modeling (MMM) in PyMC

Leverage our Bayesian MMM API to tailor your marketing strategies effectively. Based on the research [Jin, Yuxue, et al. “Bayesian methods for media mix modeling with carryover and shape effects.” (2017)](https://research.google/pubs/pub46001/),  and integrating the expertise from core PyMC developers, our API provides:

- **Adstock Transformation**: Optimize the carry-over effects in your marketing channels.
- **Saturation Effects**: Understand the diminishing returns in media investments.
- **Budget Optimization**: Allocate your marketing spend efficiently across various channels for maximum ROI.
- **Experiment Calibration**: Fine-tune your model based on empirical experiments for a more unified view of marketing.

Explore a hands-on [simulated example](https://pymc-marketing.readthedocs.io/en/stable/notebooks/mmm/mmm_example.html) for more insights into MMM with PyMC-Marketing.

### Essential Reading for Marketing Mix Modeling (MMM)

- [Bayesian Media Mix Modeling for Marketing Optimization](https://www.pymc-labs.com/blog-posts/bayesian-media-mix-modeling-for-marketing-optimization/)
- [Improving the Speed and Accuracy of Bayesian Marketing Mix Models](https://www.pymc-labs.com/blog-posts/reducing-customer-acquisition-costs-how-we-helped-optimizing-hellofreshs-marketing-budget/)
- [Johns, Michael and Wang,  Zhenyu. "A Bayesian Approach to Media Mix Modeling"](https://www.youtube.com/watch?v=UznM_-_760Y)
- [Orduz, Juan. "Media Effect Estimation with PyMC: Adstock, Saturation & Diminishing Returns"](https://juanitorduz.github.io/pymc_mmm/)
- [A Comprehensive Guide to Bayesian Marketing Mix Modeling](https://1749.io/resource-center/f/a-comprehensive-guide-to-bayesian-marketing-mix-modeling)

## Unlock Customer Lifetime Value (CLV) with PyMC

Understand and optimize your customer's value with our **CLV models**. Our API supports various types of CLV models, catering to both contractual and non-contractual settings, as well as continuous and discrete transaction modes.

Explore our detailed CLV examples using data from the [`lifetimes`](https://github.com/CamDavidsonPilon/lifetimes) package:

- [CLV Quickstart](https://pymc-marketing.readthedocs.io/en/stable/notebooks/clv/clv_quickstart.html)
- [BG/NBD model](https://pymc-marketing.readthedocs.io/en/stable/notebooks/clv/bg_nbd.html)
- [Gamma-Gamma model](https://pymc-marketing.readthedocs.io/en/stable/notebooks/clv/gamma_gamma.html)

### Examples

|                | **Non-contractual** | **Contractual**                 |
|----------------|---------------------|---------------------------------|
| **Continuous** | Buying groceries    | Audible                         |
| **Discrete**   | Cinema ticket       | Monthly or yearly subscriptions |

---

## Why PyMC-Marketing vs other solutions?

PyMC-Marketing is and will always be free for commercial use, licensed under [Apache 2.0](LICENSE). Developed by core developers behind the popular PyMC package and marketing experts, it provides state-of-the-art measurements and analytics for marketing teams.

Due to its open-source nature and active contributor base, new features are added constantly. Missing a feature or want to contribute? Fork our repository and submit a pull request. For any questions, feel free to [open an issue](https://github.com/your-repo/issues).

## Marketing AI Assistant: MMM-GPT with PyMC-Marketing

Not sure how to start or have questions? MMM-GPT is an AI that answers questions and provides expert advice on marketing analytics using PyMC-Marketing.

**[Try MMM-GPT here.](https://mmm-gpt.com/)**



## 📞 Schedule a Free Consultation for MMM & CLV Strategy

Maximize your marketing ROI with a [free 30-minute strategy session](https://calendly.com/niall-oulton) with our PyMC-Marketing experts. Learn how Bayesian Marketing Mix Modeling and Customer Lifetime Value analytics can boost your organization by making smarter, data-driven decisions.

For businesses looking to integrate PyMC-Marketing into their operational framework, [PyMC Labs](https://www.pymc-labs.com) offers expert consulting and training. Our team is proficient in state-of-the-art Bayesian modeling techniques, with a focus on Marketing Mix Models (MMMs) and Customer Lifetime Value (CLV). Explore these topics further by watching our video on [Bayesian Marketing Mix Models: State of the Art](https://www.youtube.com/watch?v=xVx91prC81g).

We provide the following professional services:

- **Custom Models**: We tailor niche marketing analytics models to fit your organization's unique needs.
- **Build Within PyMC-Marketing**: Our team members are experts leveraging the capabilities of PyMC-Marketing to create robust marketing models for precise insights.
- **SLA & Coaching**: Get guaranteed support levels and personalized coaching to ensure your team is well-equipped and confident in using our tools and approaches.
- **SaaS Solutions**: Harness the power of our state-of-the-art software solutions to streamline your data-driven marketing initiatives.
