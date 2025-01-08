# MMM Visualization App with Streamlit

## Overview

This Streamlit application is designed to provide a dynamic and interactive visualization of key Marketing Mix Modeling (MMM) concepts, including adstock, saturation, and the use of Bayesian priors. It aims to help marketers, data scientists, and anyone interested in understanding MMM more deeply. Through this application, users can explore how different parameters affect adstock, saturation, and Bayesian priors.
You may wish to run the app locally too - rather than relying on the [deployment](https://pymc-marketing-app.streamlit.app/).
In this case, you would just need to install the requirements.txt within the streamlit folder and do `streamlit run Visualise_Priors.py`

## Features

- **Adstock Transformation Visualization**: Interactive charts that demonstrate how the adstock effect changes with different decay rates and lengths of advertising impact. Users can input their parameters to see how adstocked values are calculated over time.

- **Saturation Curve Exploration**: Interactive charts that demonstrate saturation curves, which represents the diminishing returns of marketing spend as it increases. Users can adjust parameters and choose from a variety of saturation transformations.

- **Bayesian Priors**: Interactive charts that demonstrate Bayesian prior distributions, designed to showcase the power of Bayesian methods in handling uncertainty and incorporating prior knowledge into MMM.

- **Customizable Parameters**: All sections of the app include options to customize parameters, allowing users to experiment with different scenarios and understand their impacts on MMM.

## Getting Started

### Deployment

The app can be found [here](https://pymc-marketing-app.streamlit.app/)

## Contributing & Adding New Functions

We welcome contributions from the community! Whether it's adding new features, improving documentation, or reporting bugs, please feel free to make a pull request or open an issue.
It's a good idea to always develop and test your changes to the app by running it locally, before submitting a PR.

### Adding New Adstock / Saturation Transformers from pymc-marketing

New transformation functions may be added to pymc-marketing which you may want to have visualised in the app.
To do so, you would just need to add them in the import statements at the top of either `Saturation.py` or `Adstock.py`.
e.g.
```
from pymc_marketing.mmm.transformers import (
    logistic_saturation,
    michaelis_menten,
    tanh_saturation,
    my_new_saturation_function
)
```

Then you would have to create a new Streamlit tab
```
# Create tabs for plots
tab1, tab2, tab3, tab4 = st.tabs(["Logistic", "Tanh", "Michaelis-Menten", My New Saturation])
```

And then add whatever plotting code you want for your new function!

### Adding Additional Distributions from PreLiz

PreliZ contains many, many distributions - not all of which are currently visualised.
Adding new distributions is quite simple.
You would need to firstly modify the dictionary of distributions and the parameters you want the user to be able to play around with.
```
# Specify the possible distributions and their paramaters you want to visualise
DISTRIBUTIONS_DICT = {
    "Beta": ["alpha", "beta"],
    "Bernoulli": ["p"],
    "Exponential": ["lam"],
    "Gamma": ["alpha", "beta"],
    "HalfNormal": ["sigma"],
    "LogNormal": ["mu", "sigma"],
    "Normal": ["mu", "sigma"],
    "Poisson": ["mu"],
    "StudentT": ["nu", "mu", "sigma"],
    "TruncatedNormal": ["mu", "sigma", "lower", "upper"],
    "Uniform": ["lower", "upper"],
    "Weibull": ["alpha", "beta"],
    "MY_NEW_DIST": ["something", "something_else"],
}
```

And then create new Streamlit input buttons for your new parameters (unless they are covered by existing parameters in the `for param in params.keys():` block) by adding another `elif` line.
Watch out - certain distributions may share parameters of the same name, but that have different accepted ranges. For example, the `mu` parameter in Poisson has to be >0, whereas for a Normal it can be whatever you want. You may need an additional `elif` block in these edge cases.
