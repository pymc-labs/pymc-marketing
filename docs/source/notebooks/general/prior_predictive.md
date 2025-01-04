# Prior Predictive Modeling with PyMC

This guide provides an introduction to prior predictive modeling using PyMC and the `Prior` class from PyMC-Marketing. Before diving into the technical details, let's understand why priors are crucial in Bayesian analysis and their practical importance in industry applications.

## Understanding Bayesian Inference

Bayesian inference is based on Bayes' theorem, which provides a formal way to update our beliefs about parameters $\theta$ given observed data $y$:

$$p(\theta|y) = \frac{p(y|\theta)p(\theta)}{p(y)}$$

Where:
- $p(\theta|y)$ is the posterior probability (what we want to learn)
- $p(y|\theta)$ is the likelihood (how the data is generated)
- $p(\theta)$ is the prior probability (our initial beliefs)
- $p(y)$ is the evidence (a normalizing constant), which can be written as $p(y) = \int p(y|\theta)p(\theta)d\theta$

The posterior distribution combines our prior knowledge with the observed data to give us updated beliefs about the parameters. In practice, we often work with the unnormalized posterior:

$$p(\theta|y) \propto p(y|\theta)p(\theta)$$

This is because the normalizing constant $p(y)$ is often intractable to compute directly.

### Why Priors Matter in Industry

In industry applications, priors serve several crucial purposes:

1. **Domain Knowledge Integration**:
   - Incorporating expert knowledge into models
   - Leveraging historical data from similar projects
   - Encoding business constraints and requirements

2. **Risk Management**:
   - Preventing unrealistic predictions
   - Ensuring stable model behavior
   - Managing uncertainty in decision-making

3. **Data Efficiency**:
   - Making models work with limited data
   - Faster convergence to reasonable solutions
   - Robust predictions in new scenarios

4. **Model Regularization**:
   - Preventing overfitting
   - Handling multicollinearity
   - Dealing with sparse data

### Common Prior Specification Scenarios

In marketing analytics, you'll often encounter these scenarios:

1. **Marketing Mix Models**:
   - Media channel effectiveness (typically positive)
   - Diminishing returns (shape constraints)
   - Seasonal patterns (periodic effects)

2. **Customer Lifetime Value**:
   - Purchase rates (positive values)
   - Churn probabilities (between 0 and 1)
   - Monetary value distributions (positive, often log-normal)

3. **A/B Testing**:
   - Conversion rates (bounded between 0 and 1)
   - Lift measurements (centered around small effects)
   - Revenue impacts (potentially heavy-tailed)

## What is Prior Predictive Modeling?

Prior predictive modeling is a crucial step in Bayesian workflow that helps us validate our prior choices before seeing the actual data. The process involves:

1. **Specification**:
   - Define prior distributions for model parameters
   - Encode domain knowledge and constraints
   - Document assumptions and choices

2. **Simulation**:
   - Sample parameters from prior distributions
   - Generate synthetic data using the model structure
   - Create multiple scenarios of possible outcomes

3. **Validation**:
   - Check if simulated data matches domain expertise
   - Verify that impossible scenarios are excluded
   - Ensure reasonable coverage of possible outcomes

### Benefits in Practice

1. **Early Problem Detection**:
   - Identify unrealistic assumptions
   - Catch numerical issues before model fitting
   - Validate model structure

2. **Stakeholder Communication**:
   - Visualize model implications
   - Justify modeling choices
   - Set realistic expectations

3. **Model Development**:
   - Iterate on prior choices efficiently
   - Compare alternative specifications
   - Document model evolution

4. **Risk Assessment**:
   - Understand model limitations
   - Identify edge cases
   - Plan for failure modes

The prior predictive distribution $p(y)$ represents our beliefs about the data before we observe it. Mathematically, it's the distribution of the data marginalized over the prior:

$$p(y) = \int p(y|\theta)p(\theta)d\theta$$

In practice, we can sample from this distribution by:
1. Drawing parameters from the prior: $\theta^{(s)} \sim p(\theta)$
2. Generating data from the likelihood: $y^{(s)} \sim p(y|\theta^{(s)})$

This process helps us validate our model in several ways:

1. **Parameter Space Coverage**:
   The samples $\{\theta^{(s)}\}_{s=1}^S$ show us what parameter values we consider plausible

2. **Data Space Coverage**:
   The samples $\{y^{(s)}\}_{s=1}^S$ show us what data our model can generate

3. **Model Sensitivity**:
   The relationship between $\theta^{(s)}$ and $y^{(s)}$ shows how parameters influence predictions

Let's explore these concepts through practical examples using the `Prior` class from PyMC-Marketing.

## Getting Started

First, let's import the necessary libraries:

```python
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from pymc_marketing.prior import Prior

# Set random seed for reproducibility
np.random.seed(42)

# Set plot style
plt.style.use('seaborn')
```

## Simple Example: Normal Distribution

Let's start with a simple example using a normal distribution. We'll:
1. Create a normal prior
2. Visualize its distribution
3. Sample from its prior predictive distribution

```python
# Create a normal prior with mean 0 and standard deviation 1
normal_prior = Prior("Normal", mu=0, sigma=1)

# Plot the PDF of the prior
fig, ax = plt.subplots(figsize=(10, 6))
normal_prior.preliz.plot_pdf(ax=ax)
ax.set_title('Probability Density Function of Normal Prior')

# Sample from the prior predictive distribution
samples = normal_prior.sample_prior(draws=1000)

# Plot histogram of samples
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(samples.var.values, bins=30, density=True, alpha=0.7)
ax.set_title('Prior Predictive Samples')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
```

## Hierarchical Example: Marketing Channels

Now let's look at a more realistic example in the marketing context. Imagine we're modeling the effectiveness of different marketing channels using a hierarchical model where:
- Each channel has its own effectiveness parameter
- These parameters are drawn from a common distribution

```python
# Create a hierarchical prior for channel effectiveness
channel_effectiveness = Prior(
    "Normal",
    mu=Prior("Normal", mu=0, sigma=1),  # population mean
    sigma=Prior("HalfNormal", sigma=0.5),  # population standard deviation
    dims="channel"
)

# Define channels
channels = ["TV", "Radio", "Social Media", "Search"]
coords = {"channel": channels}

# Sample from the prior predictive distribution
samples = channel_effectiveness.sample_prior(coords=coords, draws=1000)

# Create a box plot to visualize the distribution of effectiveness across channels
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot([samples.var.sel(channel=ch).values for ch in channels], labels=channels)
ax.set_title('Prior Predictive Distribution of Channel Effectiveness')
ax.set_ylabel('Effectiveness')
ax.grid(True, alpha=0.3)
```

## Constrained Example: Conversion Rates

In many marketing applications, we deal with rates or proportions that must be between 0 and 1. The `Prior` class provides a convenient way to create constrained distributions:

```python
# Create a Beta prior for conversion rates
conversion_rate = Prior("Beta").constrain(lower=0.01, upper=0.1, mass=0.95)

# Plot the PDF
fig, ax = plt.subplots(figsize=(10, 6))
conversion_rate.preliz.plot_pdf(ax=ax)
ax.set_title('Prior Distribution for Conversion Rate')

# Sample from the prior predictive distribution
samples = conversion_rate.sample_prior(draws=1000)

# Plot histogram of samples
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(samples.var.values, bins=30, density=True, alpha=0.7)
ax.set_title('Prior Predictive Samples for Conversion Rate')
ax.set_xlabel('Conversion Rate')
ax.set_ylabel('Density')
```

## End-to-End Example: Linear Regression

Let's explore how different prior specifications affect a simple linear regression model. We'll:
1. Generate some synthetic data
2. Create models with different priors
3. Compare their prior predictive distributions
4. See how these priors affect our inferences

```python
# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 50)
true_intercept = 2
true_slope = 1.5
y = true_intercept + true_slope * X + np.random.normal(0, 1, size=50)

# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, y, alpha=0.5, label='Data')
ax.plot(X, true_intercept + true_slope * X, 'r--', label='True Line')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Synthetic Data for Linear Regression')
ax.legend()
```

### Comparing Different Prior Specifications

Let's create three different prior specifications:
1. Weakly informative priors (default)
2. Strongly informative priors (correct direction)
3. Strongly informative priors (wrong direction)

```python
# 1. Weakly informative priors
weak_priors = {
    'intercept': Prior("Normal", mu=0, sigma=10),
    'slope': Prior("Normal", mu=0, sigma=10)
}

# 2. Strongly informative priors (correct direction)
informed_priors = {
    'intercept': Prior("Normal", mu=2, sigma=0.5),
    'slope': Prior("Normal", mu=1, sigma=0.5)
}

# 3. Strongly informative priors (wrong direction)
wrong_priors = {
    'intercept': Prior("Normal", mu=-2, sigma=0.5),
    'slope': Prior("Normal", mu=-1, sigma=0.5)
}

prior_sets = {
    'Weakly Informative': weak_priors,
    'Strongly Informative (Correct)': informed_priors,
    'Strongly Informative (Wrong)': wrong_priors
}
```

### Visualizing Prior Predictive Distributions

Let's see how these different priors affect our predictions before seeing the data:

```python
# Function to sample from prior predictive
def sample_prior_predictive(priors, X, draws=100):
    intercept_samples = priors['intercept'].sample_prior(draws=draws).var.values
    slope_samples = priors['slope'].sample_prior(draws=draws).var.values

    y_samples = np.zeros((draws, len(X)))
    for i in range(draws):
        y_samples[i] = intercept_samples[i] + slope_samples[i] * X
    return y_samples

# Plot prior predictive for each prior set
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
X_plot = np.linspace(-2, 12, 100)

for (name, priors), ax in zip(prior_sets.items(), axes):
    y_samples = sample_prior_predictive(priors, X_plot, draws=100)

    # Plot a subset of lines
    for j in range(20):
        ax.plot(X_plot, y_samples[j], 'b-', alpha=0.1)

    # Plot the true line
    ax.plot(X_plot, true_intercept + true_slope * X_plot, 'r--', label='True Line')

    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'{name} Priors\nPrior Predictive Distribution')
    ax.set_ylim(-20, 20)
    ax.legend()

plt.tight_layout()
```

### Building and Sampling from the Model

Now let's see how these priors affect our posterior inference:

```python
def build_and_sample_model(X, y, priors):
    with pm.Model() as model:
        # Priors
        intercept = priors['intercept'].create_variable('intercept')
        slope = priors['slope'].create_variable('slope')

        # Expected value
        mu = intercept + slope * X

        # Likelihood
        likelihood = pm.Normal('y', mu=mu, sigma=1, observed=y)

        # Sample
        trace = pm.sample(1000, return_inferencedata=True)

    return trace

# Sample from each model
traces = {}
for name, priors in prior_sets.items():
    traces[name] = build_and_sample_model(X, y, priors)

# Plot the results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for (name, trace), ax in zip(traces.items(), axes):
    # Plot data
    ax.scatter(X, y, alpha=0.3, label='Data')

    # Plot true line
    ax.plot(X, true_intercept + true_slope * X, 'r--', label='True Line')

    # Plot posterior predictions
    intercept_samples = trace.posterior.intercept.values.flatten()
    slope_samples = trace.posterior.slope.values.flatten()

    # Plot a subset of posterior lines
    for j in range(100):
        ax.plot(X_plot,
               intercept_samples[j] + slope_samples[j] * X_plot,
               'b-', alpha=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'{name} Priors\nPosterior Predictions')
    ax.legend()

plt.tight_layout()
```

### Key Observations

1. **Weakly Informative Priors**:
   - Allow the data to dominate the inference
   - Prior predictive shows wide range of possible lines
   - Posterior converges close to true values

2. **Strongly Informative Priors (Correct)**:
   - Prior predictive concentrated around true values
   - Faster convergence to posterior
   - Smaller uncertainty in predictions

3. **Strongly Informative Priors (Wrong)**:
   - Prior predictive suggests negative relationship
   - Data eventually overcomes prior beliefs
   - But requires more data than with weak/correct priors

### Best Practices for Prior Selection in Regression

1. **Start Weakly Informative**:
   - Use broad normal priors for coefficients
   - Center around 0 if no prior knowledge
   - Scale based on expected magnitude of effects

2. **Incorporate Domain Knowledge**:
   - Use historical data to inform prior means
   - Set prior scales based on plausible effect sizes
   - Document and justify prior choices

3. **Validate with Prior Predictive**:
   - Check if prior predictive covers plausible outcomes
   - Ensure extreme values are possible but unlikely
   - Verify that prior constraints make sense

## A Note on Priors for Generalized Linear Models

When working with generalized linear models (GLMs), normal priors on coefficients can sometimes lead to unexpected behavior due to the non-linear link functions. The general form of a GLM is:

$$g(\mathbb{E}[y]) = \eta = X\beta$$

where:
- $g(\cdot)$ is the link function
- $\mathbb{E}[y]$ is the expected value of the response
- $\eta$ is the linear predictor
- $X$ is the design matrix
- $\beta$ are the coefficients

Let's demonstrate this with a logistic regression example, where the link function is:

$$g(\mu) = \text{logit}(\mu) = \log\left(\frac{\mu}{1-\mu}\right)$$

and its inverse (the response function) is:

$$g^{-1}(\eta) = \text{sigmoid}(\eta) = \frac{1}{1 + e^{-\eta}}$$

```python
# Example with logistic regression
# Generate some example data points
np.random.seed(42)
X = np.linspace(-5, 5, 100)

# Let's try different priors for the slope coefficient
weak_normal = Prior("Normal", mu=0, sigma=10)
strong_normal = Prior("Normal", mu=0, sigma=1)
student_t = Prior("StudentT", nu=3, sigma=2.5)

# Function to transform linear predictor to probability
def logistic(x):
    return 1 / (1 + np.exp(-x))

# Sample from prior predictive for each prior
def sample_logistic_prior_predictive(prior, X, draws=100):
    beta_samples = prior.sample_prior(draws=draws).var.values
    y_samples = np.zeros((draws, len(X)))
    for i in range(draws):
        linear_pred = beta_samples[i] * X
        y_samples[i] = logistic(linear_pred)
    return y_samples

# Plot prior predictive distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
priors = {
    'Normal(0, 10)': weak_normal,
    'Normal(0, 1)': strong_normal,
    'StudentT(3, 2.5)': student_t
}

for (name, prior), ax in zip(priors.items(), axes):
    y_samples = sample_logistic_prior_predictive(prior, X)

    # Plot a subset of curves
    for j in range(20):
        ax.plot(X, y_samples[j], 'b-', alpha=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Probability')
    ax.set_title(f'{name}\nPrior Predictive')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

plt.tight_layout()
```

### Why Normal Priors Can Be Problematic in GLMs

1. **The Link Function Effect**:
   - In logistic regression, coefficients $\beta$ are transformed through the logistic function
   - Large values of $\eta = X\beta$ get "squashed" to probabilities near 0 or 1
   - Normal priors with large variances can lead to:
     * Many predicted probabilities stuck at extremes ($\approx 0$ or $\approx 1$)
     * Unrealistic step-function-like predictions
     * Numerical instability during sampling

2. **Better Prior Choices**:
   - Student's t distribution with small degrees of freedom (e.g., $\nu=3$)
   - Regularizing priors like the Horseshoe: $\beta_j \sim \mathcal{N}(0, \lambda_j\tau)$
   - Log-odds scale considerations for intercepts: $\beta_0 \sim \mathcal{N}(0, 10)$ on logit scale

Here's a more appropriate prior specification for logistic regression:

```python
# Better prior specification for logistic regression
better_priors = {
    'intercept': Prior("StudentT", nu=3, sigma=2.5),  # Prior on intercept
    'beta': Prior("StudentT", nu=3, sigma=2.5)        # Prior on slope
}

# Create and sample from the model
def build_logistic_model(X, y, priors):
    with pm.Model() as model:
        # Priors
        intercept = priors['intercept'].create_variable('intercept')
        beta = priors['beta'].create_variable('beta')

        # Linear predictor
        eta = intercept + beta * X

        # Likelihood using logistic link function
        likelihood = pm.Bernoulli('y', logit_p=eta, observed=y)

        # Sample
        trace = pm.sample(1000, return_inferencedata=True)

    return trace

# The Student's t prior leads to more reasonable predictions
# and better numerical stability during sampling
```

### Key Recommendations for GLM Priors

1. **Consider the Scale of the Link Function**:
   - Logit link: Work on log-odds scale (-∞ to ∞)
   - Log link: Consider the natural scale of the response
   - Probit link: Similar considerations to logit

2. **Use Heavy-tailed Distributions**:
   - Student's t with low degrees of freedom
   - Laplace distribution
   - Horseshoe prior for sparse problems

3. **Prior Predictive Checks are Crucial**:
   - Verify predictions cover reasonable ranges
   - Check for unrealistic step functions
   - Ensure proper coverage of outcome space

## Advanced Features of the Prior Class

The `Prior` class offers several useful features:

1. **Distribution Visualization**
   ```python
   # Visualize any prior distribution
   my_prior = Prior("Gamma", alpha=2, beta=0.5)
   my_prior.preliz.plot_pdf()
   ```

2. **Hierarchical Models**
   ```python
   # Create nested priors for hierarchical models
   hierarchical_prior = Prior(
       "Normal",
       mu=Prior("Normal", mu=0, sigma=1),
       sigma=Prior("HalfNormal", sigma=0.5),
       dims="group"
   )
   ```

3. **Constrained Distributions**
   ```python
   # Constrain a distribution to a specific range
   constrained_prior = Prior("Normal").constrain(
       lower=0,
       upper=10,
       mass=0.95
   )
   ```

4. **Non-centered Parameterization**
   ```python
   # Create a non-centered normal distribution
   non_centered = Prior(
       "Normal",
       mu=Prior("Normal"),
       sigma=Prior("HalfNormal"),
       centered=False
   )
   ```

## Key Takeaways

1. Prior predictive modeling helps us validate our model assumptions before using real data
2. The `Prior` class provides a convenient interface for:
   - Creating and visualizing prior distributions
   - Sampling from prior predictive distributions
   - Creating hierarchical models
   - Constraining distributions to realistic ranges
3. Always visualize your prior predictive distributions to ensure they align with your domain knowledge

## Common Pitfalls and Best Practices

1. **Always Check Your Scales**
   - Make sure your priors are on the right scale for your data
   - Use domain knowledge to set reasonable bounds

2. **Hierarchical Model Considerations**
   - Start with simpler models before adding hierarchy
   - Verify that the hierarchical structure makes sense for your data

3. **Constraint Considerations**
   - Use constraints when you have clear bounds (e.g., rates between 0 and 1)
   - Be careful not to over-constrain your priors

4. **Visualization is Key**
   - Always plot your prior distributions
   - Check prior predictive samples against domain knowledge

This introduction covered the basics of prior predictive modeling with PyMC and the `Prior` class. As you build more complex models, these techniques become increasingly important for model validation and refinement.
