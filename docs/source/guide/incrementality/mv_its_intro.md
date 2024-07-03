# Multivariate interrupted time series models

One modeling approach we could use for causal analysis of product incrementality is the multivariate interrupted time series (MV-ITS) model. This model is a generalization of the interrupted time series model (ITS), which is a common approach in causal inference. The MV-ITS model allows us to estimate the causal impact of an intervention (e.g., a new product introduction) on multiple outcomes (e.g., sales of multiple products) simultaneously.

One of the differences between this model and the standard ITS modeling approach is that rather than having a binary off/on intervention, the time series of sales of the new product can be thought of as a graded intervention.

## Building our intuition

Below we outline the simplest possible version of the model to build up our intuition before we move on to more complex versions.

![](mv_its_schematic.jpg)

In this example we aggregate sales data into your companies total sales and your competitors total sales. We then have a time series of sales data for your company and your competitors. We can see that when we release a new product, our sales are not really affected, but our competitors' total sales decrease. So an intuitively right answer here is that our new product has a high level of incrementality.

The MV-ITS approach models works as follows:
* It models the time series of sales data for your company and your competitors.
* These sales are modeled as normally distributed around an expected value with some degree of observation noise.
* The model expectation could be built up of multiple components, such as a trend, seasonality, and the impact of marketing campaigns. However for this simple example we have an intercept only model.
* Importantly, the expectation described above is decreased by some fraction of the new product sales. This fraction (across multiple sales outcomes) determines where the sales of the new product are coming from. From this, we can work out the level of incrementality of the new product.

## Model specification
COMING SOON

## Saturated markets
COMING SOON

## Unsaturated markets with growth potential
COMING SOON
