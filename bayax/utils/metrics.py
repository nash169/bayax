#!/usr/bin/env python
# encoding: utf-8

import jax.numpy as jnp

from bayax.densities import MultivariateNormal
from bayax.utils.types import Scalar, Vector


def mean_absolute_error(
    pred_dist: MultivariateNormal,
    test_y: Vector,
) -> Scalar:
    """
    Mean absolute error.
    """
    return jnp.mean(jnp.abs(pred_dist._mean - test_y), axis=-1)


def mean_squared_error(
    pred_dist: MultivariateNormal,
    test_y: Vector,
    squared: bool = True,
) -> Scalar:
    """
    Mean squared error.
    """
    res = jnp.mean(jnp.square(pred_dist._mean - test_y), axis=-1)
    if not squared:
        return res**0.5
    return res


def standardized_mean_squared_error(
    pred_dist: MultivariateNormal,
    test_y: Vector,
) -> Scalar:
    """Standardized mean squared error.

    Standardizes the mean squared error by the variance of the test data.
    """
    return mean_squared_error(pred_dist, test_y, squared=True) / jnp.var(test_y)


def negative_log_predictive_density(
    pred_dist: MultivariateNormal,
    test_y: Vector,
) -> Scalar:
    """Negative log predictive density.

    Computes the negative predictive log density normalized by the size of the test data.
    """
    return -pred_dist(test_y) / test_y.shape[-1]
