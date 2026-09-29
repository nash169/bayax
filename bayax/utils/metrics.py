#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp

from bayax.densities import MultivariateNormal
from bayax.utils.types import Scalar, Vector


# ====================================================================================================
# Regression
# ====================================================================================================


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


# ====================================================================================================
# Sampling
# ====================================================================================================


def mmd(x, y, bandwidth):
    from bayax.kernels.rbf import rbf
    k_rbf = jax.jit(
        jax.vmap(
            jax.vmap(lambda x, y: rbf(x, y, jnp.log(bandwidth), 0.0), in_axes=(None, 0)),
            in_axes=(0, None),
        )
    )

    n, m = x.shape[0], y.shape[0]
    kernel_xx, kernel_yy, kernel_xy = k_rbf(x, x), k_rbf(y, y), k_rbf(x, y)

    xx = (jnp.sum(kernel_xx) - jnp.trace(kernel_xx)) / (n * (n - 1))
    yy = (jnp.sum(kernel_yy) - jnp.trace(kernel_yy)) / (m * (m - 1))
    xy = jnp.mean(kernel_xy)

    return xx + yy - 2.0 * xy


def ess_mcmc(samples):
    """Single-chain mean ESS (Effective Sample Size) per coordinate; sample order is along axis 0.

    Assumes a stationary chain with warmup already removed. FFT autocorrelations
    use zero padding and Geyer's initial positive/monotone paired sequence:
    https://mc-stan.org/docs/reference-manual/analysis.html
    This is a raw-coordinate estimate, not rank-normalized multichain ESS.
    Negative autocorrelation can yield ESS greater than the number of draws.
    NaN denotes fewer than four draws, a constant/nonfinite coordinate, or a
    nonpositive estimated integrated autocorrelation time.
    """
    samples = jnp.asarray(samples, dtype=jnp.result_type(samples, 1.0))
    n = samples.shape[0]
    if n < 4:
        return jnp.full(samples.shape[1:], jnp.nan, dtype=samples.dtype)

    centered = samples - jnp.mean(samples, axis=0)
    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = jnp.fft.rfft(centered, n=fft_size, axis=0)
    autocov = jnp.fft.irfft(
        spectrum * jnp.conj(spectrum), n=fft_size, axis=0,
    )[:n] / n
    variance = autocov[0]
    rho = autocov / jnp.where(variance > 0, variance, 1.0)

    # Pair lags (0, 1), (2, 3), ... and stop at the first nonpositive pair.
    pairs = rho[:2 * (n // 2)].reshape((n // 2, 2) + samples.shape[1:]).sum(axis=1)
    positive_prefix = jnp.cumprod((pairs > 0).astype(jnp.int32), axis=0).astype(bool)
    monotone_pairs = jax.lax.associative_scan(jnp.minimum, pairs, axis=0)
    tau = -1.0 + 2.0 * jnp.sum(jnp.where(positive_prefix, monotone_pairs, 0.0), axis=0)
    valid = (
        jnp.all(jnp.isfinite(samples), axis=0)
        & jnp.any(samples != samples[0], axis=0)
        & (variance > 0) & jnp.isfinite(tau) & (tau > 0)
    )
    return jnp.where(valid, n / jnp.where(valid, tau, 1.0), jnp.nan)
