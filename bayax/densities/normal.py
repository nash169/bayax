#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
from bayax.utils.types import Scalar, Optional, Tuple, Int
from bayax.densities.abstract_density import AbstractDensity


class Normal(AbstractDensity):
    def __init__(
        self,
        var: Scalar,
        mean: Scalar = 0.0
    ) -> None:
        self._mean, self._var = mean, var

    def __call__(
        self,
        x: Scalar
    ) -> Scalar:
        r"""
        Evaluate density.
        """
        return -0.5 * (jnp.log(self._var) + jnp.log(2 * jnp.pi) + jnp.square(x - self._mean) / self._var)

    def mean(
        self,
    ) -> Scalar:
        return self._mean

    def var(
        self,
    ) -> Scalar:
        return self._var

    def sample(
        self,
        size: Int = 1,
        seed: Int = 0,
        **kwargs
    ):
        return self._mean + jnp.sqrt(self._var) * jax.random.normal(jax.random.key(seed), shape=(size,), **kwargs)

    def jvp(
        self,
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        return 0.5 * (self._mean - x) / self._var * v

    def hvp(
        self,
        x: Scalar,
        v: Scalar
    ) -> None:
        NotImplementedError("Hessian vector product with respect to data not implemented yet.")

    def jvp_params(
        self,
        **kwargs
    ) -> Tuple:
        r"""
        Return handles for gradient function with respect to the params.
        """
        return lambda x, v: self._jvp_mean(x, v, **kwargs), lambda x, v: self._jvp_var(x, v, **kwargs)

    def hvp_params(
        self,
        **kwargs
    ) -> Tuple:
        r"""
        Return handles for hessian function with respect to the params.
        """
        return lambda x, v: self._hvp_mean(x, v, **kwargs), lambda x, v: self._hvp_var(x, v, **kwargs)

    def _jvp_mean(
        self,
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        return (x - self._mean) / self._var * v

    def _jvp_var(
        self,
        x: Scalar,
        v: Scalar,
        **kwargs
    ) -> None:
        NotImplementedError("Jacobian vector product with respect to variance not implemented yet")

    def _hvp_mean(
        self,
        x: Scalar,
        v: Scalar
    ) -> Scalar:
        return -v / self._var

    def _hvp_var(
        self,
        x: Scalar,
        v: Scalar,
        **kwargs
    ) -> None:
        raise NotImplementedError("Hessian vector product with respect to variance not implemented yet")
