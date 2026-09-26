#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.random as jr
import jax.numpy as jnp
from flax import nnx
import inspect

from bayax.utils.types import (
    Matrix,
    Optional,
    Callable,
    Vector,
)
from bayax.operators import PSDOperator
from bayax.utils.math import gram
from bayax.densities import MultivariateNormal
from bayax.utils.helper import (
    pytree_to_array,
    array_to_pytree,
    wrap_pytree_function,
)


class GP(nnx.Module):
    def __init__(
        self,
        dim: int,
        kernel: Callable,
        mean: Optional[Callable] = None,
        seed: int = 0,
        **kwargs,
    ):
        r"""
        Gaussian Proces
        """
        key = jr.key(seed)
        self.noise = nnx.Param(jr.normal(key, shape=(1,)))

        if not isinstance(kernel, nnx.Module):
            k_params = list(inspect.signature(kernel).parameters.values())[2:]
            keys = jax.random.split(key, len(k_params) + 1)
            self.kernel_params = nnx.List(
                [
                    nnx.Param(
                        jr.normal(keys[i], shape=(dim if p.annotation.dims else 1,))
                    )
                    for i, p in enumerate(k_params)
                ]
            )
        self.k_fn = kernel

        if mean is not None:
            key, subkey = jr.split(key)
            if not isinstance(mean, nnx.Module):
                mu_params = list(inspect.signature(mean).parameters.values())[1:]
                keys = jax.random.split(subkey, len(mu_params))
                self.mean_params = nnx.List(
                    [
                        nnx.Param(
                            jr.normal(keys[i], shape=(dim if p.annotation.dims else 1,))
                        )
                        for i, p in enumerate(mu_params)
                    ]
                )
            self.mu_fn = mean

    @property
    def params(self) -> Vector:
        return pytree_to_array(nnx.state(self))

    @params.setter
    def params(self, value):
        nnx.update(self, array_to_pytree(value, nnx.state(self)))

# ====================================================================================================
# Prior
# ====================================================================================================

    def mean(self, x: Vector | Matrix):
        return (
            self.mu_fn(x, *self.mean_params)
            if hasattr(self, "mean_params")
            else self.mu_fn(x)
        ) if hasattr(self, "mu_fn") else None

    def kernel(self, x: Vector, y: Vector):
        return (
            self.k_fn(x, y, *self.kernel_params)
            if hasattr(self, "kernel_params")
            else self.k_fn(x, y)
        )

    def var(
        self,
        X: Vector | Matrix,
        **kwargs
    ):
        return jax.vmap(lambda x: self.kernel(x, x))(X)

    def cov(self, X: Vector | Matrix):
        noise = jnp.exp(self.noise).squeeze() + 1e-6
        return PSDOperator(op=gram(self.kernel, X, noise), op_size=X.shape[0])

    def __call__(self, X: Vector | Matrix):
        return MultivariateNormal(cov=self.cov(X), mean=jax.vmap(self.mean)(X) if hasattr(self, "mu_fn") else None)

# ====================================================================================================
# Posterior
# ====================================================================================================

    def posterior_mean(
        self,
        X: Vector | Matrix,
        y: Vector,
        **kwargs
    ):
        k_xy = lambda x: jax.vmap(self.kernel, in_axes=(None, 0))(x, X).squeeze()
        if hasattr(self, "mu_fn"):
            return lambda x: self.mean(x) + k_xy(x) @ self.cov(X).solve(y.squeeze() - self.mean(X).squeeze(), **kwargs)
        return lambda x: k_xy(x) @ self.cov(X).solve(y.squeeze(), **kwargs)

    def posterior_kernel(
        self,
        X: Vector | Matrix,
        **kwargs
    ):
        k_xy = lambda x: jax.vmap(self.kernel, in_axes=(None, 0))(x, X).squeeze()
        k_yx = lambda x: jax.vmap(self.kernel, in_axes=(0, None))(X, x).squeeze()
        return lambda x, y: self.kernel(x, y) - k_xy(x) @ self.cov(X).solve(k_yx(y), **kwargs)

    def posterior_var(
        self,
        X: Vector | Matrix,
        **kwargs
    ):
        return lambda x: jax.vmap(lambda z: self.posterior_kernel(X, **kwargs)(z, z))(x)

    def posterior_cov(
        self,
        X: Vector | Matrix,
        **kwargs
    ):
        noise = jnp.exp(self.noise).squeeze() + 1e-6
        return lambda x: PSDOperator(op=gram(self.posterior_kernel(X, **kwargs), x, noise), op_size=x.shape[0])

    def posterior(
        self,
        X: Vector | Matrix,
        y: Vector,
        **kwargs
    ):
        return lambda x: MultivariateNormal(cov=self.posterior_cov(X, **kwargs)(x), mean=jax.vmap(self.posterior_mean(X, y, **kwargs))(x))
