#!/usr/bin/env python
# encoding: utf-8

import math
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from jaxtyping import ScalarLike

from bayax.utils.types import Scalar, Vector, Matrix, Key, Int
from bayax.densities.abstract_density import AbstractDensity
from bayax.operators.linear_operator import LinearOperator


@jax.tree_util.register_pytree_node_class
class MultivariateNormal(AbstractDensity):
    def __init__(
        self,
        cov: LinearOperator,
        mean: Optional[Scalar | Vector] = None,
    ) -> None:
        r"""
        Define normal distribution.
        covType: Float              -> spherical covariance
                 Vector             -> diagonal covariace
                 LinearOperator   -> full covariance
        """
        self._cov = cov
        if mean is not None:
            assert isinstance(mean, Scalar) or isinstance(mean, Vector), "Mean can only be a Vector or Scalar."
            self._mean = mean
        assert isinstance(cov, LinearOperator)

    def tree_flatten(self):
        has_mean = hasattr(self, "_mean")
        children = (self._cov, self._mean) if has_mean else (self._cov,)
        return children, {"has_mean": has_mean}

    @classmethod
    def tree_unflatten(cls, aux, children):
        if aux["has_mean"]:
            return cls(cov=children[0], mean=children[1])
        return cls(cov=children[0])

    def __call__(
        self,
        x: Vector,
        logdet_kwargs: Optional[dict] = None,
        invquad_kwargs: Optional[dict] = None,
    ) -> Vector:
        r"""
        Evaluate density.
        """
        return -0.5 * (self._cov.logdet(**(logdet_kwargs or {})) + self._cov.size()[0] * jnp.log(2 * jnp.pi) + self._cov.invquad(x - self._mean if hasattr(self, "_mean") else x, **(invquad_kwargs or {})))

    def mean(
        self,
    ):
        return self._mean

    def cov(
        self,
    ):
        return self._cov

    def sample(
        self,
        size: Int = 1,
        seed: Int = 0,
        key: Optional[Key] = None,
        **kwargs
    ):
        key = jax.random.key(seed) if key is None else key
        cov_sqrt = self._cov.sqrtf(**kwargs)
        rv = jax.random.normal(key=key, shape=(cov_sqrt.shape[1], ) if size == 1 else (cov_sqrt.shape[1], size))
        samples = cov_sqrt @ rv
        if hasattr(self, "_mean"):
            samples += self._mean if size == 1 else self._mean[:, None]
        return samples

    def jvp(
        self,
        x: Vector,
        v: Vector,
        **kwargs
    ) -> Scalar:
        r"""
        Gradient with respect to the input.
        """
        return jnp.dot(self._cov.solve(x - self._mean if hasattr(self, "_mean") else x, **kwargs), v)

    def hvp(
        self,
        x: Vector,
        v: Vector,
        **kwargs
    ) -> Vector:
        r"""
        Hessian with respect to the input.
        """
        return -self._cov.solve(v, **kwargs)

    def jvp_params(
        self,
        **kwargs
    ) -> Tuple:
        r"""
        Return handles for gradient function with respect to the params.
        """
        return lambda x, v: self._jvp_mean(x, v, **kwargs), lambda x, v: self._jvp_cov(x, v, **kwargs)

    def hvp_params(
        self,
        **kwargs
    ) -> Tuple:
        r"""
        Return handles for hessian function with respect to the params.
        """
        return lambda x, v: self._hvp_mean(x, v, **kwargs), lambda x, v: self._hvp_cov(x, v, **kwargs)

    def _jvp_mean(
        self,
        x: Vector,
        v: Vector,
        **kwargs
    ) -> Scalar:
        return jnp.dot(self._cov.solve(x - self._mean if hasattr(self, "_mean") else x, **kwargs), v)

    def _jvp_cov(
        self,
        x: Vector,
        v: Vector,
        **kwargs
    ) -> None:
        NotImplementedError("Jacobian vector product with respect to covariance matric not implemented yet")

    def _hvp_mean(
        self,
        x: Vector,
        v: Vector,
        **kwargs
    ) -> Vector:
        return -self._cov.solve(v, **kwargs)

    def _hvp_cov(
        self,
        x: Vector,
        v: Vector,
        **kwargs
    ) -> None:
        raise NotImplementedError("Hessian vector product with respect to covariance matric not implemented yet")
