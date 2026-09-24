#!/usr/bin/env python
# encoding: utf-8

import warnings
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from bayax.utils.types import Array, Self, Scalar, Vector, Matrix, VectorInt, Num, Optional, Int, Key, Tuple


class AbstractDensity(ABC):
    @abstractmethod
    def __init__(
        self,
        *args: Array
    ) -> None:
        r"""
        Define set of arrays needed for the probability density function
        """
        pass

    @abstractmethod
    def __call__(
        self,
        x: Num | Array
    ) -> Scalar:
        r"""
        Return density evaluation
        """
        raise NotImplementedError("The class {} requires a __call__ function.".format(self.__class__.__name__))

    def sample(
        self,
        rng_key: Optional[Key] = None,
        size: Optional[Int] = 1,
        **kwargs
    ) -> Array:
        raise NotImplementedError(f"Method not implemented.")

    def jvp(
        self,
        x: Vector,
        v: Vector,
        **kwargs
    ) -> Scalar:
        r"""
        Gradient with respect to the input.
        """
        raise NotImplementedError(f"Method not implemented.")

    def hvp(
        self,
        x: Vector,
        v: Vector,
        **kwargs
    ) -> Vector:
        r"""
        Hessian with respect to the input.
        """
        raise NotImplementedError(f"Method not implemented.")

    def jvp_params(
        self,
        **kwargs
    ) -> Tuple:
        r"""
        Return handles for gradient function with respect to the params.
        """
        raise NotImplementedError(f"Method not implemented.")

    def hvp_params(
        self,
        **kwargs
    ) -> Tuple:
        r"""
        Return handles for hessian function with respect to the params.
        """
        raise NotImplementedError(f"Method not implemented.")


@jax.tree_util.register_pytree_node_class
class ProjectedDensity(AbstractDensity):
    def __init__(self, density, lift, project, anchor):
        self.density = density
        self.lift = lift
        self.project = project
        self.anchor = anchor

    def tree_flatten(self):
        return (self.density, self.anchor), (self.lift, self.project)

    @classmethod
    def tree_unflatten(cls, aux, children):
        density, anchor = children
        lift, project = aux
        return cls(density, lift, project, anchor)

    # Pullback
    def __call__(self, x, **kwargs):
        return self.density(self.lift(x), **kwargs)

    # Pushforward
    def sample(self, key=None, **kwargs):
        samples = self.density.sample(key=key, **kwargs)
        if samples.ndim == 1:
            return self.project(samples, self.anchor)
        # Batched draws are stacked along the trailing axis; project each one.
        return jax.vmap(self.project, in_axes=(1, None), out_axes=1)(samples, self.anchor)

    def jvp(self, x, v, **kwargs):
        lifted, dlifted = jax.jvp(self.lift, (x,), (v,))
        return self.density.jvp(lifted, dlifted, **kwargs)

    def hvp(self, x, v, **kwargs):
        """Gauss-Newton pullback J^T H J v; the curvature of ``lift`` is dropped."""
        lifted, dlifted = jax.jvp(self.lift, (x,), (v,))
        return jax.vjp(self.lift, x)[1](self.density.hvp(lifted, dlifted, **kwargs))[0]
