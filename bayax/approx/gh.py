#!/usr/bin/env python
# encoding: utf-8

import inspect
import numpy as np
import jax
import jax.numpy as jnp
from bayax.utils.types import Callable
from bayax.densities import Normal, MultivariateNormal


def gh(
    fn: Callable,
    p: Normal | MultivariateNormal,
    order: int = 5,
    **kwargs
):
    r"""
    Gauss-Hermite approximation of E_p[fn(x)] for a Gaussian p, with order nodes per dimension.

    As mc: a one-argument fn returns the expectation; otherwise the last argument of fn is
    integrated out and a function of the remaining arguments is returned. kwargs go to sqrtf.
    """
    fn_params = list(inspect.signature(fn).parameters.values())

    # nodes and weights of N(0, I), mapped to p
    z, w = np.polynomial.hermite_e.hermegauss(order)
    w = w / w.sum()
    if isinstance(p, Normal):
        x, weights = p.mean() + jnp.sqrt(p.var()) * jnp.asarray(z), jnp.asarray(w)
    else:
        dim = p.cov().size()[0]
        grid = lambda a: jnp.stack(jnp.meshgrid(*dim * [jnp.asarray(a)], indexing="ij"), axis=-1).reshape(-1, dim)
        nodes, weights = grid(z), jnp.prod(grid(w), axis=-1)
        x = (p.cov().sqrtf(**kwargs) @ nodes.T).T + (p.mean() if hasattr(p, "_mean") else 0.0)

    if len(fn_params) == 1:
        return jnp.tensordot(weights, jax.vmap(fn)(x), axes=1)
    else:
        def fn_vmap(*args):
            f = lambda x: fn(*args, **{fn_params[-1].name: x})
            return jnp.tensordot(weights, jax.vmap(f)(x), axes=1)
        return fn_vmap
