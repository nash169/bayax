#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
from flax import nnx

from bayax.utils.types import Size, List, Optional, Callable, Array, Int, Vector
from bayax.utils.helper import pytree_to_array, array_to_pytree, wrap_pytree_function
from bayax.operators.linear_operator import LinearOperator
from bayax.utils.math import outer_jacobian, inner_jacobian


class MLP(nnx.Module):
    def __init__(
        self,
        layers: List[int],
        nl: Optional[Callable] = nnx.tanh,
        la: Optional[Callable] = None,
        seed: int = 0,
        **kwargs
    ):
        r"""
        Multilayer perceptron network.

        Args:
            layers: Network structure, e.g. [in_dim, layer_1_dim, ..., layer_n_dim, out_dim]
            nl: Non-linear activation function, e.g. nnx.tanh, nnx.relu
            la: Last layer activation function, e.g. nnx.sigmoid (binary classification), nnx.softmax (multi-class classification)
        """
        rngs = nnx.Rngs(seed)
        self.layers = nnx.List([nnx.Linear(m, n, rngs=rngs, **kwargs) for m, n in zip(layers[:-1], layers[1:])])
        if nl is not None:
            self.nl = nl
        if la:
            self.la = la

    def __call__(self, x: Array):
        for layer in self.layers[:-1]:
            x = self.nl(layer(x)) if hasattr(self, 'nl') else layer(x)
        x = self.layers[-1](x).squeeze()
        return self.la(x) if hasattr(self, "la") else x

    def features(self, x: Array, l: Int):
        assert l <= len(self.layers)
        if l == len(self.layers):
            return self(x)
        for i in range(l):
            x = self.nl(self.layers[i](x)) if hasattr(self, 'nl') else self.layers[i](x)
        return x

    def fwd_params(
        self
    ) -> Callable:
        graph_def, params = nnx.split(self)

        def fwd(x, p):
            return wrap_pytree_function(
                lambda p: nnx.call((graph_def, p))(x)[0],
                params
            )(p)
        return fwd

    def jvp(
        self
    ) -> Callable:
        graph_def, params = nnx.split(self)

        def jvp_fwd(x, p):
            model_lin = jax.linearize(lambda pvar: nnx.call((graph_def, pvar))(x)[0], array_to_pytree(p, params))[1]
            return wrap_pytree_function(model_lin, params)
        return jvp_fwd

    def linearized(
        self
    ) -> Callable:
        def fwd_linearized(x, p):
            return self(x) + self.jvp()(x, self.params)(p - self.params)
        return fwd_linearized

    def jtvp(
        self
    ) -> Callable:
        graph_def, params = nnx.split(self)

        def jtvp_fwd(x, p):
            model_lin = jax.linearize(lambda pvar: nnx.call((graph_def, pvar))(x)[0], array_to_pytree(p, params))[1]
            return lambda v: pytree_to_array(jax.linear_transpose(model_lin, array_to_pytree(p, params))(v)[0])

        return jtvp_fwd

    def vjp(
        self
    ) -> Callable:
        graph_def, params = nnx.split(self)

        def vjp_fwd(x, p):
            jt = jax.linearize(lambda pvar: nnx.call((graph_def, pvar))(x)[0], array_to_pytree(p, params))[1]
            return jt
        return wrap_pytree_function(vjp_fwd, params)

    def jtj(
        self,
        h: Optional[LinearOperator] = None
    ) -> Callable:
        graph_def, params = nnx.split(self)

        def jtj_fn(x, p):
            return wrap_pytree_function(
                lambda v: inner_jacobian(lambda pvar: nnx.call((graph_def, pvar))(x)[0], h)(array_to_pytree(p, params), v),
                params
            )
        return jtj_fn

    def ntk(
        self,
        h: Optional[LinearOperator] = None
    ) -> Callable:
        graph_def, params = nnx.split(self)

        def ntk_fn(x, p):
            return lambda v: outer_jacobian(lambda pvar: nnx.call((graph_def, pvar))(x)[0], h)(array_to_pytree(p, params), v)
        return ntk_fn

    @property
    def params(self) -> Vector:
        return pytree_to_array(nnx.state(self))

    @params.setter
    def params(self, value):
        nnx.update(self, array_to_pytree(value, nnx.state(self)))
        # self = nnx.merge(nnx.split(self)[0], value)

    @property
    def size(self) -> Int:
        return sum(p.size for p in jax.tree_util.tree_leaves(nnx.state(self)))

    @property
    def shape(self) -> Size:
        r"""
        Return network input-ouput dimensions
        """
        return jnp.array([self.layers[0].in_features, self.layers[-1].out_features])
