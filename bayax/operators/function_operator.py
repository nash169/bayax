"""Rectangular linear maps backed by forward and transpose products."""

import jax
import jax.numpy as jnp

from bayax.operators.linear_operator import LinearOperator


class FunctionOperator(LinearOperator):
    """Wrap a linear callable without assembling its matrix.

    If rmv is omitted, JAX transposes mv. The callable must be linear in
    its argument. Like callable-backed PSDOperator, construct this wrapper
    inside jit/grad; it is not an array pytree to pass across transformations.
    """

    def __init__(self, mv, shape, *, rmv=None, dtype=None):
        self._mv = mv
        self._shape = tuple(shape)
        self.dtype = jnp.dtype(dtype) if dtype is not None else jnp.asarray(0.).dtype
        if rmv is None:
            transpose = jax.linear_transpose(mv, jnp.zeros(shape[1], dtype=self.dtype))
            rmv = lambda v: transpose(v)[0]
        self._rmv = rmv

    def size(self):
        return self._shape

    def mv(self, v):
        return self._mv(v)

    def transpose(self):
        return FunctionOperator(
            self._rmv, self._shape[::-1], rmv=self._mv, dtype=self.dtype,
        )

    def lowrank(self, **kwargs):
        kwargs.setdefault('dtype', self.dtype)
        return super().lowrank(**kwargs)
