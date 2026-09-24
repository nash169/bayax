#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp

from bayax.operators.dense_operator import DenseOperator
from bayax.operators.linear_operator import LinearOperator
from bayax.operators.low_rank_operator import LowRankOperator
from bayax.utils.types import Scalar, Size, Vector, Matrix


@jax.tree_util.register_pytree_node_class
class ScaledOperator(LinearOperator):
    def __init__(self, scalar: Scalar, op: LinearOperator) -> None:
        if isinstance(op, ScaledOperator):
            scalar = scalar * op.scalar
            op = op.op
        self.scalar = scalar
        self.op = op

    def tree_flatten(self):
        if isinstance(self.scalar, jax.Array):
            return (self.scalar, self.op), {"scalar_is_static": False}
        return (self.op,), {"scalar_is_static": True, "scalar": self.scalar}

    @classmethod
    def tree_unflatten(cls, aux, children):
        if aux["scalar_is_static"]:
            return cls(aux["scalar"], children[0])
        return cls(children[0], children[1])

    def size(self) -> Size:
        return self.op.size()

    def mv(self, vec: Vector) -> Vector:
        return self.scalar * self.op.mv(vec)

    def solve(self, vec: Vector, **kwargs) -> Vector:
        try:
            return self.op.solve(vec, **kwargs) / self.scalar
        except (AttributeError, NotImplementedError, TypeError):
            return jnp.linalg.solve(self._dense_matrix(), vec) / self.scalar

    def invquad(self, vec: Vector, **kwargs) -> Scalar:
        try:
            return self.op.invquad(vec, **kwargs) / self.scalar
        except (AttributeError, NotImplementedError, TypeError):
            return vec @ self.solve(vec, **kwargs)

    def logdet(self, **kwargs) -> Scalar:
        try:
            logdet = self.op.logdet(**kwargs)
        except (AttributeError, NotImplementedError, TypeError):
            sign, logdet = jnp.linalg.slogdet(self._dense_matrix())
            logdet = jnp.where(sign > 0, logdet, jnp.inf)
        return self._logdet_dim(**kwargs) * jnp.log(self.scalar) + logdet

    def det(self) -> Scalar:
        try:
            det = self.op.det()
        except (AttributeError, NotImplementedError, TypeError):
            det = jnp.linalg.det(self._dense_matrix())
        return jnp.power(self.scalar, self._logdet_dim(pseudo=False)) * det

    def transpose(self) -> LinearOperator:
        try:
            return ScaledOperator(self.scalar, self.op.transpose())
        except (AttributeError, NotImplementedError, TypeError):
            return self.dense().transpose()

    def inverse(self) -> LinearOperator:
        try:
            return ScaledOperator(jnp.reciprocal(self.scalar), self.op.inverse())
        except (AttributeError, NotImplementedError, TypeError):
            return DenseOperator(jnp.linalg.inv(self._dense_matrix()) / self.scalar)

    def dense(self) -> DenseOperator:
        return DenseOperator(self.scalar * self._dense_matrix())

    def lowrank(self, **kwargs) -> LinearOperator:
        if isinstance(self.op, LowRankOperator):
            return self._scale_lowrank(self.op)
        try:
            op = self.op.lowrank(**kwargs)
            if isinstance(op, LowRankOperator):
                return self._scale_lowrank(op)
            return ScaledOperator(self.scalar, op)
        except (AttributeError, NotImplementedError, TypeError):
            eigval, eigvec = self._dense_eigh(**kwargs)
            return LowRankOperator(
                sval=eigval,
                left=eigvec,
                eps=kwargs.get("eps", 1e-8),
            )

    def _scale_lowrank(self, op: LowRankOperator) -> LowRankOperator:
        r"""
        Return the scaled operator as a LowRankOperator rather than a wrapper

        Every stored quantity is homogeneous of degree one in the scalar: the
        singular values, the eps that classifies kernel modes, and the image and
        kernel jitters. Scaling all of them leaves the kernel mask untouched and
        the effective spectrum scaled, for a positive scalar. `_right` is passed
        through so a symmetric operator stays symmetric.
        """
        return LowRankOperator(
            sval=self.scalar * op.sval,
            left=op.left,
            right=op._right,
            eps=self.scalar * op.eps,
            jimg=None if op.jimg is None else self.scalar * op.jimg,
            jker=None if op.jker is None else self.scalar * op.jker,
        )

    def diagonalize(self, **kwargs) -> tuple[Vector, Matrix]:
        try:
            eigval, eigvec = self.op.diagonalize(**kwargs)
            return self.scalar * eigval, eigvec
        except (AttributeError, NotImplementedError, TypeError):
            return self._dense_eigh(**kwargs)

    def sqrt(self, **kwargs) -> LinearOperator:
        try:
            return ScaledOperator(jnp.sqrt(self.scalar), self.op.sqrt(**kwargs))
        except (AttributeError, NotImplementedError, TypeError):
            return self._dense_sqrt()

    def sqrtf(self, **kwargs) -> LinearOperator:
        try:
            return ScaledOperator(jnp.sqrt(self.scalar), self.op.sqrtf(**kwargs))
        except (AttributeError, NotImplementedError, TypeError):
            return self._dense_sqrt_factor()

    def diag(self, **kwargs) -> Vector:
        if isinstance(self.op, LowRankOperator):
            return self.scalar * self.op.sval
        diag = getattr(self.op, "diag", None)
        if isinstance(diag, jax.Array):
            return self.scalar * diag
        if callable(diag):
            return self.scalar * diag(**kwargs)
        return self.scalar * jnp.diag(self._dense_matrix())

    def _logdet_dim(self, **kwargs) -> Scalar:
        r"""
        Return the number of modes the operand's logdet actually counts

        LowRankOperator.logdet(pseudo=True) omits the zero modes, so the scalar
        enters once per counted mode rather than once per dimension.
        """
        if not (isinstance(self.op, LowRankOperator) and kwargs.get("pseudo", True)):
            return self.size()[0]

        values = jnp.where(
            self.op.ker,
            0.0 if self.op.jker is None else self.op.jker,
            self.op.sval if self.op.jimg is None else self.op.sval + self.op.jimg,
        )
        counted = jnp.count_nonzero(values)

        if self.op.sval.shape[0] < self.op.left.shape[0] and self.op.jker is not None:
            counted += (self.op.left.shape[0] - self.op.sval.shape[0]) * (self.op.jker != 0)

        return counted

    def _dense_matrix(self) -> Matrix:
        if hasattr(self.op, "_mat"):
            return self.op._mat
        try:
            dense = self.op.dense()
            if hasattr(dense, "_mat"):
                return dense._mat
            if isinstance(dense, jax.Array):
                return dense
        except (AttributeError, NotImplementedError, TypeError):
            pass
        return self._operator_matrix(self.op)

    @staticmethod
    def _operator_matrix(op: LinearOperator) -> Matrix:
        m, n = op.size()
        eye = jnp.eye(n)
        return jax.vmap(op.mv, in_axes=1, out_axes=1)(eye).reshape(m, n)

    def _dense_sqrt(self) -> DenseOperator:
        mat = self.scalar * self._dense_matrix()
        eigval, eigvec = jnp.linalg.eigh(mat)
        eigval = jnp.clip(eigval, min=0.0)
        return DenseOperator((eigvec * jnp.sqrt(eigval)) @ eigvec.T)

    def _dense_sqrt_factor(self) -> DenseOperator:
        mat = self.scalar * self._dense_matrix()
        eigval, eigvec = jnp.linalg.eigh(mat)
        eigval = jnp.clip(eigval, min=0.0)
        return DenseOperator(eigvec * jnp.sqrt(eigval))

    def _dense_eigh(self, **kwargs) -> tuple[Vector, Matrix]:
        eigval, eigvec = jnp.linalg.eigh(self._dense_matrix())
        eigval, eigvec = self.scalar * eigval, eigvec
        eigval, eigvec = jnp.flip(eigval), jnp.flip(eigvec, axis=1)
        k = kwargs.get("k", None)
        if k is not None:
            eigval, eigvec = eigval[:k], eigvec[:, :k]
        return eigval, eigvec
