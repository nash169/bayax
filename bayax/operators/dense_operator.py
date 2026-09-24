#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp

from bayax.utils.types import Size, Scalar, Vector, Matrix, Optional
from bayax.operators.linear_operator import LinearOperator


@jax.tree_util.register_pytree_node_class
class DenseOperator(LinearOperator):
    def __init__(
        self,
        mat: Matrix
    ) -> None:
        self._mat = mat

    def tree_flatten(self):
        return (self._mat,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0])

    def __call__(self, vec: Vector) -> Vector:
        return self.mv(vec)

    def size(self) -> Size:
        r"""
        Return size of the linear operator
        """
        return self._mat.shape[0], self._mat.shape[1]

    def mv(self, vec: Vector) -> Vector:
        r"""
        Return matrix-vector multiplication of the linear operator
        """
        return jnp.matmul(self._mat, vec)

    def transpose(self) -> LinearOperator:
        r"""
        Return transposed matrix-vector multiplication of the linear operator
        """
        return DenseOperator(jnp.transpose(self._mat))

    def solve(
        self,
        vec: Vector
    ) -> Vector:
        r"""
        Return solve of the linear operator
        """
        assert self._mat.shape[0] == self._mat.shape[1], RuntimeError("Not valid operation for rectangular operators")
        return jnp.linalg.solve(self._mat, vec)

    def det(
        self,
    ) -> Scalar:
        r"""
        Return determinant of the linear operator
        """
        return jnp.linalg.det(self._mat)

    def logdet(
        self,
    ) -> Scalar:
        r"""
        Return log determinant of the linear operator
        """
        assert self._mat.shape[0] == self._mat.shape[1], RuntimeError("Not valid operation for rectangular operators")
        sign, logdet = jnp.linalg.slogdet(self._mat)
        return jnp.where(sign > 0, logdet, jnp.inf)

    def sqrtf(
        self,
    ) -> LinearOperator:
        r"""
        Return square root factor of the linear operator

        Cholesky factor L with L L^T equal to the operator, so the operator has
        to be symmetric positive definite.
        """
        assert self._mat.shape[0] == self._mat.shape[1], RuntimeError("Not valid operation for rectangular operators")
        return DenseOperator(jnp.linalg.cholesky(self._mat))

    def invquad(
        self,
        vec: Vector
    ) -> Scalar:
        r"""
        Return x^T A^-1 x for the linear operator A
        """
        return jnp.matmul(jnp.transpose(vec), self.solve(vec))

    def dense(
        self,
    ) -> LinearOperator:
        r"""
        Return dense matrix representation of the linear operator
        """
        return self

    def diag(
        self,
    ) -> Vector:
        r"""
        Return determinant of the linear operator
        """
        return self._mat.diagonal()

    def lowrank(
        self,
        *,
        k=None,
        eps: Scalar = 1e-8,
        method='dense',
        **kwargs
    ) -> LinearOperator:
        """Reduced SVD, optionally truncated to k singular modes."""
        from bayax.linalg.svd import svd
        from bayax.operators.low_rank_operator import LowRankOperator
        left, vals, right_t = svd(self._mat, k=k, method=method, **kwargs)
        return LowRankOperator(sval=vals, left=left, right=right_t.T, eps=eps)
