#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.sparse.linalg import cg

from bayax.utils.types import Scalar, Vector, Matrix, Optional, Callable, Array, VectorInt, Self
from bayax.operators.linear_operator import LinearOperator
from bayax.operators import DenseOperator, SymOperator


@jax.tree_util.register_pytree_node_class
class PSDOperator(SymOperator):
    def __init__(
        self,
        op: Optional[Matrix | Callable] = None,
        op_type: Optional[str] = None,
        op_size: Optional[int] = None,
    ) -> None:
        r"""
        mat_type: ['raw', 'tril', 'triu']
        """
        if op is not None:
            if isinstance(op, jax.Array):
                assert (op_type is not None), "Matrix provided but type not defined [ 'raw', 'tril', 'triu' ]"
                if op_type == 'raw':
                    self._op = jnp.linalg.cholesky(op)
                    self._op_is_tril = True
                elif op_type == 'tril':
                    self._op = op
                    self._op_is_tril = True
                elif op_type == 'triu':
                    self._op = op
                    self._op_is_tril = False
                else:
                    msg = "invalid operator type [ 'raw', 'tril', 'triu' ]"
                    raise ValueError(msg)
                self._op_size = op.shape[0]
            elif isinstance(op, Callable):
                assert (op_size is not None), "Matrix-free operator provided; define operator dimension."
                self._op = op
                self._op_size = op_size
            else:
                msg = "invalid operator [ Matrix | Callable ]"

    def tree_flatten(self):
        if isinstance(self._op, Callable):
            raise TypeError(
                "callable-backed PSDOperator is not a valid pytree; materialize it "
                "(e.g. via `.dense()`) before passing it through jit/scan boundaries"
            )
        return (self._op,), {"op_is_tril": self._op_is_tril}

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls.__new__(cls)
        obj._op = children[0]
        obj._op_is_tril = aux["op_is_tril"]
        obj._op_size = getattr(children[0], "shape", (None,))[0]
        return obj

    def size(self) -> VectorInt:
        r"""
        Return size of the linear operator
        """
        return (self._op_size, self._op_size) if isinstance(self._op, Callable) else (self._op.shape[0], self._op.shape[1])

    def mv(self, vec: Vector) -> Vector:
        r"""
        Return matrix-vector multiplication of the linear operator
        """
        return self._op(vec) if isinstance(self._op, Callable) else jnp.matmul(self._op, jnp.matmul(jnp.transpose(self._op), vec))

    def solve(
        self,
        vec: Vector,
        **kwargs
    ) -> Vector:
        r"""
        Return solve of the linear operator
        """
        return cg(lambda v: self.mv(v), vec, **kwargs)[0] if isinstance(self._op, Callable) else jax.scipy.linalg.cho_solve((self._op, self._op_is_tril), vec)

    def logdet(
        self,
    ) -> Scalar:
        r"""
        Return determinant of the linear operator
        """
        return super().logdet() if isinstance(self._op, Callable) else 2 * jnp.sum(jnp.log(jnp.diag(self._op)))

    def invquad(
        self,
        vec: Vector,
        **kwargs
    ) -> Scalar:
        r"""
        Return x^T A^-1 x for the linear operator A
        """
        return super().invquad(vec, **kwargs) if isinstance(self._op, Callable) else jnp.sum(jnp.pow(jax.scipy.linalg.solve_triangular(self._op, vec, lower=self._op_is_tril), 2))

    def dense(
        self,
    ) -> LinearOperator:
        r"""
        Return dense matrix representation of the linear operator
        """
        return DenseOperator(self(jnp.eye(self._op_size))) if isinstance(self._op, Callable) else DenseOperator(self._op @ self._op.T)

    def lowrank(
        self,
        eps: Scalar = 1e-8,
        jimg: Optional[Scalar] = None,
        jker: Optional[Scalar] = None,
        **kwargs
    ) -> LinearOperator:
        from bayax.operators.low_rank_operator import LowRankOperator
        eigval, eigvec = self.diagonalize(**kwargs)
        return LowRankOperator(sval=eigval, left=eigvec, eps=eps, jimg=jimg, jker=jker)

    def sqrtf(
        self,
        **kwargs
    ) -> LinearOperator:
        if isinstance(self._op, Callable):
            if self._op_size <= 100:
                from bayax.operators import DenseOperator
                return DenseOperator(jnp.linalg.cholesky(self.dense()._mat))
            else:
                return self.lowrank(**kwargs).sqrtf()
        else:
            from bayax.operators import DenseOperator
            return DenseOperator(self._op)

    @staticmethod
    def random(
        key,
        m: int,
        op_type: str = "raw"
    ) -> Self:
        k1, k2 = jr.split(key)

        d = jr.uniform(k1, (m,))
        u = jnp.linalg.qr(jr.normal(k2, (m, m)))[0]
        mat = u @ jnp.diag(d) @ u.T

        if op_type == "raw":
            return PSDOperator(op=mat, op_type="raw")
        elif op_type == "tril":
            return PSDOperator(op=jnp.linalg.cholesky(mat, upper=False), op_type="tril")
        elif op_type == "triu":
            return PSDOperator(op=jnp.linalg.cholesky(mat, upper=True), op_type="triu")
        elif op_type == "mv":
            return PSDOperator(op=lambda v: mat @ v, op_size=m)
        else:
            raise NotImplementedError()
