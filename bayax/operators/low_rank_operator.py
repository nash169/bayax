#!/usr/bin/env python
# encoding: utf-8

from bayax.operators.dense_operator import DenseOperator
import jax
import jax.numpy as jnp
import jax.random as jr

from bayax.operators.linear_operator import LinearOperator
from bayax.utils.types import Size, Scalar, Vector, Matrix, Optional, Self


@jax.tree_util.register_pytree_node_class
class LowRankOperator(LinearOperator):
    def __init__(
        self,
        sval: Vector,
        left: Matrix,
        right: Optional[Matrix] = None,
        eps: Scalar = 1e-8,
        jimg: Optional[Scalar] = None,
        jker: Optional[Scalar] = None,
    ) -> None:
        if right is not None and (jimg is not None or jker is not None):
            raise NotImplementedError(
                "Jitter implemented only for symmetric operators."
            )
        self.sval, self.left, self._right, self.eps, self.jimg, self.jker = sval, left, right, eps, jimg, jker

    @property
    def right(self):
        return self.left if self._right is None else self._right

    @property
    def ker(self):
        return self.sval <= jnp.expand_dims(self.eps, -1)

    def tree_flatten(self):
        return (self.sval, self.left, self._right, self.eps, self.jimg, self.jker), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls.__new__(cls)
        obj.sval, obj.left, obj._right, obj.eps, obj.jimg, obj.jker = children
        return obj

    def size(self) -> Size:
        r"""
        Return size of the linear operator
        """
        return self.left.shape[0], self.right.shape[0]

    def transpose(
        self,
    ) -> Self:
        r"""
        Return transposed matrix-vector multiplication of the linear operator
        """
        if self._right is None:
            return self
        return LowRankOperator(sval=self.sval, left=self.right, right=self.left, eps=self.eps, jimg=self.jimg, jker=self.jker)

    def mv(self, x: Vector) -> Vector:
        r"""
        Return matrix-vector multiplication of the linear operator
        """
        d, u, v = jnp.where(self.ker, 0.0 if self.jker is None else self.jker,
                            self.sval if self.jimg is None else self.sval + self.jimg), self.left, self.right

        vtx = v.T @ x
        res = u @ (d * vtx)

        if d.shape[0] < u.shape[0] and self.jker is not None:
            res += self.jker * (x - v @ vtx)

        return res

    def solve(
        self,
        x: Vector,
    ) -> Vector:
        if self.left.shape[0] != self.right.shape[0]:
            raise NotImplementedError(
                "Solve not implemented for non-square operators."
            )
        return self.inverse()(x)

    def logdet(
        self,
        pseudo: bool = True,
    ) -> Scalar:
        r"""
        Return log absolute determinant of the linear operator.

        With pseudo=True, sum logs of nonzero effective singular/eigenvalues
        after applying eps and jitter, omitting zero modes (including unstored
        kernel modes). An operator with no nonzero modes returns zero.
        """
        if self.left.shape[0] != self.right.shape[0]:
            raise NotImplementedError(
                "Logdet not implemented for non-square operators."
            )

        d = jnp.where(self.ker, 0.0 if self.jker is None else self.jker, self.sval if self.jimg is None else self.sval + self.jimg)
        # Mask before taking logs so omitted zero modes have finite gradients.
        d = jnp.where(pseudo & (d == 0), 1.0, d)
        res = jnp.sum(jnp.log(d))

        if self.sval.shape[0] < self.left.shape[0]:
            jker = 0.0 if self.jker is None else self.jker
            jker = jnp.where(pseudo & (jker == 0), 1.0, jker)
            res += jnp.log(jker) * (self.left.shape[0] - self.sval.shape[0])

        return res

    def invquad(
        self,
        vec: Vector
    ) -> Scalar:
        r"""
        Return x^T A^-1 x for the linear operator A
        """
        return vec @ self.solve(vec)

    def dense(
        self,
    ) -> DenseOperator:
        r"""
        Return dense matrix representation of the linear operator
        """
        d, u, v = jnp.where(self.ker, 0.0 if self.jker is None else self.jker,
                            self.sval if self.jimg is None else self.sval + self.jimg), self.left, self.right
        res = (u * d) @ v.T

        if d.shape[0] < u.shape[0] and self.jker is not None:
            res += self.jker * (jnp.eye(u.shape[0], dtype=res.dtype) - u @ u.T)

        return DenseOperator(res)

    def inverse(
            self
    ) -> LinearOperator:
        d = jnp.where(self.ker, 0.0 if self.jker is None else self.jker,
                      self.sval if self.jimg is None else self.sval + self.jimg)
        # Mask before dividing as well, so zero modes have finite gradients.
        d = jnp.where(d == 0, 0.0, jnp.reciprocal(jnp.where(d == 0, 1.0, d)))
        jker = self.jker
        if jker is not None:
            jker = jnp.where(jker == 0, 0.0, jnp.reciprocal(jnp.where(jker == 0, 1.0, jker)))
        return LowRankOperator(sval=d, left=self.right, right=None if self._right is None else self.left, eps=0.0, jker=jker)

    # square root
    def sqrt(
            self
    ) -> LinearOperator:
        if self._right is not None:
            raise NotImplementedError(
                "Square root implemented only for symmetric operators."
            )
        # d = jnp.where(self.ker, 0.0, jnp.sqrt(self.sval) if self.jimg is None else jnp.sqrt(self.sval + self.jimg))
        # return LowRankOperator(sval=d, left=self.left, eps=self.eps, jker=None if self.jker is None else jnp.sqrt(self.jker))
        d = jnp.where(self.ker, 0.0 if self.jker is None else self.jker, self.sval if self.jimg is None else self.sval + self.jimg)
        d = jnp.where(d == 0, 0.0, jnp.sqrt(jnp.where(d == 0, 1.0, d)))
        jker = self.jker
        if jker is not None:
            jker = jnp.where(jker == 0, 0.0, jnp.sqrt(jnp.where(jker == 0, 1.0, jker)))
        return LowRankOperator(sval=d, left=self.left, eps=0.0, jker=jker)

    # square root factor
    def sqrtf(
            self,
    ) -> LinearOperator:
        if self._right is not None:
            raise NotImplementedError(
                "Square root factor implemented only for symmetric operators."
            )
        root = self.sqrt()
        if self.sval.shape[0] < self.left.shape[0] and self.jker is not None:
            return root
        return DenseOperator(self.left * root.sval)

    def topcut(
        self,
        num_modes
    ) -> Self:
        self.sval, self.left = self.sval[:num_modes], self.left[:, :num_modes]
        if self._right is not None:
            self._right = self._right[:, :num_modes]
        return self

    def bottomcut(
        self,
        num_modes
    ) -> Self:
        if num_modes == 0:
            return self.topcut(0)
        self.sval, self.left = self.sval[-num_modes:], self.left[:, -num_modes:]
        if self._right is not None:
            self._right = self._right[:, -num_modes:]
        return self

    @staticmethod
    def random(
        key,
        m: int,
        n: Optional[int] = None,
        r: Optional[int] = None,
        **kwargs
    ) -> Self:
        k1, k2, k3 = jr.split(key, 3)

        if r is None:
            r = m if n is None else min(m, n)

        return LowRankOperator(sval=jr.uniform(k1, (r,)), left=jnp.linalg.qr(jr.normal(k2, (m, r)))[0], right=jnp.linalg.qr(jr.normal(k3, (n, r)))[0] if n is not None else None, **kwargs)
