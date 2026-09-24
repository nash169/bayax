#!/usr/bin/env python
# encoding: utf-8

import warnings
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from bayax.utils.types import Array, Self, Scalar, Vector, Matrix, VectorInt


class LinearOperator(ABC):
    @abstractmethod
    def __init__(
        self,
        *args: Array
    ) -> None:
        r"""
        Define set of arrays needed for the linear operator
        """
        pass

    @abstractmethod
    def size(
        self
    ) -> VectorInt:
        r"""
        Return size of the linear operator
        """
        pass

    @abstractmethod
    def mv(
        self,
        vec: Vector,
        **kwargs
    ) -> Vector:
        r"""
        Return matrix-vector multiplication of the linear operator
        """
        pass

    @abstractmethod
    def transpose(
        self,
    ) -> Self:
        r"""
        Return transposed matrix-vector multiplication of the linear operator
        """
        raise NotImplementedError("The class {} requires a transpose function!".format(self.__class__.__name__))

# ====================================================================================================
# Default methods preferably to override
# ====================================================================================================
    def dense(
        self,
    ) -> Self:
        r"""
        Return dense matrix representation of the linear operator
        """
        warnings.warn("Default `dense` methods.")
        from bayax.operators.dense_operator import DenseOperator
        return DenseOperator(self(jnp.eye(self.size()[1])))

    def diag(
        self,
        method: str = 'hutchpp',
        seed: int = 0,
        **kwargs
    ) -> Vector:
        r"""
        Return solve of the linear operator
        """
        from bayax.utils.math import diag_exact, diag_hutch, diag_hutchpp
        warnings.warn("Default `diag` method.")

        m, n = self.size()
        assert m == n

        if method == 'exact':
            return diag_exact(self.mv, m)
        elif method == 'hutch':
            return diag_hutch(self.mv, m, jax.random.key(seed), **kwargs)
        elif method == 'hutchpp':
            return diag_hutchpp(self.mv, m, jax.random.key(seed), **kwargs)
        else:
            raise NotImplementedError(f'Method "{method}" not implemented')

    def solve(
        self,
        vec: Vector,
        **kwargs
    ) -> Vector:
        r"""
        Return solve of the linear operator
        """
        warnings.warn("Default `solve` methods.")
        return self.dense().solve(vec, **kwargs)

    def invquad(
        self,
        vec: Vector,
        **kwargs
    ) -> Scalar:
        r"""
        Return x^T A^-1 x for the linear operator A
        """
        # warnings.warn("Default `invquad` methods.")
        return jnp.dot(vec, self.solve(vec, **kwargs))

    def logdet(
        self,
    ) -> Scalar:
        r"""
        Return log determinant of the linear operator
        """
        if self.shape[0] == self.shape[1]:
            warnings.warn("Default `logdet` methods.")
            return jnp.linalg.slogdet(self.dense())
        else:
            raise NotImplementedError(f"Method not implemented.")

    def inverse(
        self,
    ) -> Self:
        r"""
        Return inverse operator
        """
        warnings.warn("Default `inverse` methods.")
        mv, solve = self.mv, self.solve
        self.mv, self.solve = solve, mv
        return self

# ====================================================================================================
# Operator specific methods
# ====================================================================================================
    def diagonalize(
        self,
    ) -> tuple[Vector, Matrix]:
        r"""
        Return determinant of the linear operator
        """
        raise NotImplementedError(f"Method not implemented.")

    def lowrank(self, *, k=None, eps=1e-8, method='dense', **kwargs) -> Self:
        """Singular triplets: exact SVD or matrix-free randomized/Lanczos.

        The exact method materializes the operator; iterative methods require
        k and forward/transpose products. eps thresholds singular values.
        """
        from bayax.linalg.svd import svd
        from bayax.operators.low_rank_operator import LowRankOperator
        left, values, right_t = svd(
            self.mv, shape=self.shape, rmv=self.transpose().mv,
            k=k, method=method, **kwargs,
        )
        return LowRankOperator(sval=values, left=left, right=right_t.T, eps=eps)

    def squareroot(
        self,
    ) -> Self:
        r"""
        Return low rank operator
        """
        raise NotImplementedError(f"Method not implemented.")

# ====================================================================================================
# Function Operator overloads
# ====================================================================================================
    def __call__(
        self,
        x: Vector | Matrix
    ) -> Vector:
        r"""
        Overload () operator
        """
        return self.mv(x) if x.ndim == 1 else jax.vmap(self.mv, 1, 1)(x)

    def __matmul__(self, other):
        if isinstance(other, jax.Array):
            return self(other)
        else:
            raise NotImplementedError(f"Matrix-Vector multiplication not supported for vector type {type(other)}")

    def __add__(
        self,
        other
    ):
        if isinstance(other, LinearOperator):
            from bayax.operators.sum_operator import SumOperator
            return SumOperator(self, other)
        else:
            raise NotImplementedError(f"Addition not supported for type {type(other)}")

    def __mul__(
            self,
            other
    ):
        if isinstance(other, LinearOperator):
            from bayax.operators.mul_operator import MulOperator
            return MulOperator(self, other)
        else:
            raise NotImplementedError(f"Multiplication not supported for type {type(other)}")

    @property
    def shape(self) -> VectorInt:
        return self.size()
