
#!/usr/bin/env python
# encoding: utf-8

from typing import Callable
import jax
import jax.numpy as jnp

from bayax.utils.types import Size, Scalar, Array, Vector, Matrix, Self
from bayax.operators.linear_operator import LinearOperator
from bayax.operators import *
from jax.scipy.sparse.linalg import cg


class SumOperator(LinearOperator):
    def __init__(
        self,
        *ops: LinearOperator
    ) -> None:
        r"""
        Define set of arrays needed for the linear operator
        """
        assert ops[0].shape == ops[1].shape, "Error: cannot sum operators with different dimension."
        self._ops = ops

    def size(
        self
    ) -> Size:
        r"""
        Return size of the linear operator
        """
        return self._ops[0].size()

    def mv(
        self,
        vec: Vector
    ) -> Vector:
        r"""
        Return matrix-vector multiplication of the linear operator
        """
        return self._ops[0](vec) + self._ops[1](vec)

    def transpose(
        self,
    ) -> LinearOperator:
        r"""
        Return transposed matrix-vector multiplication of the linear operator
        """
        return SumOperator(*[op.transpose() for op in self._ops])

    def solve(
        self,
        vec: Vector,
        **kwargs
    ) -> Vector:
        r"""
        Return solve of the linear operator
        """
        if any(isinstance(op, LowRankOperator) for op in self._ops) and any(isinstance(op, DiagOperator) for op in self._ops):
            lowrank_op = [op for op in self._ops if isinstance(op, LowRankOperator)][0]
            diag_op = [op for op in self._ops if isinstance(op, DiagOperator)][0]
            # Effective spectrum: eps classifies the kernel modes, jimg shifts
            # the image ones and jker fills the (possibly unstored) complement.
            complement = 0.0 if lowrank_op.jker is None else lowrank_op.jker
            values = jnp.where(
                lowrank_op.ker,
                complement,
                lowrank_op.sval if lowrank_op.jimg is None else lowrank_op.sval + lowrank_op.jimg,
            )
            base_diag = diag_op.diag + complement
            if jnp.ndim(base_diag) == 0:
                projected = lowrank_op.right.T @ vec
                res = lowrank_op.left @ (projected / (values + diag_op.diag))
                if lowrank_op.sval.shape[0] < lowrank_op.size()[0]:
                    res += (vec - lowrank_op.left @ projected) / base_diag
                return res
            weights = values - complement
            inv_vec = vec / base_diag
            inv_left = lowrank_op.left / base_diag[:, None]
            # Woodbury without inverting weights: they may be zero or negative.
            core = jnp.eye(weights.shape[0], dtype=weights.dtype) + weights[:, None] * (lowrank_op.right.T @ inv_left)
            rhs = weights * (lowrank_op.right.T @ inv_vec)
            return inv_vec - inv_left @ jnp.linalg.solve(core, rhs)
        elif any(isinstance(op, PSDOperator) for op in self._ops) and any(isinstance(op, DiagOperator) for op in self._ops):
            from bayax.linalg.woodbury_solve import woodbury_chol_solve
            psd_op = [op for op in self._ops if isinstance(op, PSDOperator)][0]
            diag_op = [op for op in self._ops if isinstance(op, DiagOperator)][0]
            if isinstance(psd_op._op, Callable):
                return cg(lambda v: psd_op(v) + diag_op.diag * v, vec, **kwargs)[0]
            else:
                return woodbury_chol_solve(psd_op._op, diag_op.diag, vec)
        else:
            raise NotImplementedError(f"Method not implemented.")

    def diagonalize(
        self,
        **kwargs
    ) -> tuple[Vector, Matrix]:
        if any(issubclass(type(op), SymOperator) for op in self._ops):
            from bayax.linalg.diagonalize import diagonalize
            return diagonalize(self.mv, self.shape[0], **kwargs)
        else:
            return super().diagonalize()

    def lowrank(
        self,
        **kwargs
    ) -> LinearOperator:
        from bayax.operators.low_rank_operator import LowRankOperator
        if all(issubclass(type(op), (LowRankOperator, SymOperator)) for op in self._ops):
            eigval, eigvec = self.diagonalize(**kwargs)
            return LowRankOperator(sval=eigval, left=eigvec)
        else:
            return super().lowrank()

    def sqrtf(
        self,
        **kwargs
    ) -> LinearOperator:
        if all(issubclass(type(op), (LowRankOperator, SymOperator)) for op in self._ops):
            if self.shape[0] * self.shape[1] <= 1e4:
                from bayax.operators import DenseOperator
                return DenseOperator(jnp.linalg.cholesky(self.dense()._mat))
            else:
                return self.lowrank(**kwargs).sqrtf()
        else:
            return super().sqrtf()
