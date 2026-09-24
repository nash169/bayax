#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp

from bayax.utils.types import Callable, Optional
from bayax.operators import DenseOperator, DiagOperator, FunctionOperator, LowRankOperator, PSDOperator
from bayax.operators.linear_operator import LinearOperator


def pullmetric(
    f: Callable,
    h: Optional[LinearOperator] = None
) -> Callable:
    r"""
    Calculate the manifold metric via pullback of the embedding space metric

    Args:
        f: Embedding map
        h: Ambient (embedding space) metric

    Returns:
        Metric vector product
    """
    def pullmetric_fn(x, v):
        jvp = jax.linearize(f, x)[1]
        hjv = h(jvp(v)) if h is not None else jvp(v)
        return jax.linear_transpose(jvp, v)(hjv)[0]
    return pullmetric_fn


def pullmetric_grad(
    f: Callable,
    h: Optional[LinearOperator] = None
) -> Callable:
    r"""
    Calculate the directional derivative of the pullback metric vector product

    Args:
        f: Embedding map
        h: Constant ambient (embedding space) metric

    Returns:
        Callable (x, v, u) computing (Dg(x)[v])u with u held fixed,
        without materializing the metric or its derivative tensor
    """
    def fn(x, v, u):
        # Joint linear map w -> (Jw, D^2f(x)[v, w]).
        jvp = jax.linearize(lambda x: jax.jvp(f, (x,), (v,)), x)[1]
        ju, hess = jvp(u)
        hju = h(ju) if h is not None else ju
        hhess = h(hess) if h is not None else hess
        # J^T H D^2f[v, u] + D^2f[v, .]^T H Ju.
        return jax.linear_transpose(jvp, u)((hhess, hju))[0]

    return fn


def christoffel_fk(f: Callable, h: Optional[LinearOperator] = None, contraction="ij"):
    r"""
    Contract first-kind Christoffel symbols T_kij of the embedding metric.

    Args:
        f: Embedding map
        h: Constant ambient (embedding space) metric
        contraction: "ij" contracts the last two indices; "ki" the first two.

    Returns:
        Callable (x, v, u), with v and u held fixed in position derivatives.
        For "ij": sum_ij T_kij v_i u_j = J^T H D^2f[v, u].
        For "ki": sum_ki T_kij v_k u_i = D^2f[u, .]^T H Jv.
    """

    if contraction == "ij":
        def fn(x, v, u):
            jvp = jax.linearize(f, x)[1]
            hess = jax.jvp(lambda x: jax.jvp(f, (x,), (u,))[1], (x,), (v,))[1]
            return jax.linear_transpose(jvp, v)(h(hess) if h is not None else hess)[0]
    elif contraction == "ki":
        def fn(x, v, u):
            jv = jax.jvp(f, (x,), (v,))[1]
            _, hess_pullback = jax.vjp(lambda x: jax.jvp(f, (x,), (u,))[1], x)
            return hess_pullback(h(jv) if h is not None else jv)[0]
    else:
        raise ValueError('contraction must be "ij" or "ki"')

    return fn


def christoffel_sk(
    f: Callable,
    h: Optional[LinearOperator] = None,
    g: Optional[Callable] = None,
    g_inv: Optional[Callable] = None,
    contraction="ij",
):
    r"""
    Contract second-kind Christoffel symbols T^k_ij of the embedding metric.

    Args:
        f: Embedding map
        h: Constant ambient (embedding space) metric
        g: Optional metric-vector product (x, v); defaults to the pullback metric.
        g_inv: Optional inverse-metric product (x, v); otherwise use CG.
        contraction: "ij" contracts the last two indices; "ki" the first two.

    Returns:
        Callable (x, v, u).
        For "ij": sum_ij T^k_ij v^i u^j, applying G^-1 to the first-kind output.
        For "ki": sum_ki T^k_ij v_k u^i, applying G^-1 to the first input v.
        The "ki" case takes a covector v and returns a covector indexed by j.
    """
    fk = christoffel_fk(f, h, contraction=contraction)

    if g_inv is None:
        g_op = pullmetric(f, h) if g is None else g

        def g_inv(x, v):
            return jax.scipy.sparse.linalg.cg(lambda w: g_op(x, w), v)[0]

    if contraction == "ij":
        def fn(x, v, u):
            return g_inv(x, fk(x, v, u))
    else:
        def fn(x, v, u):
            return fk(x, g_inv(x, v), u)

    return fn


# class EigenEmbeddedGeometry:
#     def __init__(self, f, H, x, **kwargs):
#         self.metric = PSDOperator(lambda v: pullmetric(f, H)(x, v), op_size=x.shape[0]).lowrank(**kwargs)
#         self.metric_inv = self.metric.inverse()
#         self.metric_inv_sqrt = self.metric_inv.sqrtf()._mat
#
#         self.cfk = lambda u, v: christoffel_fk(f, H)(x, u, v)
#
#     def metric_solve(self, v):
#         return self.metric_inv @ v
#
#     def metric_sqrt_solve(self, v):
#         return self.metric_inv_sqrt @ v
#
#     def cfk_contraction(self, v):
#         return self.cfk(v, v)


@jax.tree_util.register_pytree_node_class
class EmbeddingGeom:
    r"""Cache embedding derivatives and metric factors at a fixed position.

    First-kind Christoffel indices follow
    Gamma[k, i, j] = <partial_k f, partial_i partial_j f>_H.
    """

    def __init__(self, f, H, x, **kwargs):
        self.H = H
        self.J, self.back_to_q = jax.vjp(
            lambda z: jax.linearize(f, z)[1], x
        )
        self.JT = jax.linear_transpose(self.J, x)
        self.metric = PSDOperator(self.metric_mv, op_size=x.shape[0]).lowrank(**kwargs)
        self.metric_inv = self.metric.inverse()

    def tree_flatten(self):
        return (self.H, self.J, self.back_to_q, self.JT, self.metric, self.metric_inv), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls.__new__(cls)
        obj.H, obj.J, obj.back_to_q, obj.JT, obj.metric, obj.metric_inv = children
        return obj

    def metric_mv(self, v):
        jv = self.J(v)
        hjv = self.H(jv) if self.H is not None else jv
        return self.JT(hjv)[0]

    def metric_solve(self, v):
        return self.metric_inv @ v

    def metric_logdet(self):
        return self.metric.logdet()

    def cfk_mv(self, V, contraction="ij", weights=None):
        r"""
        Contract first-kind Gamma[k, i, j], summing over columns for matrix V.

        For "ij": sum_ij Gamma[k, i, j] v[i] v[j] = J^T H D^2f[v, v].
        For "ki": sum_ki Gamma[k, i, j] v[k] v[i] = D^2f[v, .]^T H Jv.

        A matrix V contracts with V @ V.T. For an inverse-metric square-root
        factor, "ij" gives the trace used in diffusion drift; "ki" gives half
        the gradient of logdet(G) when the metric is full-rank.

        Optional weights multiply column contributions (or a scalar for a vector).
        V and weights are held fixed in position derivatives and H is constant in x.
        Cached derivatives are reused without materializing derivative tensors.
        """
        if contraction == "ij":
            def contract(v):
                jv, back_to_J = jax.vjp(lambda J: J(v), self.J)
                # This linear map is a -> D^2f(x)[v, .]^T a.
                hess_transpose = lambda a: self.back_to_q(back_to_J(a)[0])[0]
                hess_vv = jax.linear_transpose(hess_transpose, jv)(v)[0]
                hhess_vv = self.H(hess_vv) if self.H is not None else hess_vv
                return self.JT(hhess_vv)[0]

            result = contract(V) if V.ndim == 1 else jax.vmap(contract, in_axes=1, out_axes=1)(V)
            if weights is not None:
                result = result * weights
            return result if V.ndim == 1 else result.sum(axis=1)
        elif contraction == "ki":
            def apply(J):
                if V.ndim == 1:
                    return J(V)
                return jax.vmap(J, in_axes=1, out_axes=1)(V)

            JV, back_to_J = jax.vjp(apply, self.J)
            HJV = self.H(JV) if self.H is not None else JV
            if weights is not None:
                HJV = HJV * weights
            return self.back_to_q(back_to_J(HJV)[0])[0]
        else:
            raise ValueError('contraction must be "ij" or "ki"')


@jax.tree_util.register_pytree_node_class
class EmbeddingGeomSqrtf:
    r"""Cache M = J_f(x)^T L and its pullback for constant H = L L^T.

    L is an array or LinearOperator with shape (ambient dimension, factor rank).
    Only the thin matrix M is assembled, not J, H, or the parameter-space metric.
    metric_sqrt stores its reduced SVD; back_to_q differentiates M before SVD.

    All factor modes are retained for metric_mv and cfk_mv, matching the original
    pullback geometry in EmbeddingGeom. k, eps, jimg, and jker affect metric and
    its inverse only. eps thresholds squared singular values (metric eigenvalues).
    """

    def __init__(self, f, L, x, *, k=None, eps=1e-8, jimg=None, jker=None):
        if isinstance(L, LinearOperator):
            L = L._mat if isinstance(L, DenseOperator) else L.dense()._mat

        def pullfactor(z):
            _, back_to_z = jax.vjp(f, z)
            return jax.vmap(lambda l: back_to_z(l)[0], in_axes=1, out_axes=1)(L)

        M, self.back_to_q = jax.vjp(pullfactor, x)
        if k is not None and not 1 <= k <= min(M.shape):
            raise ValueError('k must be between 1 and min(M.shape)')
        left, sval, right_t = jnp.linalg.svd(M, full_matrices=False)
        self.metric_sqrt = LowRankOperator(
            sval=sval, left=left, right=right_t.T, eps=0.0,
        )
        self.metric = LowRankOperator(
            sval=sval[:k]**2, left=left[:, :k], eps=eps, jimg=jimg, jker=jker,
        )
        self.metric_inv = self.metric.inverse()
        self.metric_inv_sqrt = self.metric_inv.sqrtf()

    def tree_flatten(self):
        return (self.metric_sqrt, self.back_to_q, self.metric, self.metric_inv, self.metric_inv_sqrt), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls.__new__(cls)
        obj.metric_sqrt, obj.back_to_q, obj.metric, obj.metric_inv, obj.metric_inv_sqrt = children
        return obj

    def metric_mv(self, v):
        return self.metric_sqrt(self.metric_sqrt.transpose()(v))

    def metric_solve(self, v):
        return self.metric_inv @ v

    def metric_logdet(self):
        return self.metric.logdet()

    def cfk_mv(self, V, contraction="ij", weights=None):
        r"""Contract first-kind Gamma[k, i, j], summing over columns of V.

        "ij" returns M (DM[v]^T v); "ki" returns DM^*[v (M^T v)^T].
        Optional weights multiply column contributions (or a scalar for a vector).
        V and weights are held fixed. Derivatives reuse the pullback of the thin factor,
        without differentiating singular vectors or forming a Hessian tensor.
        """
        if contraction == "ij":
            shape = jax.ShapeDtypeStruct(self.metric_sqrt.shape, self.metric_sqrt.left.dtype)
            factor_jvp = jax.linear_transpose(self.back_to_q, shape)

            def contract(v):
                dm = factor_jvp((v,))[0]
                return self.metric_sqrt(dm.T @ v)

            result = contract(V) if V.ndim == 1 else jax.vmap(contract, in_axes=1, out_axes=1)(V)
            if weights is not None:
                result = result * weights
            return result if V.ndim == 1 else result.sum(axis=1)
        elif contraction == "ki":
            columns = V[:, None] if V.ndim == 1 else V
            mtv = self.metric_sqrt.transpose()(columns)
            if weights is not None:
                mtv = mtv * weights
            return self.back_to_q(columns @ mtv.T)[0]
        else:
            raise ValueError('contraction must be "ij" or "ki"')


@jax.tree_util.register_pytree_node_class
class EmbeddingCache:
    """
    Cache embedding derivatives and metric factors at a fixed position

    With lowrank=True, use a reduced SVD of M = J.T @ L, where H = L @ L.T.
    H_sqrtf optionally supplies L as an array or operator; k retains the
    leading singular modes. eps, jimg and jker apply to the metric eigenvalues
    (squared singular values). For lowrank=True, method='dense' materializes
    M for exact reduced SVD; method='randomized' or 'lanczos' uses only M/M.T
    products. Iterative methods require k. Other kwargs configure lowrank/eigh.

    Contractions differentiate the original embedding metric, even when the
    inverse is truncated. Use EmbeddingCacheSVD for truncated-metric derivatives.
    """

    def __init__(
        self, f, H, x, *, lowrank=False, H_sqrtf=None, k=None,
        eps=1e-8, jimg=None, jker=None, **kwargs,
    ):
        self.H = H
        self.H_sqrtf = H_sqrtf
        self.metric_sqrt = None
        self.factor_pullback = None
        self.J, self.back_to_q = jax.vjp(
            lambda z: jax.linearize(f, z)[1], x
        )
        self.JT = jax.linear_transpose(self.J, x)
        if lowrank:
            if self.H_sqrtf is None:
                self.H_sqrtf = H.sqrtf() if H is not None else DiagOperator(
                    jnp.array(1., dtype=x.dtype), dim=jax.eval_shape(self.J, x).shape[0]
                )
        if self.H_sqrtf is not None and not isinstance(self.H_sqrtf, LinearOperator):
            self.H_sqrtf = DenseOperator(self.H_sqrtf)

        if lowrank:
            self.metric_sqrt = self._svd_factor(self.J, x, k, **kwargs)
            self.metric = LowRankOperator(
                sval=self.metric_sqrt.sval**2, left=self.metric_sqrt.left,
                eps=eps, jimg=jimg, jker=jker,
            )
        else:
            self.metric = PSDOperator(self.metric_mv, op_size=x.shape[0]).lowrank(
                k=k, eps=eps, jimg=jimg, jker=jker, **kwargs
            )
        self.metric_inv = self.metric.inverse()

    def tree_flatten(self):
        return (
            self.H, self.H_sqrtf, self.J, self.back_to_q, self.JT,
            self.metric_sqrt, self.factor_pullback, self.metric, self.metric_inv,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls.__new__(cls)
        (
            obj.H, obj.H_sqrtf, obj.J, obj.back_to_q, obj.JT,
            obj.metric_sqrt, obj.factor_pullback, obj.metric, obj.metric_inv,
        ) = children
        return obj

    def _pullfactor_operator(self, J, x):
        JT = jax.linear_transpose(J, x)
        L = self.H_sqrtf
        LT = L.transpose()
        return FunctionOperator(
            lambda v: JT(L(v))[0], (x.size, L.shape[1]),
            rmv=lambda v: LT(J(v)), dtype=x.dtype,
        )

    def _pullfactor(self, J, x):
        # The full factor is needed only by the original-metric trace gradient.
        M = self._pullfactor_operator(J, x)
        return M(jnp.eye(M.shape[1], dtype=x.dtype))

    def _svd_factor(self, J, x, k, **kwargs):
        M = self._pullfactor_operator(J, x)
        return M.lowrank(k=k, eps=0.0, **kwargs)

    def _ambient_mv(self, V):
        if self.H_sqrtf is not None:
            return self.H_sqrtf(self.H_sqrtf.transpose()(V))
        return self.H(V) if self.H is not None else V

    def metric_mv(self, v):
        jv = self.J(v)
        return self.JT(self._ambient_mv(jv))[0]

    def cfk_mv(self, V, contraction="ij", weights=None):
        r"""
        Contract first-kind Gamma[k, i, j], summing over columns for matrix V.

        "ij" leaves k free: J^T H D^2f[v, v].
        "ki" leaves j free: D^2f[v, .]^T H Jv.
        Optional weights multiply column contributions (or a scalar for a vector).
        Signed weights are supported without constructing V diag(weights) V^T.

        V and weights are held fixed in position derivatives. The cached Jacobian
        residuals and their pullback avoid rebuilding the embedding derivatives.
        """
        if contraction == "ij":
            def contract(v):
                jv, back_to_J = jax.vjp(lambda J: J(v), self.J)
                hess_transpose = lambda a: self.back_to_q(back_to_J(a)[0])[0]
                hess_vv = jax.linear_transpose(hess_transpose, jv)(v)[0]
                return self.JT(self._ambient_mv(hess_vv))[0]

            result = contract(V) if V.ndim == 1 else jax.vmap(contract, in_axes=1, out_axes=1)(V)
            if weights is not None:
                result = result * weights
            return result if V.ndim == 1 else result.sum(axis=1)
        if contraction != "ki":
            raise ValueError('contraction must be "ij" or "ki"')

        def apply(J):
            if V.ndim == 1:
                return J(V)
            return jax.vmap(J, in_axes=1, out_axes=1)(V)

        JV, back_to_J = jax.vjp(apply, self.J)
        HJV = self._ambient_mv(JV)
        if weights is not None:
            HJV = HJV * weights
        return self.back_to_q(back_to_J(HJV)[0])[0]

    def half_grad_logdet(self):
        """Contract the original metric derivative with the cached inverse."""
        if self.metric_sqrt is None:
            return self.cfk_mv(self.metric_inv.sqrtf().dense()._mat, contraction="ki")

        inverse = self.metric_inv
        complement = inverse.jker if inverse.jker is not None else 0.0
        weights = jnp.where(inverse.ker, complement, inverse.sval)
        if inverse.sval.size < inverse.left.shape[0] and inverse.jker is not None:
            # G^-1 = complement * I + U diag(weights - complement) U.T.
            # Differentiate ||J.T L||_F^2 / 2 for the identity contribution,
            # avoiding a parameter-space identity or inverse-square-root matrix.
            x = jnp.zeros((inverse.left.shape[0],), dtype=inverse.left.dtype)
            M, back_to_J = jax.vjp(lambda J: self._pullfactor(J, x), self.J)
            trace_grad = self.back_to_q(back_to_J(M)[0])[0]
            return complement * trace_grad + self.cfk_mv(
                inverse.left, contraction="ki", weights=weights - complement
            )
        return self.cfk_mv(inverse.left, contraction="ki", weights=weights)

    def metric_solve(self, v):
        return self.metric_inv @ v

    def metric_logdet(self):
        return self.metric.logdet()


@jax.tree_util.register_pytree_node_class
class EmbeddingCacheSVD(EmbeddingCache):
    """
    Cache exact derivatives of the SVD-defined metric

    The SVD and its pullback are prepared once per position. A positive jker
    supplies the complementary metric when fewer than dimension modes remain.
    Matrix-free methods differentiate their computed approximation; keep the
    factorization key fixed across positions. SVD/QR derivatives require
    nondegenerate factors (distinct retained singular values and no breakdown).
    """

    def __init__(self, f, H, x, *, lowrank=True, **kwargs):
        super().__init__(f, H, x, lowrank=lowrank, **kwargs)

    def _svd_factor(self, J, x, k, **kwargs):
        factorize = super()._svd_factor
        factor, self.factor_pullback = jax.vjp(
            lambda J: factorize(J, x, k, **kwargs), J
        )
        return factor

    def _metric_from_factor(self, factor):
        return LowRankOperator(
            sval=factor.sval**2, left=factor.left,
            eps=self.metric.eps, jimg=self.metric.jimg, jker=self.metric.jker,
        )

    def _factor_gradient(self, cotangent):
        return self.back_to_q(self.factor_pullback(cotangent)[0])[0]

    def cfk_mv(self, V, contraction="ij", weights=None):
        r"""Contract Christoffels of the effective SVD-defined metric.

        "ki" differentiates tr(V.T @ metric @ V) / 2 with V held fixed.
        "ij" uses Gamma[., v, v] = (DG[v])v - Gamma[v, v, .].
        Optional signed weights multiply column contributions. Cached factor
        derivatives are reused; the SVD is not recomputed.
        """
        if self.metric_sqrt is None:
            return super().cfk_mv(V, contraction=contraction, weights=weights)
        if contraction == "ij":
            factor_jvp = jax.linear_transpose(self._factor_gradient, self.metric_sqrt)

            def directional_metric(v):
                tangent = factor_jvp(v)[0]
                return jax.jvp(
                    lambda factor: self._metric_from_factor(factor)(v),
                    (self.metric_sqrt,), (tangent,),
                )[1]

            result = directional_metric(V) if V.ndim == 1 else jax.vmap(directional_metric, in_axes=1, out_axes=1)(V)
            if weights is not None:
                result = result * weights
            result = result if V.ndim == 1 else result.sum(axis=1)
            return result - self.cfk_mv(V, contraction="ki", weights=weights)
        if contraction != "ki":
            raise ValueError('contraction must be "ij" or "ki"')

        _, pullback = jax.vjp(
            lambda factor: self._metric_from_factor(factor)(V), self.metric_sqrt
        )
        cotangent = 0.5 * V if weights is None else 0.5 * V * weights
        return self._factor_gradient(pullback(cotangent)[0])

    def half_grad_logdet(self):
        """Differentiate the factor-based log determinant without forming G^-1."""
        if self.metric_sqrt is None:
            return super().half_grad_logdet()
        cotangent = jax.grad(
            lambda factor: 0.5 * self._metric_from_factor(factor).logdet()
        )(self.metric_sqrt)
        return self._factor_gradient(cotangent)
