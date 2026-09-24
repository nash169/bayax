import jax
import jax.numpy as jnp

from typing import Callable, Optional
from bayax.operators.linear_operator import LinearOperator


def lstsq(
    f: Callable,
    u: jnp.ndarray,
    W: LinearOperator,
    x0: jnp.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10,
    hess: Optional[Callable] = None
):
    def cond(carry):
        _, iteration, step_norm = carry
        return (iteration < max_iter) & (step_norm > tol)

    def step(carry):
        x, iteration, _ = carry
        residual = f(x) - u

        if hess is None:
            jac = jax.jacobian(f)(x)
            hessian = jac.T @ W(jac)
            gradient = jac.T @ W(residual)
            dx = jnp.linalg.solve(hessian, gradient)
        else:
            hessian = hess(x)
            gradient = hessian.sqrtf @ (W.sqrtf().transpose() @ residual)
            dx = hessian.solve(gradient)

        return x - dx, iteration + 1, jnp.linalg.norm(dx)

    x, _, _ = jax.lax.while_loop(
        cond,
        step,
        (x0, jnp.array(0), jnp.array(jnp.inf)),
    )
    return x
