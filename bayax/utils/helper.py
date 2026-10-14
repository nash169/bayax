import math
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import inspect

from collections.abc import Generator, Sequence
from bayax.utils.types import (
    Matrix,
    PyTree,
    List,
    Vector,
    PyTreeDef,
    Optional,
    Int,
    Callable,
)


def cumsum(seq: Generator) -> list[int]:
    """Compute the cumulative sum of a sequence.

    This function takes a sequence of integers and returns a list of cumulative
    sums.

    Args:
        seq: A generator or sequence of integers.

    Returns:
        A list where each element is the cumulative sum up to that point
        in the input sequence.
    """
    total = 0
    return [total := total + ele for ele in seq]


def array_to_pytree(
    arr: Vector | Matrix, tree: tuple[List, PyTreeDef] | PyTree
) -> PyTree:
    if isinstance(tree, tuple):  # and list(map(type, tree)) == [List, PyTreeDef]
        shapes, tree_def = tree
    elif isinstance(tree, PyTree):
        leaves, tree_def = jax.tree.flatten(tree)
        shapes = [leaf.shape for leaf in leaves]
    else:
        msg = "`tree` must be a tuple [List, PyTreeDef] or a PyTree structure."
        raise TypeError(msg)
    arr_split = jnp.split(arr, cumsum(math.prod(sh) for sh in shapes)[:-1], axis=-1)
    return jax.tree.unflatten(
        tree_def,
        [
            a.reshape(-1, *sh) if isinstance(arr, Matrix) else a.reshape(sh)
            for a, sh in zip(arr_split, shapes, strict=True)
        ],
    )


def pytree_to_array(tree: PyTree, axis: Optional[int] = None) -> Vector | Matrix:
    leaves, _ = jax.tree.flatten(tree)
    return (
        jnp.concatenate([leaf.ravel() for leaf in leaves])
        if axis is None
        else jnp.hstack([leaf.reshape(leaf.shape[axis], -1) for leaf in leaves])
    )


def pytree_size(tree: PyTree) -> Int:
    return sum(x.size for x in jax.tree.leaves(tree))


def wrap_pytree_function(f: Callable, tree: tuple[List, PyTreeDef] | PyTree):
    def fn_wrap(*args):
        return pytree_to_array(f(*[array_to_pytree(arg, tree) for arg in args]))

    return fn_wrap


def wrap_array_function(f: Callable, tree: tuple[List, PyTreeDef] | PyTree):
    def fn_wrap(*args):
        return array_to_pytree(f(*[pytree_to_array(arg) for arg in args]), tree)

    return fn_wrap


def make_datasets_classification(dataset: str = "moons", n1: int = 50, n2: int = 50, noise: float = 0.15, seed: int = 0):
    if dataset == "moons":
        rng = np.random.default_rng(seed)
        t1 = rng.uniform(0.0, math.pi, size=n1)
        t2 = rng.uniform(0.0, math.pi, size=n2)
        x1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)
        x2 = np.stack([1.0 - np.cos(t2), 0.5 - np.sin(t2)], axis=1)
        X = np.concatenate([x1, x2], axis=0)
        y = np.concatenate([np.zeros(n1), np.ones(n2)], axis=0)
        X = X + noise * rng.normal(size=X.shape)
        perm = rng.permutation(n1 + n2)
        X = jnp.asarray(X[perm], dtype=jnp.float64)
        y = jnp.asarray(y[perm], dtype=jnp.float64).reshape(-1, 1)
    elif dataset == "clusters":
        X_1 = jax.random.multivariate_normal(jax.random.key(seed), jnp.array([-1, -1]), 0.2 * jnp.eye(2), shape=(n1,))
        y_1 = jnp.zeros(n1)
        X_2 = jax.random.multivariate_normal(jax.random.key(0), jnp.array([1, 1]), 0.2 * jnp.eye(2), shape=(n2,))
        y_2 = jnp.ones(n2)

        X = jnp.concatenate((X_1, X_2), axis=0)
        y = jnp.concatenate((y_1, y_2))
    else:
        raise ValueError("Unknown dataset.")

    return X, y


def make_datasets_regression(
    func: Callable,
    n_train: int = 150,
    n_valid: int = 50,
    n_test: int = 100,
    noise: float = 0.15,
    interval: tuple[float, float] = (-1.0, 1.0),
    train_intervals: Sequence[tuple[float, float]] | None = None,
    seed: int = 0,
):
    """
    Training: equally allocate points across train_intervals, then sample
              uniformly within each interval. None uses the full domain.
    Validation/test: sample uniformly over interval.
    noise: standard deviation of independent Gaussian observation noise.

    Returns:
        X_train, y_train, X_valid, y_valid, X_test, y_test,
        each with shape (n, 1).
    """
    low, high = interval

    if not low < high:
        raise ValueError("interval must satisfy low < high.")
    if min(n_train, n_valid, n_test) < 0:
        raise ValueError("Dataset sizes must be nonnegative.")
    if noise < 0:
        raise ValueError("noise must be nonnegative.")

    train_intervals = (
        [interval] if train_intervals is None else list(train_intervals)
    )

    if not train_intervals:
        raise ValueError("train_intervals must contain at least one interval.")

    for a, b in train_intervals:
        if not low <= a < b <= high:
            raise ValueError(
                f"Training interval {(a, b)} must have a < b "
                f"and lie within {interval}."
            )

    keys = jr.split(jr.key(seed), 6)

    # Allocate exactly n_train points across any number of intervals.
    n_intervals = len(train_intervals)
    base_count, remainder = divmod(n_train, n_intervals)
    train_keys = jr.split(keys[0], n_intervals)

    X_train = jnp.concatenate([
        jr.uniform(
            train_keys[i],
            shape=(base_count + int(i < remainder),),
            minval=a,
            maxval=b,
        )
        for i, (a, b) in enumerate(train_intervals)
    ])

    X_valid = jr.uniform(
        keys[2], shape=(n_valid,), minval=low, maxval=high
    )
    X_test = jr.uniform(
        keys[4], shape=(n_test,), minval=low, maxval=high
    )

    # Vectorize the supplied scalar function over each dataset.
    evaluate = jax.vmap(func)

    y_train = evaluate(X_train) + noise * jr.normal(keys[1], X_train.shape)
    y_valid = evaluate(X_valid) + noise * jr.normal(keys[3], X_valid.shape)
    y_test = evaluate(X_test) + noise * jr.normal(keys[5], X_test.shape)

    return (
        X_train.reshape(-1, 1),
        y_train.reshape(-1, 1),
        X_valid.reshape(-1, 1),
        y_valid.reshape(-1, 1),
        X_test.reshape(-1, 1),
        y_test.reshape(-1, 1),
    )


def random_like(x, seed: int = 0, **kwargs):
    from bayax.operators import PSDOperator
    if isinstance(x, jax.Array):
        return jax.random.uniform(jax.random.key(seed), shape=x.shape)
    elif isinstance(x, PSDOperator):
        return PSDOperator.random(key=jax.random.key(seed), m=x.shape[0])


def gradient_check(f, df, x, v):
    t = jnp.pow(10, jnp.linspace(-8, 0, 51))

    grad = jnp.abs(jax.vmap(f)(x + (t * v[:, None]).T).squeeze() - f(x) - t * df(x, v))

    t = jnp.log(t)
    grad = jnp.log(grad)

    w = [25, 35]
    slope = jnp.mean((grad[w[0] + 1:w[1]] - grad[w[0]:w[1] - 1]) / (t[w[0] + 1:w[1]] - t[w[0]:w[1] - 1]))

    print("First order Taylor expansion slope:", slope, "- It should be approximately equal to 2.0")

    return True if jnp.abs(slope - 2.0) <= 1e-1 else False, grad, t


def hessian_check(f, df, ddf, x, v):
    t = jnp.pow(10, jnp.linspace(-8, 0, 51))

    hess = jnp.abs(jax.vmap(f)(x + (t * v[:, None]).T).squeeze() - f(x) - t * df(x, v) - 0.5 * jnp.pow(t, 2) * (ddf(x, v) @ v))

    t = jnp.log(t)
    hess = jnp.log(hess)

    w = [30, 40]
    slope = jnp.mean((hess[w[0] + 1:w[1]] - hess[w[0]:w[1] - 1]) / (t[w[0] + 1:w[1]] - t[w[0]:w[1] - 1]))

    print("Second order Taylor expansion slope:", slope, "- It should be approximately equal to 3.0")

    return True if jnp.abs(slope - 3.0) <= 1e-1 else False, hess, t

# if __name__ == "__main__":
#     def fun(x):
#         return jnp.sum(x**3)
#
#     def grad_fun(x, v):
#         return (3 * x**2) @ v
#
#     def hess_fun(x, v):
#         return (6 * jnp.diag(x)) @ v
#
#     ok_grad, grads, ts = gradient_check(fun, grad_fun, in_dim=3, out_dim=1)
#     ok_hess, hesss, ts = hessian_check(fun, grad_fun, hess_fun, in_dim=1, out_dim=1)
#
#     fig, ax = plt.subplots()
#     ax.scatter(ts, hesss)
#     ax.scatter(ts[25].item(), hesss[25].item(), c='r')
#     plt.show()
