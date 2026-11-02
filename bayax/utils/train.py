import tqdm
import optax
import jax.random as jr
import jax.numpy as jnp
from flax import nnx
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class OptCfg:
    lr: float = 1e-3
    grad_clip: float = 1.0
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class TrainCfg:
    opt: optax.GradientTransformation
    steps: int = 1_000
    batch_size: Optional[int] = None
    seed: int = 0
    verbose: bool = False
    log: str = "info"
    record: bool = False


class DataLoader:
    def __init__(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.X = jnp.asarray(X)
        self.y = jnp.asarray(y)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.key = jr.key(seed)
        self.N = self.X.shape[0]
        self.indices = jnp.arange(self.N)

    def __iter__(self):
        if self.shuffle:
            self.key, subkey = jr.split(self.key)
            self.indices = jr.permutation(subkey, self.N)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.N:
            raise StopIteration

        start_idx = self.current_idx
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        self.current_idx = end_idx

        return self.X[batch_indices], self.y[batch_indices]


def train(
    model,
    X: jnp.ndarray,
    y: jnp.ndarray,
    loss_fn: Callable,
    cfg: TrainCfg,
):
    # optimizers without extra args (e.g. adam) ignore value, grad and value_fn
    opt = nnx.Optimizer(model, optax.with_extra_args_support(cfg.opt), wrt=nnx.Param)

    @nnx.jit
    def step(model, opt, x, y):
        loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
        # loss as a function of the parameters, for line searches (e.g. lbfgs)
        graphdef, _, rest = nnx.split(model, nnx.Param, ...)
        value_fn = lambda params: loss_fn(nnx.merge(graphdef, params, rest), x, y)
        opt.update(model, grads, value=loss, grad=grads, value_fn=value_fn)
        return loss

    if cfg.record:
        log = []

    iters = tqdm.tqdm(
        range(cfg.steps), desc="Training w/o replacement", disable=not cfg.verbose
    )

    if cfg.batch_size is not None:
        dataloader = DataLoader(X, y, cfg.batch_size, shuffle=True, seed=cfg.seed)
        for _ in iters:
            for x_tr, y_tr in dataloader:
                loss = step(model, opt, x_tr, y_tr)
            if cfg.log == "debug":
                iters.set_postfix({"Loss": loss.item()})
            if cfg.record:
                log.append(model.params)
    else:
        for i in iters:
            loss = step(model, opt, X, y)
            if cfg.log == "debug" and (i % 100) == 0:
                iters.set_postfix({"Loss": loss.item()})
            if cfg.record:
                log.append(model.params)

    return jnp.array(log) if cfg.record else loss


def train_batch_fixed(
    model,
    X: jnp.ndarray,
    y: jnp.ndarray,
    loss_fn: Callable,
    cfg: TrainCfg,
):
    opt = nnx.Optimizer(model, cfg.opt, wrt=nnx.Param)
    dataloader = DataLoader(X, y, cfg.batch_size, shuffle=True, seed=cfg.seed)

    @nnx.jit
    def step(model, opt):
        for x_tr, y_tr in dataloader:
            loss, grads = nnx.value_and_grad(loss_fn)(model, x_tr, y_tr)
            opt.update(model, grads)
        return loss

    if cfg.record:
        log = []

    iters = tqdm.tqdm(
        range(cfg.steps),
        desc="Training w/o replacement [Fixed batch]",
        disable=not cfg.verbose,
    )
    for _ in iters:
        loss = step(model, opt)
        if cfg.log == "debug":
            iters.set_postfix({"Loss": loss.item()})
        if cfg.record:
            log.append(loss)

    return jnp.array(log) if cfg.record else loss


def train_batch_replacement(
    model,
    X: jnp.ndarray,
    y: jnp.ndarray,
    loss_fn: Callable,
    cfg: TrainCfg,
):
    opt = nnx.Optimizer(model, cfg.opt, wrt=nnx.Param)

    @nnx.jit
    def step(model, opt, key):
        idx = jr.randint(key, (cfg.batch_size,), 0, X.shape[0])
        loss, grads = nnx.value_and_grad(loss_fn)(model, X[idx], y[idx])
        opt.update(model, grads)
        return loss

    if cfg.record:
        log = []

    keys = jr.split(jr.key(cfg.seed), cfg.steps)
    iters = tqdm.tqdm(
        range(cfg.steps), desc="Training with replacement]", disable=not cfg.verbose
    )
    for t in iters:
        loss = step(model, opt, keys[t])
        if cfg.log == "debug":
            iters.set_postfix({"Loss": loss.item()})
        if cfg.record:
            log.append(loss)

    return loss


def train_stochastic_loss(
    model,
    X: jnp.ndarray,
    y: jnp.ndarray,
    loss_fn: Callable,
    cfg: TrainCfg,
):
    opt = nnx.Optimizer(model, cfg.opt, wrt=nnx.Param)
    dataloader = DataLoader(X, y, cfg.batch_size, shuffle=True, seed=cfg.seed)

    @nnx.jit
    def step(model, opt, x, y, key):
        (loss, key), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, x, y, key)
        opt.update(model, grads)
        return loss, key

    if cfg.record:
        log = []

    key = jr.key(cfg.seed)
    iters = tqdm.tqdm(
        range(cfg.steps), desc="Training w/o replacement", disable=not cfg.verbose
    )
    for _ in iters:
        for x_tr, y_tr in dataloader:
            loss, key = step(model, opt, x_tr, y_tr, key)
        if cfg.log == "debug":
            iters.set_postfix({"Loss": loss.item()})
        if cfg.record:
            log.append(loss)

    return jnp.array(log) if cfg.record else loss
