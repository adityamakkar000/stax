import jax
from flax import linen as nn
from jaxtyping import PyTree, Array
from typing import Union, TypeVar
import jax.numpy as jnp
import optax

jax_key = TypeVar(Union[jax.random.key, jax.random.PRNGKey])
sharding = jax.sharding.NamedSharding


def process_aux(out: PyTree, has_aux: bool = True) -> PyTree:
    if has_aux:
        _, metrics = out
    else:
        metrics = {"loss": out}
    return metrics


def step(fn: callable, model: nn.Module):
    def step_fn(*batch: PyTree) -> Array:
        return fn(model, *batch)

    return step_fn


def train_step(
    step_fn: callable,
    tx: optax,
    params: PyTree,
    opt_state: PyTree,
    batch: PyTree,
    grad_steps: int = 1,
    has_aux: bool = True,
) -> tuple[PyTree, PyTree]:
    def grad_fn(grads: PyTree, batch: PyTree) -> tuple[PyTree, PyTree]:
        grad_fn = jax.value_and_grad(
            lambda params, *batch: step_fn(params, *batch), has_aux=has_aux
        )
        out, grads = grad_fn(params, *batch)
        metrics = process_aux(out, has_aux=has_aux)
        return grads, metrics

    grads = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=x.dtype), params)

    grads, metrics = jax.lax.scan(grad_fn, grads, batch, length=grad_steps)
    grads = jax.tree.map(lambda x: x / grad_steps, grads)
    metrics = jax.tree.map(lambda x: x.mean(axis=0), metrics)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return {"metrics": metrics, "opt_state": opt_state, "params": params}


def val_step(
    step_fn: callable,
    params: PyTree,
    batch: PyTree,
    eval_steps: int = 1,
    has_aux: bool = True,
) -> PyTree:
    # carry is just a placeholder for scan
    # it has no use
    def val_fn(_carry: None, batch: PyTree) -> tuple[None, PyTree]:
        out = step_fn(params, *batch)
        metrics = process_aux(out, has_aux=has_aux)
        return _carry, metrics

    _, metrics = jax.lax.scan(
        val_fn,
        None,  # start carry with None
        batch,
        length=eval_steps,
    )
    metrics = jax.tree.map(lambda x: x.mean(axis=0), metrics)
    return metrics


def get_step_fn(
    step_fn: callable,
    model: nn.Module,
    tx: optax,
    grad_steps: int = 1,
    has_aux: bool = True,
    in_shardings: sharding | None = None,
    out_shardings: sharding | None = None,
) -> tuple[callable, callable]:
    # TODO: make use of shardings

    single_step = step(step_fn, model)

    @jax.jit
    def train_fn(params, opt_state, batch):
        return train_step(
            single_step,
            tx,
            params,
            opt_state,
            batch,
            grad_steps=grad_steps,
            has_aux=has_aux,
        )

    @jax.jit
    def val_fn(params, batch):
        return val_step(single_step, params, batch)

    return train_fn, val_fn
