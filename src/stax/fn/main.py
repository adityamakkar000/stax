import jax
from flax import linen as nn
from jaxtyping import PyTree, Array
from typing import Union, Callable, Tuple
import jax.numpy as jnp
import optax

import numpy as np
from stax.sharding import (
    setup_dp, 
    get_dp_sharding, 
    SHARDING_TYPES
)

from jax.sharding import (
    NamedSharding, 
    PartitionSpec as P, 
)


#TODO: fix all type infromation

jax_key = Union[jax.random.key, jax.random.PRNGKey]
sharding = jax.sharding.NamedSharding
Params = PyTree
Batch = PyTree
OptState = PyTree

StepFn = Callable[[Params, Batch], float | tuple[float, PyTree]]
TrainFn = Callable[[Params, OptState, Batch], tuple[Params, OptState, float]]
ValFn = Callable[[Params, Batch], float]


def reshape_key_into_array(key: jax.random.PRNGKey , num_keys) -> Array: 
    
    keys = jnp.array(jax.random.split(key, num_keys))
    if keys.ndim == 1: 
        keys = keys.reshape(1, -1)
    return keys

def process_aux(out: PyTree, has_aux: bool = True) -> PyTree:
    if has_aux:
        _, metrics = out
    else:
        metrics = {"loss": out}
    return metrics


def get_single_step_fn(fn: StepFn, model: nn.Module):
    def step_fn(params, *batch: Batch) -> float:
        return fn(model, params, *batch)

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
    return {"metrics": metrics, "params": params, "opt_state": opt_state}


def val_step(
    step_fn: callable,
    params: PyTree,
    batch: PyTree,
    eval_steps: int = 1,
    has_aux: bool = True,
) -> PyTree:
    # carry is a placeholder for scan
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


def get_steps_fn(
    step_fn: callable,
    model: nn.Module,
    tx: optax,
    has_aux: bool = True,
    grad_steps: int = 1, 
    eval_steps: int = 1, 
    sharding : str  | None = None,
    devices : np.ndarray | None = None, 
    data_shard_axis: int = 0
) -> tuple[callable, callable]:
    # TODO: make use of shardings

    single_step = get_single_step_fn(step_fn, model)

    def train_fn_jit(params, opt_state, *batch):
        return train_step(
            single_step,
            tx,
            params,
            opt_state,
            batch,
            grad_steps=grad_steps,
            has_aux=has_aux,
        )

    def val_fn_jit(params, *batch):
        return val_step(single_step, params, batch, eval_steps=eval_steps, has_aux=has_aux)

    if sharding is not None: 
        assert sharding in list(SHARDING_TYPES.keys()), f"got {sharding=} but expected it to be in {list(SHARDING_TYPES.keys())}"
        mesh = setup_dp(devices=devices) 
        shard_data, (param_sharding, opt_state_sharding) = SHARDING_TYPES[sharding](mesh, data_axis=data_shard_axis)
        replicate_sharding = NamedSharding(mesh, P()) 

        train_fn = lambda params, opt_state, *batch: jax.jit(
            train_fn_jit, 
            out_shardings={
                "metrics":replicate_sharding,  
                "params": param_sharding, 
                "opt_state_sharding": opt_state_sharding
            }
        )(params, opt_state, *shard_data(batch))

        val_fn = lambda params, *batch: jax.jit(
            val_fn_jit, 
            out_shardings=replicate_sharding
        )(params, *shard_data(batch))

    else: 
        train_fn = jax.jit(train_fn_jit)
        val_fn = jax.jit(val_fn_jit)

    return train_fn, val_fn


