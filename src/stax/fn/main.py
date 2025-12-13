import jax
from flax import linen as nn
from jaxtyping import PyTree, Array
from typing import Union, Callable, Tuple, Any, Dict, Optional
import jax.numpy as jnp
import optax

import numpy as np
from stax.sharding import setup_mesh, get_sharding, ShardingConfig, ShardingType
from stax.utils import reshape_key_into_array

from jax.sharding import (
    NamedSharding,
    PartitionSpec as P,
)
from functools import partial

Params = PyTree
Batch = PyTree
OptState = PyTree
Metrics = Dict[str, Array]

# StepFn: (model, params, *batch, train=True/False) -> Union[loss, (loss, aux)]
StepFn = Callable[..., Union[float, Tuple[float, PyTree]]]
# SingleStepFn: (params, *batch, train=True/False) -> Union[loss, (loss, aux)]
SingleStepFn = Callable[..., Union[float, Tuple[float, PyTree]]]


def process_aux(
    out: Union[Array, Tuple[Array, PyTree]], has_aux: bool = True
) -> Metrics:
    """
    Processes the output of a step function to extract metrics.

    Args:
        out: The output from the step function. Can be just loss or (loss, aux).
        has_aux: Whether the output contains auxiliary data (metrics).

    Returns:
        A dictionary of metrics. If no aux data, returns {'loss': out}.
    """
    if has_aux:
        _, metrics = out  # type: ignore
    else:
        metrics = {"loss": out}
    return metrics  # type: ignore


def get_single_step_fn(fn: StepFn, model: nn.Module) -> SingleStepFn:
    """
    Wraps a generic step function to bind the model instance.

    Args:
        fn: The step function taking (model, params, *batch, train).
        model: The Flax model instance.

    Returns:
        A function taking (params, *batch, train).
    """

    def step_fn(
        params: Params, *batch: Batch, train: bool = True
    ) -> Union[float, Tuple[float, PyTree]]:
        return fn(model, params, *batch, train=train)

    return step_fn


def train_step(
    step_fn: SingleStepFn,
    tx: optax.GradientTransformation,
    params: Params,
    opt_state: OptState,
    batch: Batch,
    grad_steps: int = 1,
    has_aux: bool = True,
) -> Dict[str, Any]:
    """
    Performs a training step, including gradient calculation and parameter updates.
    Supports gradient accumulation via `grad_steps`.

    Args:
        step_fn: The function computing loss/metrics for a single micro-batch.
        tx: The Optax gradient transformation (optimizer).
        params: Current model parameters.
        opt_state: Current optimizer state.
        batch: The input batch (potentially containing multiple micro-batches).
        grad_steps: Number of gradient accumulation steps (micro-batches).
        has_aux: Whether step_fn returns auxiliary metrics.

    Returns:
        A dictionary containing updated 'metrics', 'params', and 'opt_state'.
    """

    def grad_fn(grads: Params, batch: Batch) -> Tuple[Params, Metrics]:
        def loss_fn(params, *batch):
            return step_fn(params, *batch, train=True)

        grad_fn_inner = jax.value_and_grad(loss_fn, has_aux=has_aux)
        out, new_grads = grad_fn_inner(params, *batch)
        metrics = process_aux(out, has_aux=has_aux)

        grads = jax.tree.map(lambda g, ng: g + ng, grads, new_grads)
        return grads, metrics

    grads = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=x.dtype), params)

    grads, metrics = jax.lax.scan(grad_fn, grads, batch, length=grad_steps)

    grads = jax.tree.map(lambda x: x / grad_steps, grads)
    metrics = jax.tree.map(lambda x: x.mean(axis=0), metrics)

    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return {"metrics": metrics, "params": params, "opt_state": opt_state}


def val_step(
    step_fn: SingleStepFn,
    params: Params,
    batch: Batch,
    eval_steps: int = 1,
    has_aux: bool = True,
) -> Metrics:
    """
    Performs a validation step over multiple micro-batches.

    Args:
        step_fn: The function computing loss/metrics for a single micro-batch.
        params: Current model parameters.
        batch: The input batch (potentially containing multiple micro-batches).
        eval_steps: Number of evaluation steps (micro-batches).
        has_aux: Whether step_fn returns auxiliary metrics.

    Returns:
        A dictionary of averaged metrics.
    """

    # carry is a placeholder for scan
    def val_fn(_carry: None, batch: Batch) -> Tuple[None, Metrics]:
        out = step_fn(params, *batch, train=False)
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
    step_fn: StepFn,
    model: nn.Module,
    tx: optax.GradientTransformation,
    has_aux: bool = True,
    grad_steps: int = 1,
    eval_steps: int = 1,
    sharding: ShardingConfig = ShardingConfig(),
    devices: Optional[np.ndarray] = None,
) -> Tuple[Callable, Callable, Tuple[Any, Any]]:
    """
    Creates JIT-compiled training and validation functions, optionally with sharding.

    Args:
        step_fn: The base step function.
        model: The Flax model instance.
        tx: The Optax optimizer.
        has_aux: Whether the step function returns auxiliary metrics.
        grad_steps: Number of gradient accumulation steps.
        eval_steps: Number of evaluation steps per batch.
        sharding: The type of sharding strategy to use (e.g., from SHARDING_TYPES).
        devices: Array of devices to use for sharding mesh.
        data_shard_axis: Axis along which to shard data.

    Returns:
        A tuple containing:
        - train_fn: JIT-compiled training function.
        - val_fn: JIT-compiled validation function.
        - (param_sharding, opt_state_sharding): Sharding specifications for params and optimizer state.
    """
    single_step = get_single_step_fn(step_fn, model)

    if sharding.sharding_type == ShardingType.SINGLE:
        devices = np.array([jax.devices()[0]])
    mesh = setup_mesh(devices=devices)
    shard_data, (param_sharding, opt_state_sharding) = get_sharding(mesh, sharding)
    replicate_sharding = NamedSharding(mesh, P())
    out_shardings = {
        "metrics": replicate_sharding,
        "params": param_sharding,
        "opt_state": opt_state_sharding,
    }

    @partial(jax.jit, out_shardings=out_shardings)
    def train_fn_jit(
        params: Params, opt_state: OptState, *batch: Batch
    ) -> Dict[str, Any]:
        with jax.named_scope("train_step"):
            return train_step(
                single_step,
                tx,
                params,
                opt_state,
                batch,
                grad_steps=grad_steps,
                has_aux=has_aux,
            )

    @partial(jax.jit, out_shardings=replicate_sharding)
    def val_fn_jit(params: Params, *batch: Batch) -> Metrics:
        with jax.named_scope("val_step"):
            val_metrics = val_step(
                single_step,
                params,
                batch,
                eval_steps=eval_steps,
                has_aux=has_aux,
            )
            val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
            return val_metrics

    train_fn_final = lambda p, o, *b: train_fn_jit(p, o, *shard_data(b))
    val_fn_final = lambda p, *b: val_fn_jit(p, *shard_data(b))

    return train_fn_final, val_fn_final, (param_sharding, opt_state_sharding)
