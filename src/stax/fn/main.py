from functools import partial, wraps
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import NamedSharding
from jaxtyping import Array, PyTree

from stax.logger import staxLogger as logger
from stax.model_module import modelBase
from stax.sharding import ShardingConfig, ShardingType, get_sharding, setup_mesh

Params = PyTree
Batch = PyTree
OptState = PyTree
Metrics = Dict[str, Array]

# StepFn: (model, params, *batch, train=True/False) -> Union[loss, (loss, aux)]
StepFn = Callable[[modelBase, PyTree, tuple[PyTree, ...], bool], Union[Array, Tuple[Array, PyTree]]]
# SingleStepFn: (params, *batch, train=True/False) -> Union[loss, (loss, aux)]
SingleStepFn = Callable[[PyTree, tuple[PyTree, ...], bool], Union[Array, Tuple[Array, PyTree]]]


def process_aux(out: Union[Array, Tuple[Array, PyTree]], has_aux: bool = True) -> Metrics:
    """
    Processes the output of a step function to extract metrics.

    Args:
        out: The output from the step function. Can be just loss or (loss, aux).
        has_aux: Whether the output contains auxiliary data (metrics).

    Returns:
        A dictionary of metrics. If no aux data, returns {'loss': out}.
    """
    if has_aux:
        _, metrics = out
    else:
        metrics = {"loss": out}
    return metrics  # type: ignore


def train_step(
    step_fn: SingleStepFn,
    tx: optax.GradientTransformation,
    params: Params,
    opt_state: OptState,
    batch: Batch,
    grad_steps: int = 1,
    has_aux: bool = True,
    offload_opt_state: Optional[PyTree[NamedSharding]] = None,
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
        offload_opt_state: Optional sharding for offloading optimizer state.

    Returns:
        A dictionary containing updated 'metrics', 'params', and 'opt_state'.
    """

    def grad_fn(grads: Params, batch: Batch) -> Tuple[Params, Metrics]:
        def loss_fn(params: Params, batch: Batch) -> Union[Array, Tuple[Array, PyTree]]:
            with jax.named_scope("fwd_pass"):
                return step_fn(params, *batch, train=True)  # type: ignore

        grad_fn_inner = jax.value_and_grad(loss_fn, has_aux=has_aux)
        out, new_grads = grad_fn_inner(params, batch)
        metrics = process_aux(out, has_aux=has_aux)

        grads = jax.tree.map(lambda g, ng: g + ng, grads, new_grads)
        return grads, metrics

    grads = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=x.dtype), params)

    grads, metrics = jax.lax.scan(grad_fn, grads, batch, length=grad_steps)

    grads = jax.tree.map(lambda x: x / grad_steps, grads)
    metrics = jax.tree.map(lambda x: x.mean(axis=0), metrics)

    if offload_opt_state is not None:
        opt_state = jax.tree.map(
            lambda x, sharding: jax.device_put(x, sharding),
            opt_state,
            offload_opt_state,
        )

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
        out = step_fn(params, *batch, train=False)  # type: ignore
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
    model: modelBase,
    tx: optax.GradientTransformation,
    has_aux: bool = True,
    grad_steps: int = 1,
    eval_steps: int = 1,
    sharding: ShardingConfig = ShardingConfig(),
    devices: Optional[np.ndarray] = None,
    **jit_kwargs,
):
    """
    Creates JIT-compiled training and validation functions, optionally with sharding.

    Args:
        step_fn: The step function taking (model, params, *batch, train).
        model: The modelBase instance.
        tx: The Optax optimizer.
        has_aux: Whether the step function returns auxiliary metrics.
        grad_steps: Number of gradient accumulation steps.
        eval_steps: Number of evaluation steps per batch.
        sharding: The type of sharding strategy to use (e.g., from SHARDING_TYPES).
        devices: Array of devices to use for sharding mesh.
        **jit_kwargs: Additional keyword arguments to pass to jax.jit.

    Returns:
        A tuple containing:
        - train_fn: JIT-compiled training function.
        - val_fn: JIT-compiled validation function.
        - shardings: Sharding specifications for params, optimizer state and metrics.
    """
    single_step = partial(step_fn, model)

    if sharding.sharding_type == ShardingType.SINGLE:
        if devices is None:
            devices = np.array([jax.devices()[0]])
        elif devices.size > 1 or devices.ndim > 1:
            raise ValueError(f"expected single device got {devices=}")

    mesh = setup_mesh(devices=devices)
    shard_data, shardings = get_sharding(mesh, sharding)
    out_shardings = {
        "metrics": shardings.metrics_sharding,
        "params": shardings.param_sharding,
        "opt_state": shardings.opt_state_sharding,
    }

    offload_opt_state_sharding = None
    if sharding.opt_state_offload:
        offload_opt_state_sharding = jax.tree.map(lambda x: x.with_memory_kind("device"), shardings.opt_state_sharding)

    def train_fn(params: Params, opt_state: OptState, *batch: Batch) -> Dict[str, Any]:
        """
        Performs a training step, including gradient calculation and parameter updates.

        Args:
            params: Current model parameters.
            opt_state: Current optimizer state.
            *batch: The input batch

        Returns:
            A dictionary containing updated 'metrics', 'params', and 'opt_state'.
        """
        logger.info("compiling train step fn ...")
        with jax.named_scope("train_step"):
            return train_step(
                single_step,
                tx,
                params,
                opt_state,
                shard_data(batch),
                grad_steps=grad_steps,
                has_aux=has_aux,
                offload_opt_state=offload_opt_state_sharding,
            )

    def val_fn(params: Params, *batch: Batch) -> Metrics:
        """
        Performs a validation step over multiple micro-batches.

        Args:
            params: Current model parameters.
            *batch: The input batch

        Returns:
            A dictionary of averaged metrics with 'val_' prefix.
        """
        logger.info("compiling val fn ...")
        with jax.named_scope("val_step"):
            val_metrics = val_step(
                single_step,
                params,
                shard_data(batch),
                eval_steps=eval_steps,
                has_aux=has_aux,
            )
            val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
            return val_metrics

    def compile(fn: Callable, out_shardings: PyTree, **jit_kwargs) -> Callable:
        return wraps(fn)(jax.jit(fn, out_shardings=out_shardings, **jit_kwargs))

    train_fn_jit = wraps(train_fn)(jax.jit(train_fn, out_shardings=out_shardings, **jit_kwargs))
    val_fn_jit = wraps(val_fn)(jax.jit(val_fn, out_shardings=shardings.metrics_sharding))

    return train_fn_jit, val_fn_jit, shardings
