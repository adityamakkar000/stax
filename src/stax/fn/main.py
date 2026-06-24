from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import NamedSharding
from jaxtyping import Array, PyTree

from stax.logger import staxLogger as logger
from stax.model_module import modelBase
from stax.sharding import ShardingConfig, Shardings, ShardingType, get_sharding, setup_mesh

from .utils import Batch, Metrics, OptState, Params, SingleStepFn, StepFn, TrainFn, ValFn, process_aux


def train_step(
    step_fn: SingleStepFn,
    tx: optax.GradientTransformation,
    params: Params,
    opt_state: OptState,
    batch: Batch,
    grad_steps: int = 1,
    reduce_fn: Callable[[int | Array, Batch], int | Array] = lambda s, b: s + 1,
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

    def grad_fn(rolling_grads: tuple[Params, int | Array], batch: Batch) -> Tuple[tuple[Params, int | Array], Metrics]:
        def loss_fn(params: Params, batch: Batch) -> Union[Array, Tuple[Array, PyTree]]:
            with jax.named_scope("fwd_pass"):
                return step_fn(params, *batch, train=True)  # type: ignore

        grad_fn_inner = jax.value_and_grad(loss_fn, has_aux=has_aux)
        out, new_grads = grad_fn_inner(params, batch)
        metrics = process_aux(out, has_aux=has_aux)
        grads, rolling_denom = rolling_grads
        grads = jax.tree.map(lambda g, ng: g + ng, grads, new_grads)
        rolling_denom  = reduce_fn(rolling_denom, batch)
        return (grads, rolling_denom), metrics

    grads = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=x.dtype), params)

    (grads, rolling_denom), metrics = jax.lax.scan(grad_fn, (grads, 0), batch, length=grad_steps)

    rolling_denom = jnp.maximum(rolling_denom, 1)
    grads = jax.tree.map(lambda x: x / rolling_denom, grads)
    metrics = jax.tree.map(lambda x: x.sum(axis=0) / rolling_denom, metrics)

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
    val_steps: int = 1,
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
        length=val_steps,
    )
    metrics = jax.tree.map(lambda x: x.mean(axis=0), metrics)

    return metrics


def get_steps_fn(
    step_fn: StepFn,
    model: modelBase,
    tx: optax.GradientTransformation,
    has_aux: bool = True,
    grad_steps: int = 1,
    reduce_fn: Callable[[float, Batch], float] = lambda s, b: s + 1.0,
    val_steps: int = 1,
    sharding: ShardingConfig = ShardingConfig(),
    devices: Optional[np.ndarray] = None,
    **jit_kwargs,
) -> Tuple[TrainFn, ValFn, Shardings]:
    """
    Creates JIT-compiled training and validation functions, optionally with sharding.

    Args:
        step_fn: The step function taking (model, params, *batch, train).
        model: The modelBase instance.
        tx: The Optax optimizer.
        has_aux: Whether the step function returns auxiliary metrics.
        grad_steps: Number of gradient accumulation steps.
        val_steps: Number of validation steps per batch.
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
    shardings = get_sharding(mesh, sharding)
    train_shardings = {
        "metrics": shardings.metrics_sharding,
        "params": shardings.param_sharding,
        "opt_state": shardings.opt_state_sharding,
    }

    offload_opt_state_sharding = None
    if sharding.opt_state_offload:
        offload_opt_state_sharding = jax.tree.map(lambda x: x.with_memory_kind("device"), shardings.opt_state_sharding)

    @partial(jax.jit, out_shardings=train_shardings, **jit_kwargs)
    def train_fn(params: Params, opt_state: OptState, *batch: Batch) -> Dict[str, Any]:
        logger.info("compiling train step fn ...")
        with jax.named_scope("train_step"):
            out = train_step(
                single_step,
                tx,
                params,
                opt_state,
                batch,
                grad_steps=grad_steps,
                reduce_fn=reduce_fn,
                has_aux=has_aux,
                offload_opt_state=offload_opt_state_sharding,
            )
            out["metrics"] = {f"train/{k}": v for k, v in out["metrics"].items()}
            return out

    @partial(jax.jit, out_shardings=shardings.metrics_sharding, **jit_kwargs)
    def val_fn(params: Params, *batch: Batch) -> Metrics:
        logger.info("compiling val fn ...")
        with jax.named_scope("val_step"):
            val_metrics = val_step(
                single_step,
                params,
                batch,
                val_steps=val_steps,
                has_aux=has_aux,
            )
            val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
            return val_metrics

    return train_fn, val_fn, shardings # type: ignore
