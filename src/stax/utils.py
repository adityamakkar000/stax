import os
import time
from types import TracebackType
from typing import Any, Optional, Type

import jax
import jax.experimental.multihost_utils as multihost_utils
import jax.numpy as jnp
from jax import Array
from jax._src.pjit import JitWrapped
from jax.stages import Compiled
from jaxtyping import PRNGKeyArray

from stax.logger import staxLogger as logger


class Tracker:
    """Context manager for tracking execution time and JAX profiler traces."""

    def __init__(self, timer: Optional[bool] = False, trace: Optional[str] = None):
        """Initialize the Tracker.

        Args:
            timer: Whether to track execution time. Defaults to False.
            trace: Path to save JAX profiler trace. If None, tracing is disabled. Defaults to None.

        """
        self.timer = timer
        self.trace = trace
        self.profiling = False
        self.data: dict[str, float] = {}
        self.start: float = 0.0
        self.stop: float = 0.0

    def __enter__(self) -> "Tracker":
        """Start tracking time and/or JAX trace.

        Returns:
            The tracker instance.

        """
        if self.timer:
            self.start = time.perf_counter()
        if self.trace:
            jax.profiler.start_trace(self.trace)

        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Stop tracking time and/or JAX trace.

        Args:
            exc_type: The exception type.
            exc_val: The exception value.
            exc_tb: The traceback.

        """
        if self.timer:
            self.stop = time.perf_counter()
            self.data["time"] = self.stop - self.start

        if self.trace:
            jax.profiler.stop_trace()
            logger.info(f"Stopped JAX Profiler trace at {self.trace}")


def convert_to_scalar(x: Any) -> Any:
    """Convert a JAX array to a scalar if it is an array, otherwise return as is.

    Args:
        x: The input value, potentially a JAX Array.

    Returns:
        The scalar value if x was an array, otherwise x.

    """
    return x.item() if isinstance(x, Array) else x


def estimate_compile_stats(compiled_fn: Compiled) -> dict[str, float]:
    """Estimate memory and FLOPs statistics for a JAX function.

    Args:
        compiled_fn: The JAX-compiled function to analyze.

    Returns:
        A dictionary containing estimated stats like memory usage (GB) and FLOPs (GB).

    """
    memory_compiled_stats = compiled_fn.memory_analysis()
    cost_compiled_stats = compiled_fn.cost_analysis()
    stats = dict()

    if memory_compiled_stats is not None:
        total = (
            memory_compiled_stats.temp_size_in_bytes
            + memory_compiled_stats.argument_size_in_bytes
            + memory_compiled_stats.output_size_in_bytes
            - memory_compiled_stats.alias_size_in_bytes
        )

        stats["temp_size_gb"] = memory_compiled_stats.temp_size_in_bytes / (1024**3)
        stats["argument_size_gb"] = memory_compiled_stats.argument_size_in_bytes / (1024**3)
        stats["output_size_gb"] = memory_compiled_stats.output_size_in_bytes / (1024**3)
        stats["total_size_gb"] = total / (1024**3)

    if cost_compiled_stats is not None:
        total_flops_gb = cost_compiled_stats["flops"] / (1024**3)
        stats["total_flops_gb"] = total_flops_gb

    return stats


def is_key(x: jax.Array) -> bool:
    """Check whether x is a PRNG key based on its shape.

    Args:
        x: The JAX array to check.

    Returns:
        True if x is a JAX random key else None

    """
    return isinstance(x, jax.Array) and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)


def reshape_batch_key(key: Array, n_keys: int, dim: int = 1) -> Array:
    # no carry needed
    def split_single_key(_carry: None, key: PRNGKeyArray) -> tuple[None, PRNGKeyArray]:
        return None, reshape_key_into_array(key, n_keys)

    assert key.ndim >= 2, "Key must have at least a batch dim"
    _, key = jax.lax.scan(
        split_single_key,
        None,
        key,
    )
    key = jnp.moveaxis(key, 1, dim)
    return key


def reshape_key_into_array(key: PRNGKeyArray, num_keys: int) -> Array:
    """Splits a JAX PRNGKey into multiple keys and ensures it has a 2D shape.

    Args:
        key: The source random key.
        num_keys: The number of keys to split into.

    Returns:
        An array of keys with shape (num_keys, 2).

    """
    if num_keys == 1:
        return key.reshape(1, 2)
    keys = jnp.array(jax.random.split(key, num_keys))
    return keys


def get_perf_func(trace_path, func: JitWrapped, *args, **kwargs) -> None:
    """Profiles the performance of a JAX function, logging memory usage and execution time.

    Args:
        trace_path: Path to save JAX profiler trace.
        func: The JAX function to profile.
        args: Positional arguments to pass to the function.
        kwargs: Keyword arguments to pass to the function.

    """
    compiled_fn = func.lower(*args, **kwargs).compile({"xla_enable_transpose_trace": True})

    with Tracker(timer=True, trace=trace_path) as t:
        out = compiled_fn(*args, **kwargs)
        jax.tree.map(lambda x: x.block_until_ready(), out)
    time_s = t.data.get("time", 0.0)
    stats = estimate_compile_stats(compiled_fn)
    total_flops_tb = stats.get("total_flops_gb", 0.0) / 1024
    tflops_per_s = (total_flops_tb / time_s) if time_s > 0 else 0.0

    report = (
        "\n"
        "================= Profile Report =================\n"
        "Memory Usage:\n"
        f"\tTemp:      {stats.get('temp_size_gb', 0):>12,.4f} GB\n"
        f"\tArgument:  {stats.get('argument_size_gb', 0):>12,.4f} GB\n"
        f"\tOutput:    {stats.get('output_size_gb', 0):>12,.4f} GB\n"
        f"\tTotal:     {stats.get('total_size_gb', 0):>12,.4f} GB\n"
        "\n"
        "Performance:\n"
        f"\tFLOPs:     {total_flops_tb:>12,.4f} TFLOPs\n"
        f"\tTime:      {time_s:>12,.4f} s\n"
        f"\tFLOPs/s:   {tflops_per_s:>12,.4f} TFLOPs/s\n"
        "=================================================="
    )
    logger.info(report)


def get_primary_host() -> int:
    """Returns the primary host index based on RANK environment variable."""
    if os.environ.get("RANK", None) is None:
        raise ValueError("JAX distributed got no RANK env variable")
    return multihost_utils.broadcast_one_to_all(jax.process_index(), is_source=(int(os.environ["RANK"]) == 0)).item()


def init_distributed_jax():
    """Initializes JAX distributed environment."""
    if os.environ.get("RANK", None) is None:
        raise ValueError("JAX distributed got no RANK env variable")
    if jax.distributed.is_initialized():
        logger.warning("JAX distributed is already initialized")
        return

    jax.distributed.initialize()

    logger.info(f"Current process RANK: {jax.process_index()}")
    process_count = jax.process_count()
    local_devices = len(jax.local_devices())
    logger.info(f"JAX distributed initialized with {process_count} processes with {local_devices} per host.")
    all_devices = jax.devices()
    logger.info("Devices:")
    for dev in all_devices:
        logger.info(f"\tDevice ID: {dev.id}, Platform: {dev.platform}, Kind: {dev.device_kind}")
    multihost_utils.sync_global_devices("init_distributed_jax")
    return
