import jax
import jax.numpy as jnp
from jax import Array

import time
from loguru import logger

from typing import Optional, Callable, Any, Type
from jaxtyping import PRNGKeyArray
from types import TracebackType

from jax.sharding import NamedSharding


def move_sharding(sharding: NamedSharding, memory_kind: str) -> NamedSharding:
    """
    Move the sharding to a different memory kind.

    Args:
        sharding: The original NamedSharding object.
        memory_kind: The target memory kind, either 'pinned_host' or 'device'.

    Returns:
        A new NamedSharding object with the specified memory kind.
    """
    memory_kind_arr = ["pinned_host", "device"]
    if memory_kind not in memory_kind_arr:
        raise ValueError(f"memory kind not in {memory_kind_arr} got {memory_kind}")

    return NamedSharding(sharding.mesh, sharding.spec, memory_kind=memory_kind)


class Tracker:
    """
    Context manager for tracking execution time and JAX profiler traces.
    """

    def __init__(self, timer: Optional[bool] = False, trace: Optional[str] = None):
        """
        Initialize the Tracker.
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
        """
        Start tracking time and/or JAX trace.
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
        """
        Stop tracking time and/or JAX trace.
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

    @property
    def is_profiling(self) -> bool:
        """
        Check if profiling is active.
        Returns:
            True if profiling is active, False otherwise.
        """
        return self.trace


def convert_to_scalar(x: Any) -> Any:
    """
    Convert a JAX array to a scalar if it is an array, otherwise return as is.
    Args:
        x: The input value, potentially a JAX Array.
    Returns:
        The scalar value if x was an array, otherwise x.
    """
    return x.item() if isinstance(x, Array) else x


def estimate_compile_stats(compiled_fn: Callable) -> dict[str, float]:
    """
    Estimate memory and FLOPs statistics for a JAX function.

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
        stats["argument_size_gb"] = memory_compiled_stats.argument_size_in_bytes / (
            1024**3
        )
        stats["total_size_gb"] = total / (1024**3)

    if cost_compiled_stats is not None:
        total_flops_gb = cost_compiled_stats["flops"] / (1024**3)
        stats["total_flops_gb"] = total_flops_gb

    return stats


def is_key(x: jax.Array) -> bool:
    """
    Check whether x is a PRNG key based on its shape.

    Args:
        x: The JAX array to check.

    Returns:
        True if x is a key (ndim == 1 and shape[0] == 2 for old keys, or specific dtype for new keys),
        though this implementation specifically checks for ndim=2 and shape[1]=2.
    """
    return x.ndim == 2 and x.shape[1] == 2


def reshape_batch_key(key: Array, n_keys: int, dim: int = 1) -> Array:
    # no carry needed
    def split_single_key(_carry, key: PRNGKeyArray) -> Array:
        return _carry, reshape_key_into_array(key, n_keys)

    assert key.ndim >= 2, "Key must have at least a batch dim"
    _, key = jax.lax.scan(
        split_single_key,
        None,
        key,
    )
    key = jnp.moveaxis(key, 1, dim)
    return key


def reshape_key_into_array(key: PRNGKeyArray, num_keys: int) -> Array:
    """
    Splits a JAX PRNGKey into multiple keys and ensures it has a 2D shape.

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


def get_perf_func(trace_path, func: Callable[..., Any], *args, **kwargs) -> None:
    """
    Profiles the performance of a JAX function, logging memory usage and execution time.
    Args:
        trace_path: Path to save JAX profiler trace.
        func: The JAX function to profile.
        args: Positional arguments to pass to the function.
        kwargs: Keyword arguments to pass to the function.
    """

    compiled_fn = func.lower(*args, **kwargs).compile(
        {"xla_enable_transpose_trace": True}
    )
    stats = estimate_compile_stats(compiled_fn)
    with Tracker(timer=True, trace=trace_path) as t:
        out = func(*args, **kwargs)
        jax.tree.map(lambda x: x.block_until_ready(), out)
    report = (
        "\n"
        "================= Profile Report =================\n"
        "Memory Usage:\n"
        f"\tTemp:      {stats.get('temp_size_gb', 0):>10.4f} GB\n"
        f"\tArgument:  {stats.get('argument_size_gb', 0):>10.4f} GB\n"
        f"\tTotal:     {stats.get('total_size_gb', 0):>10.4f} GB\n"
        "\n"
        "Compute:\n"
        f"\tFLOPs:     {stats.get('total_flops_gb', 0):>10.4f} GB\n"
        "\n"
        "Performance:\n"
        f"\tTime:      {t.data.get('time', 0):>10.4f} s\n"
        f"\tFLOPs/s:   {stats.get('total_flops_gb', 0) / t.data.get('time', 1):>10.4f} GFLOPs/s\n"
        "=================================================="
    )
    logger.info(report)
