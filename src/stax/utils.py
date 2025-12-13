import time
from typing import Optional, Callable, Any, Type
from types import TracebackType
from loguru import logger

import jax
import jax.numpy as jnp
from jax import Array

from typing import Optional, Callable, Any, Type
from jaxtyping import PRNGKeyArray
from types import TracebackType


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

def estimate_compile_stats(fn: Callable, *args: Any, **kwargs: Any) -> dict[str, float]:
    """
    Estimate memory and FLOPs statistics for a JAX function.

    Args:
        fn: The JAX function (e.g., jitted) to analyze.
        args: Positional arguments to pass to the function.
        kwargs: Keyword arguments to pass to the function.

    Returns:
        A dictionary containing estimated stats like memory usage (GB) and FLOPs (GB).
    """
    compiled_fn = fn.lower(*args, **kwargs).compile()
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


def convert_to_scalar(x: Any) -> Any:
    """
    Convert a JAX array to a scalar if it is an array, otherwise return as is.

    Args:
        x: The input value, potentially a JAX Array.

    Returns:
        The scalar value if x was an array, otherwise x.
    """
    return x.item() if isinstance(x, Array) else x
