import time
from typing import Optional
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
