import time
from typing import Optional
from loguru import logger

import jax


class Tracker:
    def __init__(self, timer: Optional[bool] = False, trace: Optional[str] = None):
        self.timer = timer
        self.trace = trace
        self.data = {}

    def __enter__(self):
        if self.timer:
            self.start = time.perf_counter()
        if self.trace:
            jax.profiler.start_trace(self.trace)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer:
            self.stop = time.perf_counter()
            self.data["time"] = self.elapsed

        if self.trace:
            jax.profiler.stop_trace()
            logger.info(f"Stopped JAX Profiler trace at {self.trace}")
