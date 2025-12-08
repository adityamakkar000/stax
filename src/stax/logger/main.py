import jax
import wandb
from typing import Any, Mapping, Optional
from jaxtyping import Array
import os

from loguru import logger
import abc
import time


class BaseLogger(abc.ABC):
    def __init__(self, metrics_to_print: list[str] = ["loss"]):
        self.prev_metric = None
        self.metric_to_print = metrics_to_print
        self.start = None 

    @abc.abstractmethod
    def async_log(self, step: int, data):
        raise NotImplementedError("base class ")

    @abc.abstractmethod
    def finish(self):
        raise NotImplementedError("base class ")

    def __call__(self, step: int, data: dict[str, any]):
        self.flush()
        self.prev_metric = {"step": step, "data": data}
        self.start = time.perf_counter()

    def log(self, step: int, data: dict[str, any]):
        str_to_print = ", ".join(
            [f"{k}: {v:.4f}" for k, v in data.items() if k in self.metric_to_print]
        )
        logger.info(f"Step {step}: " + str_to_print)

    def flush(self):
        if self.prev_metric is None:
            return
        time_elapsed = time.perf_counter() - self.start
        self.prev_metricsic["data"]["time_per_step"] = time_elapsed
        self.async_log(**self.prev_metric)
        self.log(**self.prev_metric)

class WandBLogger(BaseLogger):
    def __init__(
        self,
        entity: str,
        project: str,
        config: Optional[Mapping[str, Any]] = None,
        run_id: Optional[str] = None,
        metrics_to_print: list[str] = ["loss"],
    ):
        super().__init__(metrics_to_print=metrics_to_print)
        assert (config is not None) or (run_id is not None), (
            "Either config or run_id must be provided"
        )
        init_args = {"entity": entity, "project": project, "resume": "allow"}
        if run_id is not None:
            init_args["id"] = run_id
            init_args["resume"] = "must"
        else:
            init_args["config"] = config

        self._run = wandb.init(
            **init_args,
        )

        logger.info(f"Initialized WandB Logger with run id {self.id}")

    def async_log(self, step: int, data: dict[str, Any]):
        data = jax.tree.map(lambda x: x.item() if isinstance(x, Array) else x, data)
        self._run.log(data, step=step)

    def finish(self) -> None:
        self._run.finish()

    @property
    def id(self) -> Optional[str]:
        return self._run.id
