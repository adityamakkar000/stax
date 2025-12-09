import jax
import wandb
from typing import Any, Mapping, Optional
from jaxtyping import Array
import os
import itertools as it

from loguru import logger
import abc


def convert_to_scalar(x):
    return x.item() if isinstance(x, Array) else x


class BaseLogger(abc.ABC):
    def __init__(self, metrics_to_print: list[str] = ["loss"]):
        self.prev_metric = None
        self.metric_to_print = metrics_to_print

    def __call__(self, step: int, data: dict[str, any]):
        cur_metrics = {"step": step, "data": data}
        self.prev_metrics, log_metrics = cur_metrics, self.prev_metric
        if log_metrics is None:
            return
        self.async_log(**log_metrics)
        self._log(**log_metrics)

    def _log(self, step: int, metrics: dict[str, any]):
        log_str = it.starmap(
            lambda k, v: f"{k}: {convert_to_scalar(v):.4f}",
            filter(lambda kv: kv[0] in self.metric_to_print, metrics.items()),
        )
        fmt_str = " | ".join((f"Step : {step}\t\t", *log_str))
        logger.info(fmt_str)

    @abc.abstractmethod
    def async_log(self, step: int, data):
        raise NotImplementedError("base class ")

    @abc.abstractmethod
    def finish(self):
        raise NotImplementedError("base class ")

    @property
    @abc.abstractmethod
    def id(self) -> Optional[str]:
        raise NotImplementedError("base class ")


class TextLogger(BaseLogger):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

    def async_log(self, step: int, data: dict[str, Any]):
        pass

    def finish(self) -> None:
        pass

    @property
    def id(self) -> Optional[str]:
        return None


class WandBLogger(BaseLogger):
    def __init__(
        self,
        entity: str,
        project: str,
        config: Optional[Mapping[str, Any]] = None,
        run_id: Optional[str] = None,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
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
        data = jax.tree.map(convert_to_scalar, data)
        self._run.log(data, step=step)

    def finish(self) -> None:
        self._run.finish()

    @property
    def id(self) -> Optional[str]:
        return self._run.id
