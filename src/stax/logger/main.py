import abc
import itertools as it
from typing import Any, Mapping, Optional

import jax
import wandb
from loguru import logger

from stax.utils import convert_to_scalar


class BaseLogger(abc.ABC):
    def __init__(self, metrics_to_print: list[str] = ["loss"]):
        self.prev_metrics = None
        self.metrics_to_print = metrics_to_print
        if jax.process_index() == 0:
            self.setup_logger()

    def __call__(self, step: int, data: dict[str, Any]):
        if jax.process_index() == 0:
            cur_metrics = {"step": step, "data": data}
            self.prev_metrics, log_metrics = cur_metrics, self.prev_metrics
            if log_metrics is None:
                return

            self.async_log(**log_metrics)  # type: ignore
            self._log(**log_metrics)  # type: ignore

    def _log(self, step: int, data: dict[str, Any]):
        log_str = it.starmap(
            lambda k, v: f"{k}: {convert_to_scalar(v):.4f}",
            filter(lambda kv: kv[0] in self.metrics_to_print, data.items()),
        )
        fmt_str = " | ".join((f"Step : {step}\t\t", *log_str))
        logger.info(fmt_str)

    def finish(self):
        if jax.process_index() == 0:
            self._finish()

    @property
    def id(self) -> Optional[str]:
        return self._id() if jax.process_index() == 0 else None

    @abc.abstractmethod
    def setup_logger(self):
        raise NotImplementedError("base class ")

    @abc.abstractmethod
    def async_log(self, step: int, data):
        raise NotImplementedError("base class ")

    @abc.abstractmethod
    def _finish(self) -> None:
        raise NotImplementedError("base class ")

    @abc.abstractmethod
    def _id(self) -> Optional[str]:
        raise NotImplementedError("base class ")


class TextLogger(BaseLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_logger(self):
        pass

    def async_log(self, step: int, data: dict[str, Any]):
        pass

    def _finish(self) -> None:
        pass

    def _id(self) -> Optional[str]:
        return None


class WandBLogger(BaseLogger):
    def __init__(
        self,
        entity: str,
        project: str,
        config: Optional[Mapping[str, Any]] = None,
        run_id: Optional[str] = None,
        *args,
        **kwargs,
    ):
        assert (config is not None) or (run_id is not None), "Either config or run_id must be provided"
        init_args = {"entity": entity, "project": project, "resume": "allow"}
        if run_id is not None:
            init_args["id"] = run_id
            init_args["resume"] = "must"
        else:
            init_args["config"] = config

        self.init_args = init_args
        super().__init__(*args, **kwargs)

    def setup_logger(self, **kwargs):
        self._run = wandb.init(
            **self.init_args  # type: ignore
        )
        logger.info(f"Initialized WandB Logger with run id {self.id}")

    def async_log(self, step: int, data: dict[str, Any]):
        data = jax.tree.map(convert_to_scalar, data)
        self._run.log(data, step=step)

    def _finish(self) -> None:
        self._run.finish()

    def _id(self) -> Optional[str]:
        return self._run.id
