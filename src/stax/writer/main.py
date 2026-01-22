import abc
import itertools as it
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import jax
import wandb
from jax.experimental.multihost_utils import sync_global_devices
from jaxtyping import PyTree

from stax.logger import staxLogger as logger
from stax.utils import convert_to_scalar


@dataclass
class Metric:
    step: int
    data: dict[str, Any]


"""
Writer classes for logging metrics during training and evaluation.
They are designed to work in multi-host environments, ensuring that only the primary host
performs actual logging to avoid duplication.
"""


class BaseMetricWriter(abc.ABC):
    """Base writer that only writes from the primary host in multihost environments."""

    def __init__(self, metrics_to_print: list[str] = ["loss"]):
        """Initialize the metric writer.

        Args:
            metrics_to_print: List of metric names to print to console.
        """
        self.metrics_to_print = metrics_to_print
        self.prev_metric: None | Metric = None

        self._run = None
        if self.is_primary_host:
            # should set self._run
            self._setup_writer()
            if self._run is None:
                logger.warning("no writer is set")

        sync_global_devices("writer_setup")

    def __call__(self, step: int, data: PyTree):
        """Write metrics. Only primary host performs actual writing.

        Args:
            step: Training step number.
            data: PyTree containing metric values.
        """
        if self.is_primary_host:
            cur_metrics = Metric(step, data)
            self.prev_metric, metric_to_write = cur_metrics, self.prev_metric

            if metric_to_write is None:
                return

            self._async_write_metrics(metric_to_write)
            self._log(metric_to_write)
        sync_global_devices("writer_sync")

    def _log(self, metric: Metric):
        """Print formatted metrics to console.

        Args:
            metric: Metric object containing step and data to write to stdout.
        """

        step = metric.step
        data = metric.data
        metric_strs = it.starmap(
            lambda k, v: f"{k}: {convert_to_scalar(v):.4f}",
            filter(lambda kv: kv[0] in self.metrics_to_print, data.items()),
        )
        fmt_str = " | ".join((f"Step: {step}", *metric_strs))
        logger.info(fmt_str)

    def finish(self):
        """Finish writing. Only primary host performs cleanup."""
        if self.is_primary_host:
            self(step=-1, data={})  # flush last metric
            self._finish()
        sync_global_devices("writer_finish")

    @property
    def id(self) -> Optional[str]:
        """Get logger ID. Returns None on non-primary hosts.

        Returns:
            Logger/run ID or None if not on primary host.
        """
        _id = None
        if self.is_primary_host:
            _id = self._id()
        sync_global_devices("id")
        return _id

    @property
    def is_primary_host(self) -> bool:
        """Check if current process is the primary host (process 0).

        Returns:
            True if this is the primary host, False otherwise.
        """
        return os.environ.get("RANK", None) == "0"

    @abc.abstractmethod
    def _setup_writer(self):
        """Initialize the writer. Called only on primary host."""
        raise NotImplementedError

    @abc.abstractmethod
    def _async_write_metrics(self, metric: Metric):
        """Asynchronously write metrics. Called only on primary host.

        Args:
            metric: Metric object to write.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _finish(self):
        """Cleanup resources. Called only on primary host."""
        raise NotImplementedError

    @abc.abstractmethod
    def _id(self) -> Optional[str]:
        """Return logger ID. Called only on primary host.

        Returns:
            Logger/run ID or None.
        """
        raise NotImplementedError


class TextWriter(BaseMetricWriter):
    """Simple text-based logger that only prints to console."""

    def __init__(self, *init_args, **init_kwargs):
        super().__init__(*init_args, **init_kwargs)

    def _setup_writer(self): ...
    def _async_write_metrics(self, metric: Metric): ...
    def _finish(self) -> None: ...
    def _id(self) -> Optional[str]:
        return None


class WandBWriter(BaseMetricWriter):
    """Weights & Biases logger for experiment tracking."""

    def __init__(
        self,
        entity: str,
        project: str,
        config: Optional[Mapping[str, Any]] = None,
        run_id: Optional[str] = None,
        **init_kwargs,
    ):
        """Initialize WandB writer.

        Args:
            entity: WandB entity name.
            project: WandB project name.
            config: Configuration dictionary for new run.
            run_id: ID of existing run to resume.
        """
        self.entity = entity
        self.project = project
        self.config = config
        self.run_id = run_id
        super().__init__(**init_kwargs)

    def _setup_writer(self):
        """Initialize WandB run on primary host."""
        if self.config is None and self.run_id is None:
            raise ValueError("Either config or run_id must be provided")

        init_args = {
            "entity": self.entity,
            "project": self.project,
            "resume": "must" if self.run_id is not None else "allow",
        }

        if self.run_id is not None:
            init_args["id"] = self.run_id
        else:
            init_args["config"] = self.config

        self._run = wandb.init(**init_args)  # type: ignore
        logger.info(f"Initialized WandB Logger with run id: {self.id}")

    def _async_write_metrics(self, metric: Metric):
        """Log metrics to WandB.

        Args:
            metric: Metric object containing step and data.
        """
        if self._run is None:
            raise ValueError("run is None")

        step = metric.step
        scalar_data = jax.tree.map(convert_to_scalar, metric.data)
        self._run.log(scalar_data, step=step)

    def _finish(self) -> None:
        """Finish WandB run."""
        if self._run is not None:
            self._run.finish()

    def _id(self) -> Optional[str]:
        """Return WandB run ID.

        Returns:
            WandB run ID or None if run not initialized.
        """
        return self._run.id if self._run is not None else None
