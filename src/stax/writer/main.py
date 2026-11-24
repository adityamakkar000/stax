import abc
import itertools as it
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import jax
import wandb
from jaxtyping import PyTree

from stax.logger import staxLogger as logger
from stax.multihost_utils import broadcast_over_mesh, sync_over_mesh
from stax.utils import convert_to_scalar, get_rank


@dataclass
class TableMetrics:
    table_name: str
    table: dict[str, list[str]]


@dataclass
class Metric:
    step: int
    data: dict[str, Any]
    table_metrics: list[TableMetrics]


"""
Writer classes for logging metrics during training and evaluation.
They are designed to work in multi-host environments, ensuring that only the primary host
performs actual logging to avoid duplication.
"""


class BaseMetricWriter(abc.ABC):
    """Base writer that only writes from the primary host in multihost environments."""

    def __init__(
        self, metrics_to_print: list[str] = ["loss"], mesh: jax.sharding.Mesh | None = None, lag_one_step: bool = True
    ):
        """Initialize the metric writer.

        Args:
            metrics_to_print: List of metric names to print to console.
            mesh: The JAX sharding mesh.
            lag_one_step: If True, each call writes the previous step's metrics instead of the
                current ones, so logging doesn't block on the current step's device computation.
                The final metric is flushed in `finish()`. If False, metrics are written immediately.

        Each call also records `step_time`: the wall-clock seconds since the previous call
        (absent on the first call). It is added to the metric data, so it is printed and
        sent to the writer backend alongside the other metrics.
        """
        self.metrics_to_print = metrics_to_print
        self.prev_metric: None | Metric = None
        self.mesh = mesh
        self.lag_one_step = lag_one_step
        self._last_call_time: float | None = None

        self._run = None
        if self.is_primary_host:
            # should set self._run
            self._setup_writer()
            if self._run is None:
                logger.warning("no writer is set")

        logger.info(f"Setup writer with id: {_id if (_id := self.id) is not None else 'n/a'}")
        sync_over_mesh("writer_setup", self.mesh)

    def __call__(
        self,
        step: int,
        data: PyTree,
        table_metrics: list[TableMetrics] = [],
    ):
        """Write metrics. Only primary host performs actual writing.

        Args:
            step: Training step number.
            data: PyTree containing metric values.
            table_metrics: Tables to log alongside this step's metrics.
        """
        if self.is_primary_host:
            now = time.perf_counter()
            if self._last_call_time is not None:
                data = {**data, "step_time": now - self._last_call_time}
            self._last_call_time = now

            cur_metrics = Metric(step, data, table_metrics)
            if self.lag_one_step:
                self.prev_metric, metric_to_write = cur_metrics, self.prev_metric
            else:
                metric_to_write = cur_metrics

            if metric_to_write is not None:
                self._async_write_metrics(metric_to_write)
                self._log(metric_to_write)
        sync_over_mesh("writer_sync", self.mesh)

    def _log(self, metric: Metric):
        """Print formatted metrics to console.

        Args:
            metric: Metric object containing step and data to write to stdout.
        """

        step = metric.step
        data = metric.data

        for table_metric in metric.table_metrics:
            self._log_generations(step, table_metric.table, table_metric.table_name)

        metric_strs = it.starmap(
            lambda k, v: f"{k}: {convert_to_scalar(v):.4f}",
            filter(lambda kv: kv[0] in self.metrics_to_print, data.items()),
        )
        step_time_str = (f"step time: {data['step_time']:.2f}s",) if "step_time" in data else ()
        fmt_str = " | ".join((f"Step: {step}", *metric_strs, *step_time_str))
        logger.info(fmt_str)

    def finish(self):
        """Finish writing. Only primary host performs cleanup."""
        if self.lag_one_step:
            self(step=-1, data={})  # flush last metric
        if self.is_primary_host:
            self._finish()
        sync_over_mesh("writer_finish", self.mesh)

    @property
    def id(self) -> str:
        """Get logger ID. Returns None on non-primary hosts.

        Returns:
            Logger/run ID or None if not on primary host.
        """
        _id = broadcast_over_mesh(self._id(), is_source=self.is_primary_host, mesh=self.mesh).item()
        return str(_id)

    @property
    def is_primary_host(self) -> bool:
        """Check if current process is the primary host (process 0).

        Returns:
            True if this is the primary host, False otherwise.
        """
        return get_rank() == 0

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
    def _log_generations(self, step: int, table: dict[str, list[str]], table_name: str) -> None:
        """Log a generation table. Called only on primary host.

        Args:
            step: Training step number.
            table: Mapping of column name to column values, zipped row-wise.
            table_name: Key the table is logged under.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _id(self) -> int:
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
    def _log_generations(self, step: int, table: dict[str, list[str]], table_name: str) -> None: ...
    def _id(self) -> int:
        return -1


class WandBWriter(BaseMetricWriter):
    """Weights & Biases logger for experiment tracking."""

    def __init__(
        self,
        name: str,
        entity: str,
        project: str,
        config: Optional[Mapping[str, Any]] = None,
        run_id: Optional[str] = None,
        **init_kwargs,
    ):
        """Initialize WandB writer.

        Args:
            name: Name of the WandB run.
            entity: WandB entity name.
            project: WandB project name.
            config: Configuration dictionary for new run.
            run_id: ID of existing run to resume.
        """
        self.entity = entity
        self.project = project
        self.name = name
        self.config = config
        self.run_id = run_id
        super().__init__(**init_kwargs)

    def _setup_writer(self):
        """Initialize WandB run on primary host."""
        if self.config is None and self.run_id is None:
            raise ValueError("Either config or run_id must be provided")

        init_args: dict = {
            "entity": self.entity,
            "project": self.project,
            "resume": "must" if self.run_id is not None else "allow",
        }

        if self.run_id is not None:
            init_args["id"] = self.run_id
            logger.info(f"Resuming WandB run with id: {self.run_id}")
        else:
            init_args["id"] = str(random.randint(1, 1_000_000))
            init_args["config"] = self.config
            init_args["name"] = self.name
            logger.info(f"Starting a new WandB run with id: {init_args['id']}")

        self._run = wandb.init(**init_args)

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

    def _log_generations(self, step: int, table: dict[str, list[str]], table_name: str) -> None:
        """Log a generation table as a wandb Table.

        Args:
            step: Training step number.
            table: Mapping of column name to column values, zipped row-wise.
            table_name: Key the table is logged under.
        """
        if self._run is None:
            raise ValueError("run is None")

        assert len(set(len(v) for v in table.values())) == 1, f"{table_name}: all columns must have equal rows"

        wandb_table = wandb.Table(columns=list(table.keys()))
        for row in zip(*table.values(), strict=True):
            wandb_table.add_data(*row)

        self._run.log({table_name: wandb_table}, step=step)

    def _id(self) -> int:
        """Return WandB run ID.

        Returns:
            WandB run ID or -1 if run not initialized.
        """
        return int(self._run.id) if self._run is not None else -1
