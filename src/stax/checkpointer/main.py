from typing import Any, Optional

import jax
import orbax.checkpoint as ocp
from etils import epath
from jaxtyping import PyTree
from orbax.checkpoint._src.multihost import multihost as ocp_multihost

from stax.logger import staxLogger as logger
from stax.multihost_utils import sync_over_mesh
from stax.utils import get_rank


def to_abstract(x: Any) -> jax.ShapeDtypeStruct | int | float:
    if isinstance(x, jax.ShapeDtypeStruct):
        return x
    return ocp.utils.to_shape_dtype_struct(x)

def init_dist_ids():
    ocp_multihost.use_experimental_distributed_process_id = lambda: True 
    ocp_multihost.initialize_runtime_to_distributed_ids()

class Checkpointer:
    """A helper class to manage saving and restoring checkpoints in JAX using Orbax.

    This class handles checkpoint creation, restoration, and management for model state
    and metadata, supporting saving to Google Cloud Storage (GCS) paths.

    Attributes:
        checkpoint_dir (str): Directory path where checkpoints are stored.
        options (ocp.CheckpointManagerOptions): Options controlling checkpoint retention.
        checkpoint_manager (ocp.CheckpointManager): Orbax checkpoint manager instance.

        best_checkpoint_dir (str | None): Directory path for best checkpoints, if applicable.
        best_options (ocp.CheckpointManagerOptions | None): Options for best checkpoint retention.
        best_checkpoint_manager (ocp.CheckpointManager | None): Orbax manager for best checkpoints

    """

    def __init__(
        self,
        output_dir: str,
        max_to_keep: int = 1,
        best_key: str | None = None,
        best_mode: str = "min",
        *, 
        active_processes: Optional[set[int]] = None,
        train_mesh: Optional[jax.sharding.Mesh] = None
    ) -> None:
        """Initialize the Checkpointer.

        Args:
            output_dir (str): Google Cloud Storage path (must start with 'gs').
            max_to_keep (int, optional): Maximum number of regular checkpoints to retain. Defaults to 1. Applies only to regular checkpoints
            best_key (str | None, optional): The key for checkpointing. Must return a scalar value when indexed into metrics (which will be in metadata). Defaults to None.
                if best_key is provided, ensure that the metrics dict passed in as metadata contains this key.
            best_mode (str, optional): 'min' or 'max' to indicate whether lower or higher values of best_key are better. Defaults to 'min'.
            active_processes (set[int], optional): Set of process indices that are allowed to perform checkpointing. If None, all processes are used. Defaults to None.

        Raises:
            AssertionError: If the provided output_dir is not a valid GCS path.

        """
        if not output_dir.startswith("gs"):
            logger.info("NOT using gs path -- ensure you are not running multicontroller jax")
            raise AssertionError("output_dir must be a valid GCS path starting with gs")

        self.best_key = best_key

        # latest checkpointer
        self.checkpoint_dir = output_dir
        if active_processes is not None:
            assert train_mesh is not None, "train_mesh must be provided when active_processes is specified"

            init_dist_ids()
            active_processes = {ocp_multihost.runtime_to_distributed_process_id(rt) for rt in active_processes}
            logger.info(f"Initialized distributed process ID mapping for active processes: {active_processes}", log_for_all=True)
            directory = epath.Path(output_dir)
            logger.info(f"Active processes for checkpointing: {active_processes}")
            if get_rank() == 0 and not directory.exists():
                logger.info(f"Creating checkpoint directory at {directory}")
                directory.mkdir(parents=True, exist_ok=True)
            sync_over_mesh("checkpoint_dir_sync", train_mesh)

        mp_options = ocp.options.MultiprocessingOptions(primary_host=0, active_processes=active_processes)
        self.options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            multiprocessing_options=mp_options,
            create=(active_processes is None), 
            save_root_metadata=False
        )
        self.checkpoint_manager = ocp.CheckpointManager(self.checkpoint_dir, options=self.options)

        # best checkpointer
        if self.best_key:
            assert best_mode in ["min", "max"], "best_mode must be 'min' or 'max'."
            self.best_mode = best_mode
            self.best_fn = lambda metrics: metrics[self.best_key]

            self.best_checkpoint_dir: str | None = f"{output_dir}/best"
            self.best_options = ocp.CheckpointManagerOptions(
                max_to_keep=1, best_fn=self.best_fn, best_mode=self.best_mode, multiprocessing_options=mp_options, create=(active_processes is None)
            )
            self.best_checkpoint_manager = ocp.CheckpointManager(self.best_checkpoint_dir, options=self.best_options)
        else:
            # if no best_key, we just point to the same manager/dir
            self.best_checkpoint_dir = None
            self.best_options = None
            self.best_checkpoint_manager = None

        if self.found_checkpoint:
            logger.info(f"Found latest checkpoint @ step {self.latest_step}")
        if self.best_key and self.best_found_checkpoint:
            logger.info(f"Found best checkpoint @ step {self.best_step}")

    def save_checkpoint(self, step: int, *, save_tree: PyTree, metadata: dict[str, Any]) -> None:
        """Save a checkpoint containing model state and metadata.
        If self.best_key was provided, this automatically saves the best checkpoint as well.

        Args:
            step (int): Training step number.
            save_tree (PyTree): Model state or other data to checkpoint.
            metadata (PyTree): Metadata to be saved. NOTE: "metrics" field is REQUIRED in metadata if utilizing best checkpointing.
                metadata["metrics"] (PyTree): Scalar evaluation metrics used to determine if this is the best checkpoint.

        """
        self.checkpoint_manager.save(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(save_tree),
                metadata=ocp.args.JsonSave(metadata),
            ),
        )

        if self.best_key and self.best_checkpoint_manager is not None:
            if "metrics" not in metadata:
                logger.warning(
                    "Metadata missing 'metrics' field required for best checkpointing. Skipping best checkpoint save."
                )
                return

            metrics = metadata["metrics"]
            if self.best_key not in metrics:
                logger.warning(
                    f"Metric '{self.best_key}' missing in metadata['metrics']. Skipping best checkpoint save."
                )
                return

            self.best_checkpoint_manager.save(
                step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardSave(save_tree),
                    metadata=ocp.args.JsonSave(metadata),
                ),
                metrics=metrics,
            )

    def restore(self, *, state: PyTree, use_best: bool = False) -> dict[str, PyTree]:
        """Restore a checkpoint from the latest saved step or the best (if use_best = True).

        Args:
            state (PyTree): Model state structure to match the checkpoint data.
                Can be concrete (real data) or abstract (jax.ShapeDtypeStructs).
            use_best (bool): Whether to use the best checkpoint or not.

        Returns:
            dict[str, PyTree]: A dictionary with keys:
                - "state": Restored model state.
                - "metadata": Restored metadata.

        Raises:
            ValueError: If no latest or best checkpoint is found.

        """
        if use_best and (self.best_key is None or self.best_checkpoint_manager is None):
            raise ValueError("Cannot use best checkpointing when no best_key was provided during initialization.")

        manager = (
            self.best_checkpoint_manager
            if (use_best and self.best_checkpoint_manager is not None)
            else self.checkpoint_manager
        )

        step = manager.best_step() if use_best else manager.latest_step()
        logger.info(f"Restoring {'best' if use_best else 'latest'} checkpoint @ step {step}")

        if step is None:
            raise ValueError(f"No checkpoint found in {'best' if use_best else 'latest'} directory.")

        abstract_tree_state: PyTree = jax.tree.map(to_abstract, state)

        tree = manager.restore(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(abstract_tree_state),
                metadata=ocp.args.JsonRestore(),
            ),
        )
        assert hasattr(tree, "state") and hasattr(tree, "metadata"), "Restored tree missing required fields."
        logger.info("Checkpoint restoration complete.")

        return {"state": tree.state, "metadata": tree.metadata}

    def wait_until_finished(self) -> None:
        """Block execution until all pending checkpoint save operations have completed."""
        self.checkpoint_manager.wait_until_finished()
        if self.best_key and self.best_checkpoint_manager is not None:
            self.best_checkpoint_manager.wait_until_finished()

    @property
    def found_checkpoint(self) -> bool:
        return self.latest_step is not None

    @property
    def latest_step(self) -> Optional[int]:
        return self.checkpoint_manager.latest_step()

    @property
    def best_found_checkpoint(self) -> bool:
        return self.best_step is not None

    @property
    def best_step(self) -> Optional[int]:
        if self.best_key and self.best_checkpoint_manager is not None:
            return self.best_checkpoint_manager.best_step()
        return None
