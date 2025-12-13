import jax
import orbax.checkpoint as ocp
from loguru import logger
from jaxtyping import PyTree
from typing import Any, Optional

def to_abstract(x: any) -> jax.ShapeDtypeStruct:
    if isinstance(x, jax.ShapeDtypeStruct):
        return x
    return ocp.utils.to_shape_dtype_struct(x)


class Checkpointer:
    """
    A helper class to manage saving and restoring checkpoints in JAX using Orbax.

    This class handles checkpoint creation, restoration, and management for model state
    and metadata, supporting saving to Google Cloud Storage (GCS) paths.

    Attributes:
        checkpoint_dir (str): Directory path where checkpoints are stored.
        options (ocp.CheckpointManagerOptions): Options controlling checkpoint retention.
        checkpoint_manager (ocp.CheckpointManager): Orbax checkpoint manager instance.
        load (int | None): Latest checkpoint step if found, otherwise None.
    """

    def __init__(self, output_dir: str, max_to_keep: int = 1, best_key: str | None = None, best_mode: str = 'min') -> None:
        """
        Initialize the Checkpointer.

        Args:
            output_dir (str): Google Cloud Storage path (must start with 'gs').
            max_to_keep (int, optional): Maximum number of checkpoints to retain. Defaults to 1. Applies to both regular and best checkpoints.
            best_key (str | None, optional): The key in metadata to monitor for best checkpointing. Must return a scale value when indexed into metrics. Defaults to None.
            best_mode (str, optional): 'min' or 'max' to indicate whether lower or higher values of best_key are better. Defaults to 'min'.

        Raises:
            AssertionError: If the provided output_dir is not a valid GCS path.
        """
        if not output_dir.startswith("gs"):
            logger.info(
                "NOT using gs path -- ensure you are not running multicontroller jax"
            )
        
        self.best_key = best_key
        self.checkpoint_dir = output_dir
        self.options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep)
        
        self.checkpoint_manager: ocp.CheckpointManager = ocp.CheckpointManager(
            self.checkpoint_dir, options=self.options
        )

        if self.best_key:
            assert best_mode in ["min", "max"], "best_mode must be 'min' or 'max'."
            self.best_mode = best_mode
            self.best_fn = lambda metrics: metrics[self.best_key]

            self.best_checkpoint_dir: str = f"{output_dir}/best"
            self.best_options: ocp.CheckpointManagerOptions= ocp.CheckpointManagerOptions(
                max_to_keep=max_to_keep, 
                best_fn=self.best_fn, 
                best_mode=self.best_mode
            )
            self.best_checkpoint_manager: ocp.CheckpointManager = ocp.CheckpointManager(
                self.best_checkpoint_dir, options=self.best_options
            )
        else:
            # if no best_key, we just point to the same manager/dir
            self.best_checkpoint_manager = self.checkpoint_manager

        if self.found_checkpoint:
            logger.info(f"Found latest checkpoint @ step {self.latest_step}")
        if self.best_key and self.best_found_checkpoint:
            logger.info(f"Found best checkpoint @ step {self.best_step}")

        # best checkpoint logging
        if self.found_best_checkpoint:
            logger.info(f"Found BEST checkpoint @ step {self.best_step}")
        else:
            logger.info("No BEST checkpoint found")

    def save_checkpoint(
        self, step: int, *, save_tree: PyTree, metadata: dict[str, any]
    ) -> None:
        """
        Save a checkpoint containing model state and metadata.

        Args:
            step (int): Training step number.
            save_tree (PyTree): Model state or other data to checkpoint.
            metadata (PyTree): Metadata to be saved (e.g., metrics or config).
        """

        self.checkpoint_manager.save(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(save_tree),
                metadata=ocp.args.JsonSave(metadata),
            ),
        )
    
    # best checkpointer save
    def save_best_checkpoint(
        self, step: int, *, save_tree: PyTree, metadata: dict[str, any], metrics: dict[str, any]
    ) -> None:
        """
        Save a best checkpoint containing model state and metadata. Orbax will only keep this step 
        if the metric is better than the previous 'best' step.

        Args:
            step (int): Training step number.
            save_tree (PyTree): Model state or other data to checkpoint.
            metadata (PyTree): Metadata to be saved (e.g., metrics or config).
            metrics (PyTree): 
        """
        if not self.best_key:
            logger.error("Attempted to save 'best' checkpoint but no best_key was configured.")
            return

        if self.best_key not in metrics:
            logger.error(f"Metric '{self.best_key}' missing. Skipping best checkpoint save.")
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
        """
        Restore a checkpoint from the latest saved step or the best (if use_best = True).

        Args:
            state (PyTree): Model state structure to match the checkpoint data.
                Can be concrete (real data) or abstract (jax.ShapeDtypeStructs).
            use_best: Bool which determines whether to use the best checkpoint or not

        Returns:
            dict[str, PyTree]: A dictionary with keys:
                - "state": Restored model state.
                - "metadata": Restored metadata.

        Raises:
            ValueError: If no latest or best checkpoint is found.
        """
        manager = self.best_checkpoint_manager if use_best else self.checkpoint_manager
        
        step = manager.best_step() if (use_best and self.best_key) else manager.latest_step()
        
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

        return {"state": tree.state, "metadata": tree.metadata}

    # restore best checkpoint (auto step)
    def restore_best_auto(self, *, state: PyTree) -> dict[str, PyTree]:
        """
        Restore the BEST checkpoint from the best checkpoint directory.
        Uses the latest_step of best_checkpoint_manager.
        """
        if self.best_step is None:
            raise ValueError("No best checkpoint found")

        abstract_tree_state: PyTree = jax.tree.map(to_abstract, state)

        tree = self.best_checkpoint_manager.restore(
            self.best_step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(abstract_tree_state),
                metadata=ocp.args.JsonRestore(),
            ),
        )

        tree_state, tree_metadata = tree.state, tree.metadata
        return {"state": tree_state, "metadata": tree_metadata}

    def wait_until_finished(self) -> None:
        """
        Block execution until all pending checkpoint save operations have completed.
        """
        self.checkpoint_manager.wait_until_finished()
        if self.best_key:
            self.best_checkpoint_manager.wait_until_finished()

    # wait for both managers
    def wait_until_finished_all(self) -> None:
        self.checkpoint_manager.wait_until_finished()
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
        if self.best_key:
            return self.best_checkpoint_manager.best_step()
        return self.checkpoint_manager.latest_step()
