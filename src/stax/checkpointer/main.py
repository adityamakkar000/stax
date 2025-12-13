import jax
import orbax.checkpoint as ocp
from loguru import logger
from jaxtyping import PyTree


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

    def __init__(self, output_dir: str, max_to_keep: int = 1) -> None:
        """
        Initialize the Checkpointer.

        Args:
            output_dir (str): Google Cloud Storage path (must start with 'gs').
            max_to_keep (int, optional): Maximum number of checkpoints to retain. Defaults to 1.

        Raises:
            AssertionError: If the provided output_dir is not a valid GCS path.
        """
        if not output_dir.startswith("gs"):
            logger.info(
                "NOT using gs path -- ensure you are not running multicontroller jax"
            )

        self.checkpoint_dir: str = output_dir
        self.options: ocp.CheckpointManagerOptions = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep
        )
        self.checkpoint_manager: ocp.CheckpointManager = ocp.CheckpointManager(
            self.checkpoint_dir, options=self.options
        )

        # best checkpoint support
        self.best_checkpoint_dir: str = f"{self.checkpoint_dir.rstrip('/')}/best"
        self.best_options: ocp.CheckpointManagerOptions = ocp.CheckpointManagerOptions(
            max_to_keep=1
        )
        self.best_checkpoint_manager: ocp.CheckpointManager = ocp.CheckpointManager(
            self.best_checkpoint_dir, options=self.best_options
        )

        if self.found_checkpoint:
            logger.info(f"Found checkpoint @ step {self.latest_step}")
        else:
            logger.info(f"No checkpoint found")

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

    # save best checkpoint
    def save_best_checkpoint(
        self, step: int, *, save_tree: PyTree, metadata: dict[str, any]
    ) -> None:
        """
        Save a BEST checkpoint (separate directory) containing model state and metadata.
        This does NOT affect the normal/latest checkpoint retention policy.
        """
        self.best_checkpoint_manager.save(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(save_tree),
                metadata=ocp.args.JsonSave(metadata),
            ),
        )

    def restore(self, *, state: PyTree) -> dict[str, PyTree]:
        """
        Restore a checkpoint from the latest saved step.

        Args:
            state (PyTree): Model state structure to match the checkpoint data.
                Can be concrete (real data) or abstract (jax.ShapeDtypeStructs).

        Returns:
            dict[str, PyTree]: A dictionary with keys:
                - "state": Restored model state.
                - "metadata": Restored metadata.

        Raises:
            ValueError: If no latest checkpoint is found.
        """
        if self.found_checkpoint is None:
            raise ValueError("No latest checkpoint found")

        abstract_tree_state: PyTree = jax.tree.map(to_abstract, state)

        tree = self.checkpoint_manager.restore(
            self.latest_step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(abstract_tree_state),
                metadata=ocp.args.JsonRestore(),
            ),
        )

        tree_state, tree_metadata = tree.state, tree.metadata
        return {"state": tree_state, "metadata": tree_metadata}

    def restore_best(self, *, state: PyTree, best_step: int) -> dict[str, PyTree]:
        """
        Restore a checkpoint from a specified best step.

        Args:
            state (PyTree): Model state structure to match the checkpoint data.
                Can be concrete (real data) or abstract (jax.ShapeDtypeStructs).
            best_step (int): The step number of the best checkpoint to restore.
        Returns:
            dict[str, PyTree]: A dictionary with keys:
                - "state": Restored model state.
                - "metadata": Restored metadata.
        Raises:
            ValueError: If no checkpoint is found at the specified best step.
        """
        abstract_tree_state: PyTree = jax.tree.map(to_abstract, state)

        tree = self.checkpoint_manager.restore(
            best_step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(abstract_tree_state),
                metadata=ocp.args.JsonRestore(),
            ),
        )

        tree_state, tree_metadata = tree.state, tree.metadata
        return {"state": tree_state, "metadata": tree_metadata}

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

    # wait for both managers
    def wait_until_finished_all(self) -> None:
        self.checkpoint_manager.wait_until_finished()
        self.best_checkpoint_manager.wait_until_finished()

    @property
    def found_checkpoint(self) -> int | None:
        return isinstance(self.latest_step, int)

    @property
    def latest_step(self) -> int:
        return self.checkpoint_manager.latest_step()

    # best checkpoint properties
    @property
    def best_step(self) -> int | None:
        return self.best_checkpoint_manager.latest_step()

    @property
    def found_best_checkpoint(self) -> bool:
        return isinstance(self.best_step, int)
