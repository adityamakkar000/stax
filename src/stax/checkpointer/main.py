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
        self.load: int | None = self.checkpoint_manager.latest_step()

        if self.load is not None:
            logger.info(f"Found checkpoint @ step {self.load}")
        else:
            logger.info(f"No checkpoint found")

    def make_save_tree(
        self,
        *,
        model_state: dict[PyTree],
        key: PyTree,
        dataset: dict[PyTree],
        **metadata,
    ) -> PyTree:
        save_tree = {"model_state": model_state, "key": key, "dataset": dataset}
        return save_tree, metadata

    def save_checkpoint(self, step: int, **ckpt_info) -> None:
        """
        Save a checkpoint containing model state and metadata.

        Args:
            step (int): Training step number.
            save_tree (PyTree): Model state or other data to checkpoint.
            metadata (PyTree): Metadata to be saved (e.g., metrics or config).
        """
        save_tree, metadata = self.make_save_tree(**ckpt_info)
        self.checkpoint_manager.save(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(save_tree),
                metadata=ocp.args.JsonRestore(metadata),
            ),
        )

    def restore(self, **ckpt_info) -> dict[str, PyTree]:
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
        if self.load is None:
            raise ValueError("No latest checkpoint found")

        save_tree, _ = self.make_save_tree(**ckpt_info)
        abstract_tree_state: PyTree = jax.tree.map(
            to_abstract, save_tree["model_state"]
        )

        tree = self.checkpoint_manager.restore(
            self.load,
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

    @property
    def found_checkpoint(self) -> int | None:
        """
        Returns the most recent checkpoint step if available.

        Returns:
            int | None: The latest checkpoint step, or None if no checkpoint exists.
        """
        return self.load

    @property
    def latest_step(self) -> int:
        latest_step = self.checkpoint_manager.latest_step()
        if latest_step is None:
            raise ValueError("no latest step found")
        return latest_step
