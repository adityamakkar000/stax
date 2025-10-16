import jax
import orbax.checkpoint as ocp
from loguru import logger
from jaxtyping import PyTree

#TODO:
# 1. Find ways to get best metric restored

class Checkpointer:

  def __init__(self, output_dir: str, max_to_keep : int=1):

      assert output_dir.startswith("gs"), f"expected gs path got {output_dir}"

      self.checkpoint_dir = output_dir
      #TODO: find some way to add the best model
      self.options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep)
      self.checkpoint_manager = ocp.CheckpointManager(self.checkpoint_dir, options=self.options)
      self.load = self.checkpoint_manager.latest_step() is not None

      if self.load:
         logger.info(f"Found checkpoint @ step {self.load}")
      else:
         logger.ingo(f"No checkpoint found")

  @property
  def found_checkpoint(self):
     return self.load

  def save_checkpoint(self, step : int , save_tree: PyTree, metadata: PyTree) -> None:
     self.checkpoint_manager.save(
        step,
        args=ocp.args.Composite(
           state=ocp.args.StandardSave(save_tree),
           metadata=ocp.args.JsonRestore(metadata)
        )
     )

  def restore(self, state: PyTree) -> dict[str, PyTree]:
         if self.load is None:
            raise ValueError("No latest checkpoint found")

         is_abstract = jax.tree.reduce(
            lambda acc, current: acc and isinstance(current, jax.ShapeDtypeStruct), state, True
         )
         abstract_tree_state = state
         if not is_abstract:
            abstract_tree_state = jax.tree.map(
               ocp.utils.to_shape_dtype_struct, state
            )

         tree = self.checkpoint_manager.restore(
            self.load,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(abstract_tree_state),
                metadata=ocp.args.JsonRestore(),
            ),
         )

         tree_state, tree_metadata = tree.state, tree.metadata
         return {
            "state": tree_state,
            "metadata": tree_metadata
         }