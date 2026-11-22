import abc
from typing import Optional

import jax
from jaxtyping import Array, PyTree
from optax import GradientTransformation

from stax.sharding import Shardings


class modelBase(abc.ABC):
    """
    Main model abstract base class.
    This class provides foundational methods for model initialization and checkpoint loading.
    """

    @abc.abstractmethod
    def init_state(
        self, rng: Array, tx: GradientTransformation | None, *, sharding: Shardings | None = None
    ) -> PyTree:
        """
        Initialize model weights and optimizer state.
        Args:
            rng (Array): JAX random key for initialization.
            tx (GradientTransformation | None): Optax gradient transformation (optimizer).
            sharding (Shardings | None): Shardings from `get_sharding` / `get_steps_fn`. If not provided then abstract evaluation of model state.
        Returns:
            PyTree: Initialized model weights and optimizer state if tx is provided, otherwise just model weights.
        """

        raise NotImplementedError("This function hasn't been implemented yet")

    @abc.abstractmethod
    def load_from_ckpt(self, path: str, step_number: Optional[int] = None, use_best=False) -> PyTree:
        """
        Load model weights from a checkpoint.
        Args:
            checkpointer (Checkpointer): The Checkpointer instance to load from.
            state (PyTree): The model state structure to match the checkpoint data.
            use_best (bool): Whether to use the best checkpoint or not.
        Returns:
            PyTree: The loaded model weights.
        """
        raise NotImplementedError("This function hasn't been implemented yet")

    def count_params(self, params: PyTree) -> int:
        """
        Calculate the total number of parameters in the model.
        Args:
            params (PyTree): The model parameters.
        Returns:
            int: Total number of parameters.
        """

        return jax.tree.reduce(lambda acc, p: acc + p.size, params, 0)


class HFModelBase(modelBase):
    """
    Hugging Face model abstract base class.
    This class provides methods for loading models from Hugging Face.
    """

    @abc.abstractmethod
    def load_from_hf(self, params: PyTree, model_name: str) -> PyTree:
        """
        Load model weights from Hugging Face.
        Args:
            model_name (str): The name or path of the Hugging Face model.
        Returns:
            PyTree: The loaded model weights.
        """
        raise NotImplementedError("This function hasn't been implemented yet")
