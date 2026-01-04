import abc
from typing import Optional

from jaxtyping import Array, PyTree
from optax import GradientTransformation

from stax.checkpointer import Checkpointer


class modelBase(abc.ABC):
    """
    Main model abstract base class.
    This class provides foundational methods for model initialization and checkpoint loading.
    """

    @abc.abstractmethod
    def init_weights(self, key: Array, tx: Optional[GradientTransformation]) -> PyTree:
        """
        Initialize model weights and optimizer state.
        Args:
            key (Array): JAX random key for initialization.
            tx (Optional[GradientTransformation]): Optax gradient transformation (optimizer).
        Returns:
            PyTree: Initialized model weights and optimizer state if tx is provided, otherwise just model weights.
        """

        raise NotImplementedError("This function hasn't been implemented yet")

    # TODO: implement loading from ckpt_path instead of using checkpointer for only loading params
    @abc.abstractmethod
    def load_from_ckpt(self, checkpointer: Checkpointer, state: PyTree, use_best: bool = True) -> PyTree:
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


class HFModelBase(abc.ABC, modelBase):
    """
    Hugging Face model abstract base class.
    This class provides methods for loading models from Hugging Face.
    """

    @abc.abstractmethod
    def load_from_hf(self, model_name: str) -> PyTree:
        """
        Load model weights from Hugging Face.
        Args:
            model_name (str): The name or path of the Hugging Face model.
        Returns:
            PyTree: The loaded model weights.
        """
        raise NotImplementedError("This function hasn't been implemented yet")
