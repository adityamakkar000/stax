from typing import Any, Callable, Dict, Protocol, Tuple, Union

from jaxtyping import Array, PyTree

from stax.model_module import modelBase

Params = PyTree
Batch = PyTree
OptState = PyTree
Metrics = Dict[str, Array]

# StepFn: (model, params, *batch, train=True/False) -> Union[loss, (loss, aux)]
StepFn = Callable[[modelBase, PyTree, tuple[PyTree, ...], bool], Union[Array, Tuple[Array, PyTree]]]
# SingleStepFn: (params, *batch, train=True/False) -> Union[loss, (loss, aux)]
SingleStepFn = Callable[[PyTree, tuple[PyTree, ...], bool], Union[Array, Tuple[Array, PyTree]]]


class TrainFn(Protocol):
    def __call__(self, params: Params, opt_state: OptState, *batch: Batch, **loss_kwargs) -> Dict[str, Any]:
        """
        Performs a training step, including gradient calculation and parameter updates.

        Args:
            params (Params): Current model parameters.
            opt_state (OptState): Current optimizer state.
            *batch (Batch): The input batch.
            **loss_kwargs: Additional keyword arguments for the loss function.

        Returns:
            Dict[str, Any]: A dictionary containing updated 'metrics', 'params', and 'opt_state'.
        """
        ...


class ValFn(Protocol):
    def __call__(self, params: Params, *batch: Batch, **loss_kwargs) -> Metrics:
        """
        Performs a validation step over batches.

        Args:
            params (Params): Current model parameters.
            *batch (Batch): The input batch.
            **loss_kwargs: Additional keyword arguments for the loss function.
        Returns:
            Metrics: A dictionary of averaged metrics.
        """
        ...


def process_aux(out: Union[Array, Tuple[Array, PyTree]], has_aux: bool = True) -> Metrics:
    """
    Processes the output of a step function to extract metrics.

    Args:
        out: The output from the step function. Can be just loss or (loss, aux).
        has_aux: Whether the output contains auxiliary data (metrics).

    Returns:
        A dictionary of metrics. If no aux data, returns {'loss': out}.
    """
    if has_aux:
        _, metrics = out
    else:
        metrics = {"loss": out}
    return metrics  # type: ignore
