"""Batch computation, independent of optimization and train/eval mode."""
from typing import Any
from collections.abc import Callable
from torch import Tensor
from .state import State

class Step:
    """Interface for computing a model output and scalar batch loss.

    Subclass this interface for custom batches, multiple model inputs, or a custom
    objective. Implement ``forward`` and ``compute_loss``. The engine stores their
    return values in ``state.output`` and ``state.loss`` respectively.

    Backward and optimizer stepping belong to ``Experiment``. A step does not set
    train/eval mode or move the model to a device."""
    def forward(self, state: State) -> Any:
        """Compute an output from the current batch and model.

        Parameters
        ----------
        state : State
            Current context containing the model, batch, and phase.

        Returns
        -------
        object
            Model output to store in ``state.output``.

        Raises
        ------
        NotImplementedError
            Unless overridden by a subclass."""
        raise NotImplementedError

    def compute_loss(self, state: State) -> Tensor:
        """Compute a scalar loss from ``state.output`` and ``state.batch``.

        Parameters
        ----------
        state : State
            Context after forward and ``on_after_forward`` callbacks.

        Returns
        -------
        torch.Tensor
            A zero-dimensional loss tensor. During training it must support backward.

        Raises
        ------
        NotImplementedError
            Unless overridden by a subclass."""
        raise NotImplementedError

class SupervisedStep(Step):
    """Supervised forward/loss computation for batches of ``(x, y)``.

    Parameters
    ----------
    loss_fn : callable
        Callable receiving ``(state.output, y)`` and returning a scalar tensor,
        for example ``torch.nn.CrossEntropyLoss()`` or ``torch.nn.MSELoss()``.

    Notes
    -----
    Forward calls ``state.model(x)`` with a single positional input. Models taking
    EEG plus positions need a custom ``forward`` method. The same instance may be
    used for training and validation; it has no optimizer or phase-mode logic."""
    def __init__(self, loss_fn: Callable[[Any, Any], Tensor]):
        self.loss_fn = loss_fn

    def forward(self, state: State) -> Any:
        """Unpack ``(x, y)`` and return ``state.model(x)``."""
        x, _ = state.batch
        return state.model(x)

    def compute_loss(self, state: State) -> Tensor:
        """Return ``loss_fn(state.output, y)`` from the current batch."""
        _, y = state.batch
        return self.loss_fn(state.output, y)
