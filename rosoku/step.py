"""Batch computation, independent of optimization and train/eval mode."""
from typing import Any
from collections.abc import Callable
from torch import Tensor
from .state import State

class Step:
    """Override forward and compute_loss; Experiment owns backward/step."""
    def forward(self, state: State) -> Any:
        raise NotImplementedError

    def compute_loss(self, state: State) -> Tensor:
        raise NotImplementedError

class SupervisedStep(Step):
    """Supervised batches (x, y); custom batch formats can subclass Step."""
    def __init__(self, loss_fn: Callable[[Any, Any], Tensor]):
        self.loss_fn = loss_fn

    def forward(self, state: State) -> Any:
        x, _ = state.batch
        return state.model(x)

    def compute_loss(self, state: State) -> Tensor:
        _, y = state.batch
        return self.loss_fn(state.output, y)
