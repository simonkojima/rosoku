"""Mutable state shared by steps and callbacks."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from torch import Tensor, nn
from torch.optim import Optimizer
if TYPE_CHECKING:
    from .stage import Stage

@dataclass
class State:
    """Mutable experiment context shared by steps and callbacks.

    Attributes
    ----------
    model : torch.nn.Module
        Current model.
    stage : Stage or None
        Current stage; ``None`` before the first stage.
    stage_index : int
        Zero-based stage index.
    phase : str or None
        ``"train"`` or ``"valid"`` during phase hooks. ``None`` at outer epoch hooks.
    epoch : int
        Zero-based epoch index, reset at every stage.
    global_epoch : int
        Count of epochs completed so far. Updated after epoch-end callbacks and
        scheduler stepping, so it still reflects prior epochs inside those hooks.
    step : int
        Zero-based batch index, reset for each train or validation phase.
    global_step : int
        Number of training batches completed so far. Updated after batch-end hooks.
    optimizer : torch.optim.Optimizer or None
        Optimizer for the current stage.
    scheduler : object or None
        Scheduler for the current stage.
    batch : object
        Current batch after recursive movement of tensor leaves to the device.
    output : object
        Current output set by ``Step.forward``. Reset before each batch.
    loss : torch.Tensor or None
        Current scalar loss set by ``Step.compute_loss``. Reset before each batch.
    metrics : dict
        Mutable metric storage. The engine adds ``train/loss`` and, when run,
        ``valid/loss``. Cleared at each stage and epoch start.
    logs : dict
        Arbitrary logged values, cleared alongside ``metrics``. ``log`` detaches
        tensor values; direct assignment does not.
    should_stop : bool
        Request to stop the current stage, reset when that stage finishes.

    Notes
    -----
    This is transient state, not a checkpoint or a history accumulator. The final
    batch/output/loss remain available, and loss/output may retain training graphs.
    Detach values when keeping them in callback-owned history across iterations."""
    model: nn.Module
    stage: Stage | None = None
    stage_index: int = 0
    phase: str | None = None
    epoch: int = 0
    global_epoch: int = 0
    step: int = 0
    global_step: int = 0
    optimizer: Optimizer | None = None
    scheduler: Any = None
    batch: Any = None
    output: Any = None
    loss: Tensor | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: dict[str, Any] = field(default_factory=dict)
    should_stop: bool = False

    def log(self, name: str, value: Any) -> None:
        """Store a value in the current epoch's log dictionary.

        Parameters
        ----------
        name : str
            Log key. Existing values at the same key are replaced.
        value : object
            Value to store. Tensor values are detached and moved to CPU; one-element
            tensors become Python scalars. Other values are stored unchanged.

        Notes
        -----
        This does not persist a history or write to disk. ``logs`` is cleared at the
        start of the next epoch or stage."""
        if isinstance(value, Tensor):
            value = value.detach().cpu()
            if value.numel() == 1:
                value = value.item()
        self.logs[name] = value
