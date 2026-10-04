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
    """Counters are zero-based; global_step counts completed training batches.

    Set should_stop to stop the current stage; the next stage starts normally.
    Metrics and logs are cleared at each epoch start.
    """
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
        if isinstance(value, Tensor):
            value = value.detach().cpu()
            if value.numel() == 1:
                value = value.item()
        self.logs[name] = value
