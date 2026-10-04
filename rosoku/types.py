"""Factory and parameter selection types."""
from collections.abc import Callable, Iterable, Sequence
from typing import Any
from torch import nn
from torch.optim import Optimizer

OptimizerFactory = Callable[[Iterable[nn.Parameter]], Optimizer]
SchedulerFactory = Callable[[Optimizer], Any]
TrainableSelector = str | Sequence[str] | Callable[[nn.Module], Iterable[nn.Parameter]]
