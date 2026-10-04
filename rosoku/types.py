"""Factory and parameter selection types."""
from collections.abc import Callable, Iterable, Sequence
from typing import Any
from torch import nn
from torch.optim import Optimizer

#: Create a new optimizer from the selected model parameters at stage start.
OptimizerFactory = Callable[[Iterable[nn.Parameter]], Optimizer]
#: Create a new scheduler from the current stage optimizer.
SchedulerFactory = Callable[[Optimizer], Any]
#: Select all parameters, named submodules, or a callable parameter iterable.
TrainableSelector = str | Sequence[str] | Callable[[nn.Module], Iterable[nn.Parameter]]
