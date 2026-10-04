"""Configuration for one training stage."""
from dataclasses import dataclass
from .step import Step
from .types import OptimizerFactory, SchedulerFactory, TrainableSelector

@dataclass
class Stage:
    """Configure one training phase, including optimization and validation.

    Parameters
    ----------
    name : str
        Stage label available to steps and callbacks through ``state.stage``.
    epochs : int
        Positive number of epochs. Boolean and noninteger values are rejected.
    optimizer : OptimizerFactory
        Callable receiving the selected model parameters and returning a new
        PyTorch optimizer at stage start.
    train_step : Step
        Forward/loss definition used for training.
    trainable : TrainableSelector, default "all"
        ``"all"``, a submodule name, a sequence of submodule names, or a callable
        returning model parameters. Unselected parameters are frozen and stale
        gradients are cleared. Shared parameters are deduplicated.
    scheduler : SchedulerFactory, optional
        Callable receiving the stage optimizer and returning a new scheduler.
        The engine calls ``step`` once per epoch, after epoch-end callbacks.
    validate_every : int or None, default 1
        ``None`` disables validation, ``1`` validates every epoch, and ``N`` runs
        validation after each N completed epochs relative to this stage. Must be
        a positive integer when enabled. The last epoch is not forcibly validated.
    valid_step : Step, optional
        Validation forward/loss definition. Defaults to ``train_step``.

    Notes
    -----
    Parameter freezing changes ``requires_grad``; it does not freeze BatchNorm
    buffers or disable Dropout. To keep a frozen encoder in eval mode during
    linear probing, use ``Callback.on_train_epoch_start`` after the engine calls
    ``model.train()``. Fine tuning with ``trainable="all"`` unfreezes all parameters.

    Examples
    --------
    Create a stage without validation:

    >>> from torch import nn
    >>> from torch.optim import SGD
    >>> from rosoku import Stage, SupervisedStep
    >>> stage = Stage("training", 3, lambda p: SGD(p, lr=0.1),
    ...               SupervisedStep(nn.MSELoss()), validate_every=None)
    >>> stage.should_validate(0)
    False"""
    name: str
    epochs: int
    optimizer: OptimizerFactory
    train_step: Step
    trainable: TrainableSelector = "all"
    scheduler: SchedulerFactory | None = None
    validate_every: int | None = 1
    valid_step: Step | None = None

    def __post_init__(self) -> None:
        if type(self.epochs) is not int or self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if self.validate_every is not None and (
            type(self.validate_every) is not int or self.validate_every <= 0
        ):
            raise ValueError("validate_every must be None or a positive integer")

    def should_validate(self, epoch: int) -> bool:
        """Return whether validation is scheduled after a zero-based stage epoch.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index within the current stage.

        Returns
        -------
        bool
            Whether ``(epoch + 1)`` is divisible by the validation interval."""
        return self.validate_every is not None and (epoch + 1) % self.validate_every == 0

    @property
    def validation_step(self) -> Step:
        """Step used during validation, falling back to ``train_step``."""
        return self.train_step if self.valid_step is None else self.valid_step
