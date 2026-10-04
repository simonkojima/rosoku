"""Configuration for one training stage."""
from dataclasses import dataclass
from .step import Step
from .types import OptimizerFactory, SchedulerFactory, TrainableSelector

@dataclass
class Stage:
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
        """epoch is zero-based, relative to this stage."""
        return self.validate_every is not None and (epoch + 1) % self.validate_every == 0

    @property
    def validation_step(self) -> Step:
        return self.train_step if self.valid_step is None else self.valid_step
