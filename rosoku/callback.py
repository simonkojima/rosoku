"""Common callback events, dispatched in registration order."""
from .state import State

class Callback:
    """Override only needed events. Hooks may inspect or modify State.

    Generic batch hooks run for both phases; phase-specific hooks surround the
    computation. Backward and optimizer hooks run only during training.
    """
    def on_experiment_start(self, state: State) -> None:
        pass

    def on_experiment_end(self, state: State) -> None:
        pass

    def on_stage_start(self, state: State) -> None:
        pass

    def on_stage_end(self, state: State) -> None:
        pass

    def on_epoch_start(self, state: State) -> None:
        pass

    def on_epoch_end(self, state: State) -> None:
        pass

    def on_batch_start(self, state: State) -> None:
        pass

    def on_batch_end(self, state: State) -> None:
        pass

    def on_train_epoch_start(self, state: State) -> None:
        pass

    def on_train_epoch_end(self, state: State) -> None:
        pass

    def on_train_batch_start(self, state: State) -> None:
        pass

    def on_train_batch_end(self, state: State) -> None:
        pass

    def on_valid_epoch_start(self, state: State) -> None:
        pass

    def on_valid_epoch_end(self, state: State) -> None:
        pass

    def on_valid_batch_start(self, state: State) -> None:
        pass

    def on_valid_batch_end(self, state: State) -> None:
        pass

    def on_after_forward(self, state: State) -> None:
        pass

    def on_after_loss(self, state: State) -> None:
        pass

    def on_after_backward(self, state: State) -> None:
        pass

    def on_before_optimizer_step(self, state: State) -> None:
        pass

    def on_after_optimizer_step(self, state: State) -> None:
        pass

