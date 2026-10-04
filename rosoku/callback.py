"""Common callback events, dispatched in registration order."""
from .state import State

class Callback:
    """Base class of experiment observers and control hooks.

    Subclass and override only the events you need. Hooks receive the shared
    mutable ``State`` and run synchronously in callback registration order. The
    base implementations do nothing.

    Generic ``on_batch_start`` precedes the phase-specific batch-start hook;
    phase-specific batch-end precedes ``on_batch_end``. Validation phase hooks,
    including forward/loss hooks, run under ``torch.no_grad()``. Backward and
    optimizer hooks run only during training. ``on_epoch_end`` runs after both
    phases but before the scheduler advances.

    Set ``state.should_stop=True`` for stage-local early stopping. Store persistent
    history on the callback instance, since state metrics/logs are epoch-local.
    There are no exception hooks; end hooks are not guaranteed on failure.

    Examples
    --------
    >>> from rosoku import Callback
    >>> class History(Callback):
    ...     def __init__(self):
    ...         self.rows = []
    ...     def on_epoch_end(self, state):
    ...         self.rows.append(dict(state.metrics))"""
    def on_experiment_start(self, state: State) -> None:
        """Called once after fit creates fresh state, before configuring stages.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_experiment_end(self, state: State) -> None:
        """Called after all stages finish successfully.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_stage_start(self, state: State) -> None:
        """Called after parameter selection and optimizer/scheduler creation.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_stage_end(self, state: State) -> None:
        """Called after a stage finishes, before resetting its stop flag.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_epoch_start(self, state: State) -> None:
        """Called after epoch metrics/logs are cleared, before training.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_epoch_end(self, state: State) -> None:
        """Called after training and scheduled validation, before scheduler stepping.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_batch_start(self, state: State) -> None:
        """Called before the phase-specific batch-start hook in either phase.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_batch_end(self, state: State) -> None:
        """Called after the phase-specific batch-end hook in either phase.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_train_epoch_start(self, state: State) -> None:
        """Called after model.train(), with gradients enabled.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_train_epoch_end(self, state: State) -> None:
        """Called after training phase loss is stored in metrics and logs.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_train_batch_start(self, state: State) -> None:
        """Called after the generic batch-start hook, before forward.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_train_batch_end(self, state: State) -> None:
        """Called after an optimizer update, before generic batch-end.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_valid_epoch_start(self, state: State) -> None:
        """Called after model.eval(), inside torch.no_grad().

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_valid_epoch_end(self, state: State) -> None:
        """Called after validation phase loss is stored, inside no_grad().

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_valid_batch_start(self, state: State) -> None:
        """Called before validation forward, inside torch.no_grad().

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_valid_batch_end(self, state: State) -> None:
        """Called after validation loss, before generic batch-end.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_after_forward(self, state: State) -> None:
        """Called after state.output is set, before computing loss.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_after_loss(self, state: State) -> None:
        """Called after state.loss is set, before backward during training.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_after_backward(self, state: State) -> None:
        """Called after training backward, before optimizer-step hooks.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_before_optimizer_step(self, state: State) -> None:
        """Called immediately before the training optimizer updates parameters.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

    def on_after_optimizer_step(self, state: State) -> None:
        """Called immediately after the training optimizer update.

        Parameters
        ----------
        state : State
            Shared mutable experiment state."""
        pass

