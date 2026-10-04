"""Experiment -> Stage -> Step execution engine."""
from collections.abc import Iterable, Sequence
from typing import Any
import torch
from torch import Tensor, nn
from .callback import Callback
from .parameters import configure_trainable_parameters
from .stage import Stage
from .state import State
from .step import Step

class Experiment:
    def __init__(self, model: nn.Module, stages: Sequence[Stage],
                 callbacks: Sequence[Callback] | None = None,
                 device: str | torch.device = "cpu"):
        if not stages:
            raise ValueError("At least one Stage is required")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.stages = list(stages)
        self.callbacks = list(callbacks or [])
        self.state = State(model=self.model)

    def _call(self, event: str) -> None:
        for callback in self.callbacks:
            getattr(callback, event)(self.state)

    def _move_to_device(self, value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.to(self.device)
        if isinstance(value, tuple):
            return tuple(self._move_to_device(v) for v in value)
        if isinstance(value, list):
            return [self._move_to_device(v) for v in value]
        if isinstance(value, dict):
            return {k: self._move_to_device(v) for k, v in value.items()}
        return value

    def fit(self, train_loader: Iterable, valid_loader: Iterable | None = None) -> State:
        """Run all stages with fresh state and optimizers on each invocation.

        Loaders must be re-iterable across epochs. Validation requires a loader
        whenever any stage enables it. Schedulers step once after epoch_end;
        ReduceLROnPlateau uses the current valid/loss (or train/loss when disabled).
        Phase losses are means of batch scalar losses, not sample-weighted means.
        """
        if valid_loader is None and any(s.validate_every is not None for s in self.stages):
            raise ValueError("valid_loader is required when validation is enabled")
        self.state = State(model=self.model)
        state = self.state
        self._call("on_experiment_start")
        for index, stage in enumerate(self.stages):
            state.stage, state.stage_index = stage, index
            state.epoch, state.step, state.phase = 0, 0, None
            state.batch = state.output = state.loss = None
            state.metrics.clear()
            state.logs.clear()
            state.should_stop = False
            params = configure_trainable_parameters(self.model, stage.trainable)
            state.optimizer = stage.optimizer(params)
            state.scheduler = stage.scheduler(state.optimizer) if stage.scheduler else None
            self._call("on_stage_start")
            for epoch in range(stage.epochs):
                if state.should_stop:
                    break
                state.epoch = epoch
                state.phase = None
                state.metrics.clear()
                state.logs.clear()
                self._call("on_epoch_start")
                if state.should_stop:
                    break
                self._run_epoch(train_loader, stage.train_step, "train")
                if not state.should_stop and stage.should_validate(epoch):
                    self._run_epoch(valid_loader, stage.validation_step, "valid")
                state.phase = None
                self._call("on_epoch_end")
                if state.scheduler is not None:
                    if isinstance(state.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        metric = state.metrics.get("valid/loss", state.metrics["train/loss"])
                        state.scheduler.step(metric)
                    else:
                        state.scheduler.step()
                state.global_epoch += 1
            self._call("on_stage_end")
            state.should_stop = False
        self._call("on_experiment_end")
        return state

    def _run_epoch(self, loader: Iterable, step: Step, phase: str) -> None:
        state = self.state
        training = phase == "train"
        state.phase, state.step = phase, 0
        state.batch = state.output = state.loss = None
        self.model.train(training)
        total, count = 0.0, 0
        # Include validation callbacks in no_grad, not just forward computation.
        with torch.enable_grad() if training else torch.no_grad():
            self._call(f"on_{phase}_epoch_start")
            for index, batch in enumerate(loader):
                if state.should_stop:
                    break
                state.step = index
                state.batch = self._move_to_device(batch)
                state.output = state.loss = None
                self._call("on_batch_start")
                self._call(f"on_{phase}_batch_start")
                if state.should_stop:
                    break
                if training:
                    state.optimizer.zero_grad(set_to_none=True)
                state.output = step.forward(state)
                self._call("on_after_forward")
                state.loss = step.compute_loss(state)
                self._call("on_after_loss")
                if not isinstance(state.loss, Tensor) or state.loss.ndim != 0:
                    raise ValueError("Step.compute_loss must return a scalar Tensor")
                if training:
                    state.loss.backward()
                    self._call("on_after_backward")
                    self._call("on_before_optimizer_step")
                    state.optimizer.step()
                    self._call("on_after_optimizer_step")
                total += state.loss.detach().item()
                count += 1
                self._call(f"on_{phase}_batch_end")
                self._call("on_batch_end")
                if training:
                    state.global_step += 1
            if count == 0 and not state.should_stop:
                raise ValueError(f"{phase} loader produced no batches")
            if count:
                state.metrics[f"{phase}/loss"] = total / count
                state.log(f"{phase}/loss", total / count)
            self._call(f"on_{phase}_epoch_end")
