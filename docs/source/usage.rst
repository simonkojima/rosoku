Training guide
==================

Define a model, a Step, and a Stage
---------------------------------------

``SupervisedStep`` expects batches of ``(x, y)`` and calls ``model(x)``. Choose a
loss that returns a scalar tensor. For classification, logits and integer
labels work with PyTorch's cross-entropy loss.

.. code-block:: python

   import torch
   from torch import nn
   from rosoku import Experiment, Stage, SupervisedStep

   model = nn.Linear(6, 2)
   stage = Stage(
       name="training",
       epochs=10,
       train_step=SupervisedStep(nn.CrossEntropyLoss()),
       optimizer=lambda params: torch.optim.AdamW(params, lr=3e-4),
       validate_every=1,
   )
   experiment = Experiment(model, stages=[stage], device="cpu")
   state = experiment.fit(train_loader, valid_loader)

``train_loader`` and ``valid_loader`` must be re-iterable. The engine recursively
moves tensor leaves of tuple, list, and dictionary batches to its device. A
custom object with its own tensor fields needs caller-side handling.

Validation frequency
------------------------

================== ==============================================
``validate_every`` Behavior
================== ==============================================
``None``           Disabled; use ``fit(train_loader)``.
``1``              Validation after every training epoch.
``N``              Validation after epochs N, 2N, ... in each stage.
================== ==============================================

Epoch indices in ``State`` are zero-based. Frequency counts completed epochs,
so ``validate_every=2`` runs at ``state.epoch == 1, 3, ...``. Validation is not
forced on the last epoch. A validation loader is required for every enabled
configuration, even if that stage has fewer epochs than its interval.

A separate ``valid_step`` can change the validation objective. Otherwise the
same ``train_step`` instance is reused. Training enables gradients and calls
``model.train()``; validation and its phase hooks use ``model.eval()`` and
``torch.no_grad()``. The final model mode is whichever phase ran last.

Linear probing and fine tuning
----------------------------------

Select parameters by ``"all"``, a submodule name, a sequence of submodule
names, or a callable returning model parameters:

.. code-block:: python

   step = SupervisedStep(nn.CrossEntropyLoss())
   stages = [
       Stage("linear_probe", 10,
             optimizer=lambda p: torch.optim.AdamW(p, lr=1e-3),
             train_step=step, trainable=["classifier"], validate_every=1),
       Stage("fine_tune", 30,
             optimizer=lambda p: torch.optim.AdamW(p, lr=1e-4),
             train_step=step, trainable="all", validate_every=1),
   ]

Each stage clears stale gradients, freezes unselected parameters, enables the
selected parameters, and creates a fresh optimizer/scheduler. Fine tuning with
``"all"`` unfreezes every model parameter, including ones frozen before the run.
Nested names such as ``encoder.blocks.0`` use PyTorch's named-submodule paths.
Callable selections must return parameters belonging to the model.

Parameter freezing does not disable Dropout or freeze BatchNorm statistics.
Use a training-phase callback to keep a frozen feature extractor in eval mode:

.. code-block:: python

   from rosoku import Callback

   class FrozenEncoderMode(Callback):
       def on_train_epoch_start(self, state):
           if state.stage.name == "linear_probe":
               state.model.encoder.eval()

This hook runs after the engine's ``model.train()``. See the runnable
:ref:`linear-probe-example` for checks that parameters and buffers stay fixed.

Custom batches and EEG positions
------------------------------------

Subclass ``Step`` for models requiring several inputs. The engine assigns
``state.output`` before computing loss:

.. code-block:: python

   from rosoku import Step

   class EEGStep(Step):
       def forward(self, state):
           return state.model(state.batch["eeg"], state.batch["pos"])

       def compute_loss(self, state):
           return nn.functional.cross_entropy(
               state.output, state.batch["label"])

For EEGSetTransformer, use its ``get_positions(channel_names)`` result in the
same order as the EEG channel dimension. Position computation and dataset
construction remain outside the training engine. The
:ref:`eeg-position-example` demonstrates dictionary batches and multiple inputs.

Callback event order
------------------------

Callbacks run synchronously in registration order. Normal execution follows:

.. code-block:: text

   on_experiment_start
     configure selected parameters, optimizer, scheduler
     on_stage_start
       clear metrics/logs; on_epoch_start
         model.train(); on_train_epoch_start
           on_batch_start; on_train_batch_start
           forward; on_after_forward
           loss; on_after_loss
           backward; on_after_backward
           on_before_optimizer_step; optimizer.step(); on_after_optimizer_step
           on_train_batch_end; on_batch_end; increment global_step
         store train/loss; on_train_epoch_end
         if scheduled: model.eval(); enter no_grad
           on_valid_epoch_start
             on_batch_start; on_valid_batch_start
             forward; on_after_forward
             loss; on_after_loss
             on_valid_batch_end; on_batch_end
           store valid/loss; on_valid_epoch_end
       on_epoch_end; scheduler.step(); increment global_epoch
     on_stage_end; reset should_stop
   on_experiment_end

``state.phase`` is ``"train"``/``"valid"`` in phase hooks and ``None`` in outer
epoch hooks. Stage-end and experiment-end callbacks can inspect final state;
exceptions propagate and do not guarantee end callbacks.

State, metrics, and stopping
--------------------------------

``epoch``, ``stage_index``, and ``step`` are zero-based indices. ``global_epoch``
and ``global_step`` count completed epochs and training batches. These counters
advance after their corresponding end callbacks.

The engine stores ``train/loss`` and ``valid/loss`` as **means of batch scalar
losses**. For unequal batch sizes, this differs from a sample-weighted loss.
Implement sample-weighted metrics in phase/batch callbacks, as demonstrated in
:ref:`eeg-position-example`.

Both ``metrics`` and ``logs`` are cleared at each stage/epoch start. Skipped
validation leaves no stale valid loss. ``State.log`` detaches tensors, moves
them to CPU, and converts one-element tensors to scalars. Store persistent
history on callback instances, detaching any tensors to avoid retaining graphs.

Set ``state.should_stop = True`` to stop the **current stage**; the next stage
starts with the flag cleared. At batch start this prevents computation. At
later computation hooks the current batch finishes before the stop takes
effect. A stop during training skips the remaining validation for that epoch.

Schedulers and repeated fits
--------------------------------

``Stage.scheduler`` is an optimizer-to-scheduler factory. Its ``step`` runs once
per epoch after ``on_epoch_end``. Thus an LR read inside that hook is the value
used for training the just-completed epoch. ``ReduceLROnPlateau`` receives this
epoch's valid loss when present, otherwise its train loss.

Repeated ``fit`` calls keep model weights but replace ``State`` and recreate
optimizers/schedulers. Callback objects are retained, so reset callback-local
counters/history in ``on_experiment_start`` when needed. This is not checkpoint
resume: optimizer state, counters, and completed stages are not restored.
