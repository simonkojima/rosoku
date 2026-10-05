API reference
=================

All main classes and factory/selector aliases are importable from ``rosoku``.
``rosoku.core`` offers the same convenience imports. This page is built from
source docstrings, so its descriptions follow the implemented behavior.

Experiment
--------------

.. autoclass:: rosoku.Experiment
   :members: fit

Stage
---------

.. autoclass:: rosoku.Stage
   :members: should_validate, validation_step

Step
--------

.. autoclass:: rosoku.Step
   :members: forward, compute_loss

.. autoclass:: rosoku.SupervisedStep
   :members: forward, compute_loss

State
---------

.. autoclass:: rosoku.State
   :members: log

Callback
------------

.. autoclass:: rosoku.Callback
   :members:

Factory and selector types
------------------------------

.. autodata:: rosoku.types.OptimizerFactory
   :annotation: = Callable[[Iterable[nn.Parameter]], Optimizer]

.. autodata:: rosoku.types.SchedulerFactory
   :annotation: = Callable[[Optimizer], Any]

.. autodata:: rosoku.types.TrainableSelector
   :annotation: = str | Sequence[str] | Callable[[nn.Module], Iterable[nn.Parameter]]

Internal parameter helper
-----------------------------

Available from ``rosoku.parameters``, not from the package root:

.. autofunction:: rosoku.parameters.configure_trainable_parameters
