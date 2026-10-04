Runnable examples
=====================

These scripts live in ``rosoku/examples`` in the source repository. Run them
from its root after installing rosoku. They use synthetic CPU data, require no
downloads, and have a ``main`` guard so importing them does not train a model.

One-stage supervised training
---------------------------------

.. code-block:: bash

   python examples/01_supervised.py

Shows a model, a shared supervised Step, every-epoch validation, a scheduler,
and callback-owned history. The docstring explains how to disable validation
or change its interval.

.. literalinclude:: ../../examples/01_supervised.py
   :language: python
   :linenos:

.. _linear-probe-example:

Linear probing → fine tuning
--------------------------------

.. code-block:: bash

   python examples/02_linear_probe_fine_tune.py

Uses a small encoder as a stand-in for a pretrained model. Load your pretrained
weights before Experiment construction in real applications. Verifies that
both parameters and BatchNorm buffers stay fixed during probing, then checks
unfreezing and parameter updates during fine tuning.

.. literalinclude:: ../../examples/02_linear_probe_fine_tune.py
   :language: python
   :linenos:

.. _eeg-position-example:

EEG with channel positions
------------------------------

.. code-block:: bash

   python examples/03_eeg_positions.py

Shows dictionary batches, a custom Step calling a two-input model, and
sample-weighted loss/accuracy callbacks. The illustrative position-aware
network can be replaced with your EEGSetTransformer; obtain positions using
its ``get_positions`` method and keep the channel order consistent.

.. literalinclude:: ../../examples/03_eeg_positions.py
   :language: python
   :linenos:
