Current core examples
=====================

Run these scripts from the repository root after installing rosoku:

* ``python examples/01_supervised.py``: one-stage training with validation.
* ``python examples/02_linear_probe_fine_tune.py``: freeze/unfreeze and factories.
* ``python examples/03_eeg_positions.py``: dictionary batches and EEG positions.

All use small synthetic data on CPU and require no downloads or external EEG
models. The ``legacy/`` directory contains examples for the removed API.
