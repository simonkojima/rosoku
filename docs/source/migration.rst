Current API and historical examples
=======================================

This checkout exposes the Stage-based PyTorch core documented here. The
package version is ``0.0.7``. Earlier site versions describe APIs that are no
longer exported by this checkout, including:

* ``rosoku.conventional`` and ``rosoku.deeplearning``;
* the preprocessing, transfer-learning, attribution, utilities, and
  visualization subpackages from the old pipeline.

For current code, import ``Experiment``, ``Stage``, ``Step``, ``SupervisedStep``,
``State``, and ``Callback`` from ``rosoku``. Define data/preprocessing externally,
express optimization phases as stages, and add logging/checkpointing/metrics
through callbacks. The current package does not automatically split subjects,
seed random number generators, select checkpoints, or evaluate test datasets.

Old example scripts are preserved under ``examples/legacy`` and excluded from
the current examples page. They are historical reference, not runnable examples
of the current API. Older HTML snapshots in the documentation repository are
retained in their version directories; ``latest`` documents this checkout.
