Rosoku |release|: PyTorch experiment core
=============================================

Rosoku organizes training as **Experiment → Stage → Step**. You provide a
PyTorch model, batch loaders, and scientific choices; the engine manages phase
mode, device movement, optimization, validation, and callback events.

A single stage supports ordinary EEG model training. Multiple stages naturally
express linear probing followed by fine tuning. Import the public API directly:

.. code-block:: python

   from rosoku import Experiment, Stage, Step, SupervisedStep, State, Callback

The current package is a small training core. Data splitting, preprocessing,
seeding, test evaluation, history persistence, and checkpoint selection belong
to caller code or callbacks. The historical ``conventional()`` and
``deeplearning()`` entry points are not part of this release's API.

.. toctree::
   :maxdepth: 2

   install
   usage
   documentation
   examples
   migration
