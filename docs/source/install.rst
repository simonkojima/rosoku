Installation and documentation builds
=========================================

Install the current source
------------------------------

Python 3.11 or newer is required. From your local repository:

.. code-block:: bash

   cd ~/git/rosoku
   python -m pip install -e .

The package metadata declares its runtime dependencies, including PyTorch.
The current training examples use PyTorch only; they do not download datasets
or require MOABB/Braindecode. Use an environment with an appropriate PyTorch
build for your CPU or CUDA installation.

.. code-block:: bash

   python examples/01_supervised.py
   python -m unittest discover -s tests -v

Build documentation
-----------------------

The API reference is generated from the Python docstrings using Sphinx autodoc
and Napoleon. Model code is imported from the local checkout by ``conf.py``.
Install documentation dependencies into the same environment:

.. code-block:: bash

   python -m pip install -e '.[docs]'
   python -m sphinx -W --keep-going -b html docs/source ~/git/rosoku-docs/latest

Alternatively, install ``docs/requirements.txt`` in an environment that already
contains rosoku's runtime dependencies. To build locally without updating the
separate documentation repository:

.. code-block:: bash

   python -m sphinx -W --keep-going -b html docs/source docs/build/html

Open ``index.html`` inside the chosen output directory. The separate
``rosoku-docs`` repository keeps older version folders intact and its root
``index.html`` redirects to ``latest/``. Building HTML does not publish or push
changes to a remote repository.
