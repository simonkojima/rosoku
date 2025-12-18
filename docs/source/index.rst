.. rosoku documentation master file, created by
   sphinx-quickstart on Wed Jul 23 13:48:32 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Rosoku: Flexible EEG/BCI Experiment Pipelines for Researchers
==============================================================

**Rosoku** is a research-oriented Python framework for running reproducible EEG/BCI
experiments with both conventional machine-learning models and deep-learning
models.
It bridges the gap between high-level EEG/BCI frameworks (such as MOABB and Braindecode) and
low-level machine-learning libraries (such as scikit-learn and PyTorch).

It provides flexible, callback-driven pipelines to load data, preprocess signals,
train models, evaluate performance, and export results, while explicitly supporting
EEG-specific experimental structures such as subject/session splits and grouped
test evaluations.

rosoku is designed for researchers who need to rapidly prototype, compare, and
analyze multiple experimental configurations in a transparent and reproducible way,
rather than for generic end-to-end machine-learning workflows.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:
   
   documentation 
   auto_examples/index
   install