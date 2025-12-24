Rosoku: Flexible EEG/BCI Experiment Pipelines for Researchers
==============================================================

**Rosoku** is a research-oriented Python framework for running **reproducible EEG/BCI
experiments** with both conventional machine-learning models and deep-learning
models.

It bridges the gap between **high-level EEG/BCI frameworks** (such as MOABB and
Braindecode) and **low-level machine-learning libraries** (such as scikit-learn
and PyTorch), by providing structured yet flexible experiment pipelines.

Rosoku emphasizes **clarity, explicit control, and reproducibility** over maximum
automation or throughput, making it particularly suitable for research-oriented
experimentation.

----

Core Philosophy
---------------

Rosoku is built around a simple principle:

**You define what the data are and how they should be processed.  
Rosoku defines how experiments are executed, evaluated, and recorded.**

Instead of enforcing a fixed dataset or model API, Rosoku relies on
**explicit, callback-driven interfaces**. This design keeps every experimental
decision visible and reproducible, which is essential for method development,
ablation studies, and careful comparison of pipelines.

----

What Rosoku Provides
--------------------

Rosoku handles the *structure* of experiments, while leaving scientific choices
to the user:

- orchestration of train / validation / test splits
- grouped test evaluation (e.g., session- or subject-level aggregation)
- standardized training and evaluation loops
- transparent result aggregation and export

At the same time, Rosoku does **not** impose:

- a specific dataset format
- a fixed preprocessing pipeline

----

Two Complementary Pipelines
---------------------------

Rosoku provides two high-level APIs with a shared design philosophy:

``conventional()``
    Classical machine-learning pipelines based on scikit-learn style estimators
    (e.g., Riemannian classifiers, CSP + LDA, SVM).

``deeplearning()``
    Deep-learning pipelines based on PyTorch models
    (e.g., EEGNet, Braindecode models, or custom architectures).

Both pipelines rely on the same concepts:
**items**, **callbacks**, and **explicit evaluation groups**.

----

Who Rosoku Is For
-----------------

Rosoku is designed for researchers who:

- need fine-grained control over EEG/BCI experiments
- want transparent and inspectable pipelines
- compare multiple preprocessing or modeling choices
- prioritize reproducibility over convenience

----

Getting Started
---------------

See the :doc:`install` page for installation instructions, and explore the
:doc:`auto_examples/index` section for fully runnable examples covering both
conventional and deep-learning pipelines.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:
   
   documentation 
   auto_examples/index
   install