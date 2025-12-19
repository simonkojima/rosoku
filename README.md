# 🕯️ Rosoku — A Flexible EEG/BCI Experiment Pipeline Toolkit

**Rosoku** is a research-oriented Python framework for running **reproducible EEG/BCI
experiments** with both conventional machine-learning models and deep-learning
models.

It bridges the gap between **high-level EEG/BCI frameworks** (such as MOABB and
Braindecode) and **low-level machine-learning libraries** (such as scikit-learn
and PyTorch), by providing structured yet flexible experiment pipelines without
hiding critical details.

Rosoku emphasizes **clarity, reproducibility, and experimental control** over
maximum automation or throughput.

---

## 🔥 Core Philosophy

Rosoku is designed around a simple idea:

> **You define what the data are and how they should be processed.  
> Rosoku defines how experiments are run, evaluated, and recorded.**

Rather than enforcing a fixed dataset or model API, Rosoku relies on
**explicit, callback-driven interfaces** that make each experimental choice
visible and reproducible.

This makes Rosoku particularly suitable for:
- method development
- ablation studies
- cross-subject / cross-session analysis
- careful comparison of pipelines in research papers

---

## 🧠 What Rosoku Does (and What You Control)

| Task                               | Rosoku handles                          | You define                                  |
|------------------------------------|-----------------------------------------|----------------------------------------------|
| Dataset orchestration              | train/valid/test grouping               | what an *item* means                         |
| Data loading                       | unified pipeline                        | how to load (MNE, NumPy, custom)             |
| Preprocessing                      | execution & split handling              | any signal processing you write              |
| Training loop                      | fitting, scheduling, checkpointing      | sklearn estimator / PyTorch model             |
| Evaluation                         | scoring, grouping, aggregation          | metrics, saliency, logging                   |
| Result export                      | parquet / msgpack / pth                 | downstream analysis or plotting              |

Rosoku **does not**:
- impose a dataset format
- hide training logic behind opaque abstractions
- silently modify randomness or preprocessing behavior

---

## 🔧 Two Complementary Pipelines

| API              | Purpose                                | Typical models                             |
|------------------|----------------------------------------|--------------------------------------------|
| `conventional()` | classical ML classification             | MDM / TSClassifier / CSP / SVM / LDA        |
| `deeplearning()` | deep learning with PyTorch              | EEGNet / Braindecode / custom CNN/RNN       |

Both pipelines follow the same design:

1. You define **items** describing which data belong to each split
2. You provide **callbacks** to load and preprocess data
3. Rosoku runs training, evaluation, and result aggregation

This shared structure makes it easy to compare classical and deep-learning
approaches within the same experimental setup.

---

## 🧪 Reproducibility First

Rosoku is designed with **reproducibility as a first-class concern**:

- deterministic training is supported via explicit seeding
- data loading behavior is transparent
- no implicit parallelism is used

> **For maximum reproducibility, Rosoku recommends running with**
> ```python
> num_workers = 0
> ```
> especially when publishing or debugging experiments.

---

## 🚀 Quick Start

Full runnable examples are available under `examples/`.

Recommended first files:

- `examples/example_within-subject-classification-riemannian.py`
- `examples/example_within-subject-classification-deeplearning.py`

These examples demonstrate:
- item-based dataset definition
- grouped test evaluation
- conventional vs deep-learning pipelines
- reproducible experiment execution

---

## ✨ Who Is Rosoku For?

Rosoku is **not** a black-box AutoML tool.

It is designed for researchers who:
- want to **understand and control** every step of their pipeline
- need **transparent experiments** for publications
- work across **multiple datasets, subjects, or sessions**
- value **explicitness over convenience**

If you prefer maximum automation, MOABB or Braindecode may be a better fit.  
If you want a clear, inspectable bridge between theory and implementation,
Rosoku is built for you.