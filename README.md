# 🕯️ Rosoku — A Flexible EEG/BCI Experiment Pipeline Toolkit

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

---

## 🔥 Key Idea

| Task                               | rosoku handles                    | You define                        |
|------------------------------------|-----------------------------------|-----------------------------------|
| Dataset loading                    | receives via callback             | how to load (MNE, NumPy, custom)  |
| Preprocessing / feature extraction | pluggable via callbacks           | any processing you write          |
| Training loop                      | model fitting, scheduling, saving | sklearn estimator / PyTorch model |
| Evaluation & logging               | accuracy / saliency               | optional W&B configuration        |
| Result export                      | DataFrame / parquet / msgpack     | downstream analysis or plotting   |

---

## 🔧 Two Complementary Pipelines

| API              | Purpose                       | Typical models                        |
|------------------|-------------------------------|---------------------------------------|
| `conventional()` | traditional ML classification | MDM / TSClassifier / CSP / SVM / LDA  |
| `deeplearning()` | deep learning with PyTorch    | EEGNet / Braindecode / custom CNN/RNN |

Both follow the same concept:
You write data & preprocessing.
Rosoku handles training & evaluation.

---

## 🚀 Quick Start

Full runnable examples are available under `examples/`.

Recommended first files:

- `examples/example_within-subject-classification-riemannian.py`
- `examples/example_within-subject-classification-deeplearning.py`