# 🕯️ Rosoku — A Flexible EEG/BCI Experiment Pipeline Toolkit

**rosoku** is a *callback-based experiment pipeline* for EEG/BCI research.  
You are free to design **how your data is loaded, processed, and shaped**,  
while rosoku handles **training, evaluation, logging, and result export**.

> **You control the data.  
> rosoku handles everything after.**

Working directly with PyTorch or scikit-learn provides flexibility but requires heavy boilerplate.  
Meanwhile, frameworks like MOABB or Braindecode are convenient, but restrict custom processing.

**rosoku fills the space between them — flexibility without overhead.**

---

## 🔥 Key Idea

| Task                               | rosoku handles                    | You define                        |
|------------------------------------|-----------------------------------|-----------------------------------|
| Dataset loading                    | receives via callback             | how to load (MNE, NumPy, custom)  |
| Preprocessing / feature extraction | pluggable via callbacks           | any processing you write          |
| Training loop                      | model fitting, scheduling, saving | sklearn estimator / PyTorch model |
| Evaluation & logging               | accuracy / saliency               | optional W&B configuration        |
| Result export                      | DataFrame / parquet / msgpack     | downstream analysis or plotting   |

→ rosoku performs the *plumbing*  
→ you focus on *ideas and research*

---

## 🔧 Two Complementary Pipelines

| API              | Purpose                       | Typical models                        |
|------------------|-------------------------------|---------------------------------------|
| `conventional()` | traditional ML classification | MDM / TSClassifier / CSP / SVM / LDA  |
| `deeplearning()` | deep learning with PyTorch    | EEGNet / Braindecode / custom CNN/RNN |

Both follow the same concept:
You write data & preprocessing.
rosoku handles training & evaluation.

---

## 🚀 Quick Start

Full runnable examples are available under `examples/`.

Recommended first files:

- `examples/example_within-subject-classification-riemannian.py`
- `examples/example_within-subject-classification-deeplearning.py`