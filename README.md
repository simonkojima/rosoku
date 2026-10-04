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

## Stage-based PyTorch core

The core API is `Experiment -> Stage -> Step`. `Step` defines forward and a
scalar loss; `Experiment` owns device movement, backward, optimizer updates,
and train/eval mode. Batches are supplied by the caller, so the same API works
with a custom EEG Set Transformer or a pretrained foundation model.

```python
import torch
from torch import nn
from rosoku import Experiment, Stage, SupervisedStep

step = SupervisedStep(nn.BCEWithLogitsLoss())
experiment = Experiment(
    model=model,  # your nn.Module
    stages=[Stage(
        name="training",
        epochs=300,
        train_step=step,
        optimizer=lambda params: torch.optim.AdamW(params, lr=3e-4),
        validate_every=1,
    )],
    device="cuda",  # defaults to CPU
)
state = experiment.fit(train_loader, valid_loader)
```

For linear probing followed by fine tuning, replace `stages` with:

```python
stages = [
    Stage(
        name="linear_probe", epochs=50, train_step=step,
        trainable=["classifier"],
        optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
        validate_every=1,
    ),
    Stage(
        name="fine_tune", epochs=100, train_step=step,
        trainable="all",
        optimizer=lambda params: torch.optim.AdamW(params, lr=1e-4),
        scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1,
        ),
        validate_every=5,
    ),
]
```

`trainable` accepts `"all"`, a submodule name, a sequence of submodule names,
or a callable such as `lambda model: model.classifier.parameters()`.
Each stage freezes unselected parameters, clears gradients, and creates a new
optimizer and scheduler. Parameter freezing does not freeze BatchNorm buffers
or disable Dropout; a callback can set the backbone to `eval()` in
`on_train_epoch_start` when fixed feature extraction is required.

`validate_every=None` disables validation (then `fit(train_loader)` works),
`1` validates each epoch, and `N` validates after every N completed epochs
within each stage. Enabled validation requires `valid_loader`.
`valid_step` optionally overrides `train_step` during validation.
Loaders must support fresh iteration each epoch.

Subclass `Callback` and override the events you need: experiment/stage/epoch
start and end, generic batch start and end, train/valid epoch and batch start
and end, `on_after_forward`, `on_after_loss`, `on_after_backward`, and
`on_before_optimizer_step` / `on_after_optimizer_step`. Callbacks receive the
same mutable `State` in registration order. Validation and its phase callbacks
run under `torch.no_grad()`; backward and optimizer events are train-only.
Epoch-end events run after training and any scheduled validation, before the
scheduler update. `State.should_stop=True` stops the current stage and allows
the next stage to run.

`State.epoch`, `stage_index`, and `step` are zero-based; `global_epoch` and
`global_step` count completed epochs and training batches. `metrics` and `logs`
are reset each epoch. `train/loss` and `valid/loss` are means of batch scalar
losses; custom sample-weighted metrics can be implemented in callbacks.
`state.log(name, value)` detaches tensors for logging.
Schedulers run once per epoch; `ReduceLROnPlateau` receives the current
validation loss, or training loss on epochs without validation.
Each `fit()` starts with a fresh State while retaining the model's weights.

Run the CPU tests with a Python environment containing PyTorch:

```sh
python -m unittest discover -s tests -v
```
