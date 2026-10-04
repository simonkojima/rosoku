# Rosoku — PyTorch experiment core

Rosoku structures model training as **Experiment → Stage → Step**. Define your
data, preprocessing, model, and loss; rosoku handles train/validation loops,
parameter selection, device movement, optimizer updates, and callback events.

The current version is `0.0.7`. This checkout exposes the Stage-based training
core; older `conventional()` and `deeplearning()` pipelines are no longer
exported. Historical examples are in `examples/legacy/`.

```python
import torch
from torch import nn
from rosoku import Experiment, Stage, SupervisedStep

model = nn.Linear(6, 2)
experiment = Experiment(model, stages=[Stage(
    name="training", epochs=10,
    train_step=SupervisedStep(nn.CrossEntropyLoss()),
    optimizer=lambda params: torch.optim.AdamW(params, lr=3e-4),
    validate_every=1,
)], device="cpu")
state = experiment.fit(train_loader, valid_loader)  # supply your DataLoaders
```

`validate_every=None` disables validation; `1` runs it every epoch; `N` runs it
after each N completed epochs within a stage. Stage-specific selectors accept
`"all"`, submodule names, or a callable returning model parameters. Every stage
creates a new optimizer/scheduler, so linear probing → fine tuning uses the same
API as ordinary one-stage training.

Public imports: `Experiment`, `Stage`, `Step`, `SupervisedStep`, `State`,
`Callback`, `OptimizerFactory`, `SchedulerFactory`, and `TrainableSelector`.

## Install and examples

```bash
python -m pip install -e .
python examples/01_supervised.py
python examples/02_linear_probe_fine_tune.py
python examples/03_eeg_positions.py
python -m unittest discover -s tests -v
```

Examples use synthetic data on CPU, without dataset downloads. The third shows
EEG plus positions through a custom Step and sample-weighted metrics.

## Documentation

Sphinx generates API pages from the source docstrings. Guides cover callback
order, stage-local stopping, validation intervals, freeze/unfreeze, BatchNorm
mode, metrics, and scheduler behavior. Build into the separate docs repository:

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs/source ~/git/rosoku-docs/latest
```

Open `~/git/rosoku-docs/latest/index.html`. Previous version folders are retained.
A local build can instead target `docs/build/html`.
