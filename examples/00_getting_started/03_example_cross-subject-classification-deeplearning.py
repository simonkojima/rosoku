"""
Example: Cross-subject classification with deep learning
========================================================
"""

# Authors: Simon Kojima <simon.kojima@inria.fr>
#
# License: BSD (3-clause)

# %%
# Import Packages
# ===============
import functools
import numpy as np
from pathlib import Path
import mne
from moabb.datasets import Dreyer2023
import torch
import braindecode
import rosoku


# %%
# Define a callback function to load ndarray data
# ===============================================


def callback_load_ndarray(
    items,
    split,
    tmin,
    tmax,
    l_freq,
    h_freq,
    order_filter,
    label_keys,
    dataset,
):
    X_list = []
    y_list = []

    for item in items:
        subject = item
        sessions = dataset.get_data(subjects=[subject])
        raws = sessions[subject]["0"]

        for name, raw in raws.items():
            raw.filter(
                l_freq=l_freq,
                h_freq=h_freq,
                method="iir",
                iir_params={
                    "ftype": "butter",
                    "order": order_filter,
                    "btype": "bandpass",
                },
            )

            raw = raw.pick(picks="eeg")

            epochs = mne.Epochs(
                raw=raw,
                tmin=tmin,
                tmax=tmax,
                baseline=None,
            ).load_data()

            # Apply Domain Adoptation (Euclidean Alignment)
            X = rosoku.tl.euclidean_alignment(epochs.get_data())

            y = rosoku.utils.get_labels_from_epochs(epochs, label_keys=label_keys)

            X_list.append(X)
            y_list.append(y)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y


# %%
# Define a callback function to load PyTorch model
# ================================================


def callback_get_model(X, y):
    _, n_chans, n_times = X.shape
    F1 = 4
    D = 2
    F2 = F1 * D

    model = braindecode.models.EEGNet(
        n_chans=n_chans,
        n_outputs=2,
        n_times=n_times,
        F1=F1,
        D=D,
        F2=F2,
        drop_prob=0.25,
    )

    return model


# %%
# Run the Experiment
# ==================

lr = 1e-3
weight_decay = 1e-2
n_epochs = 500
batch_size = 64
patience = 75
enable_normalization = True
device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42

dataset = Dreyer2023()

save_base = Path("~").expanduser() / "rosoku-log"
(save_base / "checkpoint").mkdir(parents=True, exist_ok=True)
(save_base / "history").mkdir(parents=True, exist_ok=True)
(save_base / "saliency").mkdir(parents=True, exist_ok=True)
(save_base / "samples").mkdir(parents=True, exist_ok=True)
(save_base / "normalization").mkdir(parents=True, exist_ok=True)

criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
scheduler_params = {"T_max": n_epochs, "eta_min": 1e-6}
optimizer = torch.optim.AdamW
optimizer_params = {"lr": lr, "weight_decay": weight_decay}
early_stopping = rosoku.utils.EarlyStopping(patience=patience)

label_keys = {"left_hand": 0, "right_hand": 1}

results = rosoku.deeplearning(
    items_train=[1, 2, 3],
    items_valid=[4],
    items_test=[21, 56],
    callback_load_ndarray=functools.partial(
        callback_load_ndarray,
        dataset=dataset,
        tmin=dataset.interval[0] + 0.5,
        tmax=dataset.interval[1],
        l_freq=8.0,
        h_freq=30.0,
        order_filter=4,
        label_keys=label_keys,
    ),
    batch_size=batch_size,
    n_epochs=n_epochs,
    criterion=criterion,
    optimizer=optimizer,
    optimizer_params=optimizer_params,
    callback_get_model=callback_get_model,
    scheduler=scheduler,
    scheduler_params=scheduler_params,
    device=device,
    callback_proc_epochs=None,
    early_stopping=early_stopping,
    enable_normalization=enable_normalization,
    scoring=["accuracy", "f1"],
    history_fname=(save_base / "history" / f"cross-subject-deeplearning.parquet"),
    checkpoint_fname=(save_base / "checkpoint" / f"cross-subject-deeplearning.pth"),
    samples_fname=(save_base / "samples" / f"cross-subject-deeplearning.parquet"),
    normalization_fname=(
        save_base / "normalization" / f"cross-subject-deeplearning.msgpack"
    ),
    saliency_map_fname=(save_base / "saliency" / f"cross-subject-deeplearning.msgpack"),
    label_keys=label_keys,
    seed=seed,
    additional_values={"example_key": "example_value"},
)

# %%
# Print Results
# =============

print(results.to_string())
