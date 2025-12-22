"""
Example 02: Plot Saliency Map
==========================
"""

# Authors: Simon Kojima <simon.kojima@inria.fr>
#
# License: BSD (3-clause)

# %%
# Import Packages
# ===============
import functools
from pathlib import Path
import numpy as np
import mne
import torch
import braindecode
import rosoku
import msgpack
from moabb.datasets import Dreyer2023
import matplotlib.pyplot as plt
import seaborn as sns


# %%
# Define callback functions
# =========================
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
        drop_prob=0.5,
    )

    return model


def callback_load_epochs(
        items, split, dataset, l_freq, h_freq, order_filter, tmin, tmax
):
    subject = items[0]
    items = items[1:]

    sessions = dataset.get_data(subjects=[subject])
    raws_dict = sessions[subject]["0"]

    epochs_list = []

    for name_run, raw in raws_dict.items():
        if not True in [item in name_run for item in items]:
            continue

        raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            method="iir",
            iir_params={"ftype": "butter", "order": order_filter, "btype": "bandpass"},
        )

        raw = raw.pick(picks="eeg")

        epochs = mne.Epochs(
            raw=raw,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
        )

        epochs_list.append(epochs)

    return mne.concatenate_epochs(epochs_list)


def callback_proc_epochs(epochs, split):
    # do nothing in this example
    return epochs


def convert_epochs_to_ndarray(
        epochs,
        split,
        label_keys,
):
    X = epochs.get_data()
    y = rosoku.utils.get_labels_from_epochs(epochs, label_keys)

    return X, y


# %%
# Run the Experiment
# ==================

subject = 56
resample = 128

lr = 1e-3
weight_decay = 1e-2
n_epochs = 100
batch_size = 8
patience = 75
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
    items_train=[subject, "R1", "R2"],
    items_valid=[subject, "R3"],
    items_test=[[subject, "R4", "R5"]],
    callback_load_epochs=functools.partial(
        callback_load_epochs,
        dataset=dataset,
        l_freq=8.0,
        h_freq=30.0,
        order_filter=4,
        tmin=dataset.interval[0] + 0.5,
        tmax=dataset.interval[1],
    ),
    callback_proc_epochs=callback_proc_epochs,
    callback_convert_epochs_to_ndarray=functools.partial(
        convert_epochs_to_ndarray, label_keys=label_keys
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
    early_stopping=early_stopping,
    history_fname=(save_base / "history" / f"sub-{subject}.parquet"),
    checkpoint_fname=(save_base / "checkpoint" / f"sub-{subject}.pth"),
    samples_fname=(save_base / "samples" / f"sub-{subject}.parquet"),
    normalization_fname=(save_base / "normalization" / f"sub-{subject}.msgpack"),
    saliency_map_fname=(save_base / "saliency" / f"sub-{subject}.msgpack"),
    label_keys=label_keys,
    seed=seed,
    additional_values={"subject": subject},
)

# %%
# Print Results
# =============

print(results.to_string())

# %%
# Load Saliency Map Data
# ======================

with open(save_base / "saliency" / f"sub-{subject}.msgpack", "rb") as f:
    saliency_map_data = msgpack.load(f, strict_map_key=False)

# %%
# Plot Spatial Saliency
# =====================

spatial_saliency_left = rosoku.attribution.saliency_spatial(
    saliency_map_data[0]["left_hand"]
)
spatial_saliency_right = rosoku.attribution.saliency_spatial(
    saliency_map_data[0]["right_hand"]
)

# get channel position info
raw = list(Dreyer2023().get_data(subjects=[subject])[subject]["0"].values())[0]
ch_names = raw.pick(picks="eeg").ch_names
montage = mne.channels.make_standard_montage("standard_1005")
ch_pos = montage.get_positions()["ch_pos"]
ch_pos = np.array([ch_pos[ch_name][:2] for ch_name in ch_names])

fig, axes = plt.subplots(1, 2, figsize=(8, 5))

axes[0].set_title("Left Hand")
mne.viz.plot_topomap(
    spatial_saliency_left,
    ch_pos,
    names=None,
    cmap="seismic",
    axes=axes[0],
)

axes[1].set_title("Right Hand")
mne.viz.plot_topomap(
    spatial_saliency_left,
    ch_pos,
    names=None,
    cmap="seismic",
    axes=axes[1],
)

# %%
# Plot Temporal Saliency
# ======================

temporal_saliency_left = rosoku.attribution.saliency_temporal(
    saliency_map_data[0]["left_hand"]
)
temporal_saliency_right = rosoku.attribution.saliency_temporal(
    saliency_map_data[0]["right_hand"]
)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

sns.set()
sns.lineplot(temporal_saliency_left, ax=axes[0])
sns.lineplot(temporal_saliency_right, ax=axes[1])

axes[0].set_title("Left Hand")
axes[1].set_title("Right Hand")
