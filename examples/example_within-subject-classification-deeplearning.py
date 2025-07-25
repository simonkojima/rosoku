"""
Example: Within-subject classification with deep learning
=========================================================
"""

# %%
import random
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import mne
import tag_mne as tm

import torch
import braindecode
import rosoku

# %%

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

# %%

subject = 7

lr = 1e-3
weight_decay = 1e-2
n_epochs = 500
batch_size = 2
patience = 75
enable_euclidean_alignment = False
enable_normalization = True
device = "cuda" if torch.cuda.is_available() else "cpu"
enable_ddp = False
enable_dp = False

save_base = Path("~").expanduser() / "rosoku-log"
(save_base / "checkpoint").mkdir(parents=True, exist_ok=True)
(save_base / "history").mkdir(parents=True, exist_ok=True)

# %%


def func_proc_epochs(epochs, mode, tmin=0, tmax=2):
    epochs = epochs.pick(picks="eeg").crop(tmin=tmin, tmax=tmax)
    return epochs


def func_load_epochs(keywords, mode, tmin=-1, tmax=2.5, runs=[4, 8, 12]):

    subject = keywords[0]

    fnames = mne.datasets.eegbci.load_data(subject, runs, update_path=True)

    epochs_list = []
    for fname, run in zip(fnames, runs):
        raw = mne.io.read_raw_edf(fname, preload=True)

        events, event_id = mne.events_from_annotations(raw)

        samples, markers = tm.markers_from_events(events, event_id)

        event_names = {"left": "T1", "right": "T2"}
        markers = tm.add_event_names(markers, event_names)

        markers = tm.add_tag(markers, f"run:{run}")
        samples, markers = tm.remove(samples, markers, "marker:T0")

        events, event_id = tm.events_from_markers(samples, markers)

        raw.filter(
            l_freq=12,
            h_freq=15,
            method="iir",
            iir_params={"order": 4, "ftype": "butterworth", "btype": "bandpass"},
            phase="zero",
        )

        epochs_run = mne.Epochs(
            raw, events, event_id, tmin=tmin, tmax=tmax, preload=True
        )

        epochs_run = epochs_run.resample(128)

        epochs_list.append(epochs_run)

    epochs = tm.concatenate_epochs(epochs_list)

    if mode == "train":
        return epochs["run:4"]
    elif mode == "valid":
        return epochs["run:8"]
    elif mode == "test":
        return epochs["run:12"]


def func_get_model(X, y):
    _, n_chans, n_times = X.shape
    F1 = 4
    D = 2
    F2 = F1 * D

    model = braindecode.models.EEGNetv4(
        n_chans=n_chans,
        n_outputs=2,
        n_times=n_times,
        F1=F1,
        D=D,
        F2=F2,
        drop_prob=0.5,
    )

    return model


# %%
label_keys = {"event:left": 0, "event:right": 1}

criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
scheduler_params = {"T_max": n_epochs, "eta_min": 1e-6}
optimizer = torch.optim.AdamW
optimizer_params = {"lr": lr, "weight_decay": weight_decay}
early_stopping = rosoku.utils.EarlyStopping(patience=patience)

results = rosoku.deeplearning(
    keywords_train=[subject],
    keywords_valid=[subject],
    keywords_test=[subject],
    func_load_epochs=func_load_epochs,
    apply_func_proc_per_obj=True,
    batch_size=batch_size,
    n_epochs=n_epochs,
    criterion=criterion,
    optimizer=optimizer,
    optimizer_params=optimizer_params,
    func_get_model=func_get_model,
    scheduler=scheduler,
    scheduler_params=scheduler_params,
    device=device,
    enable_ddp=enable_ddp,
    func_proc_epochs=func_proc_epochs,
    label_keys=label_keys,
    early_stopping=early_stopping,
    enable_normalization=enable_normalization,
    name_classifier="eegnet4.2",
    history_fname=(save_base / "history" / f"sub-{subject}"),
    checkpoint_fname=(save_base / "checkpoint" / f"sub-{subject}"),
    desc="eegnet4.2/drop_prob=0.25",
    enable_wandb_logging=False,
    wandb_params={
        "project": "wandb-project-name",
        "name": f"sub-{subject}",
    },
)

print(results.loc[0])
