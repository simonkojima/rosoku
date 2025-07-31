"""
Example: Within-subject classification with deep learning
=========================================================
"""

# %%
import random
import functools
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import mne
import tag_mne as tm

import moabb.datasets

import torch
import braindecode
import rosoku

# %%

subject = 56
resample = 128

lr = 1e-3
weight_decay = 1e-2
n_epochs = 500
batch_size = 8
patience = 75
enable_euclidean_alignment = False
enable_normalization = True
device = "cuda" if torch.cuda.is_available() else "cpu"
enable_ddp = False
enable_dp = False

seed = 42

save_base = Path("~").expanduser() / "rosoku-log"
(save_base / "checkpoint").mkdir(parents=True, exist_ok=True)
(save_base / "history").mkdir(parents=True, exist_ok=True)

# %%

# load dataset and generate epochs


def epochs_from_raws(
    raws, runs, rtypes, tmin, tmax, l_freq, h_freq, order_filter, subject
):
    epochs_list = list()
    for raw, run, rtype in zip(raws, runs, rtypes):

        raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            method="iir",
            iir_params={"ftype": "butter", "order": 4, "btype": "bandpass"},
        )

        # eog and emg mapping
        mapping = dict()
        for ch in raw.ch_names:
            if "EOG" in ch:
                mapping[ch] = "eog"
            elif "EMG" in ch:
                mapping[ch] = "emg"

        raw.set_channel_types(mapping)
        raw.set_montage("standard_1020")

        events, event_id = mne.events_from_annotations(raw)

        samples, markers = tm.markers_from_events(events, event_id)
        markers = tm.add_tag(markers, f"subject:{subject}")
        markers = tm.add_event_names(
            markers, {"left": ["left_hand"], "right": ["right_hand"]}
        )
        markers = tm.add_tag(markers, f"run:{run}")
        markers = tm.add_tag(markers, f"rtype:{rtype}")

        samples, markers = tm.remove(samples, markers, "event:misc")

        events, event_id = tm.events_from_markers(samples, markers)
        epochs = mne.Epochs(
            raw=raw,
            tmin=tmin,
            tmax=tmax,
            events=events,
            event_id=event_id,
            baseline=None,
        )

        epochs_list.append(epochs)

    epochs = tm.concatenate_epochs(epochs_list)

    return epochs


dataset = moabb.datasets.Dreyer2023()
sessions = dataset.get_data(subjects=[subject])
raws = sessions[subject]["0"]

epochs_acquisition = epochs_from_raws(
    raws=[raws[key] for key in ["0R1acquisition", "1R2acquisition"]],
    runs=[1, 2],
    rtypes=["acquisition", "acquisition"],
    tmin=-1.0,
    tmax=5.5,
    l_freq=8.0,
    h_freq=30.0,
    order_filter=4,
    subject=subject,
).resample(resample)

epochs_online = epochs_from_raws(
    raws=[raws[key] for key in ["2R3online", "3R4online", "4R5online"]],
    runs=[3, 4, 5],
    rtypes=["online", "online", "online"],
    tmin=-1.0,
    tmax=5.5,
    l_freq=8.0,
    h_freq=30.0,
    order_filter=4,
    subject=subject,
).resample(resample)

epochs = tm.concatenate_epochs([epochs_acquisition, epochs_online])


# %%


def func_proc_epochs(epochs, mode, tmin=0.5, tmax=4.5):
    epochs = epochs.pick(picks="eeg").crop(tmin=tmin, tmax=tmax)
    return epochs


def func_load_epochs(keywords, mode, epochs):
    return epochs[keywords]


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
    keywords_train=["run:1", "run:2"],
    keywords_valid=["run:3"],
    keywords_test=["run:4"],
    func_load_epochs=functools.partial(func_load_epochs, epochs=epochs),
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
    seed=seed,
)

print(results.loc[0])
