"""
Example: Cross-subject classification with deep learning
========================================================
"""

# %%
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

lr = 1e-3
weight_decay = 1e-2
n_epochs = 500
batch_size = 64
patience = 75
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


def func_load_ndarray(
    keywords,
    mode,
    tmin,
    tmax,
    l_freq,
    h_freq,
    order_filter,
    resample,
    label_keys,
    dataset,
):

    X = []
    y = []
    for keyword in keywords:
        subject = int(keyword[1:])
        sessions = dataset.get_data(subjects=[subject])
        raws = sessions[subject]["0"]

        epochs_subject = list()
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
            markers = tm.add_tag(markers, f"run:{name}")
            # markers = tm.add_tag(markers, f"rtype:{rtype}")

            samples, markers = tm.remove(samples, markers, "event:misc")

            events, event_id = tm.events_from_markers(samples, markers)
            epochs = mne.Epochs(
                raw=raw,
                tmin=tmin - 1.0,
                tmax=tmax + 1.0,
                events=events,
                event_id=event_id,
                baseline=None,
            ).load_data()

            epochs.resample(resample)

            epochs_subject.append(epochs)

        epochs_subject = tm.concatenate_epochs(epochs_subject)

        print(epochs_subject)

        epochs_subject = epochs_subject.crop(tmin=tmin, tmax=tmax).pick(picks="eeg")
        print(epochs_subject.get_data().shape)

        y_subject = rosoku.utils.get_labels_from_epochs(
            epochs_subject, label_keys=label_keys
        )

        X_subject = rosoku.tl.euclidean_alignment(epochs_subject.get_data())

        y.append(y_subject)
        X.append(X_subject)

    if mode != "test":
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

    return X, y


# %%


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
        drop_prob=0.25,
    )

    return model


# %%
# label_keys = {"event:left": 0, "event:right": 1}

criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
scheduler_params = {"T_max": n_epochs, "eta_min": 1e-6}
optimizer = torch.optim.AdamW
optimizer_params = {"lr": lr, "weight_decay": weight_decay}
early_stopping = rosoku.utils.EarlyStopping(patience=patience)

results = rosoku.deeplearning(
    keywords_train=[f"A{num}" for num in range(1, 16)],
    keywords_valid=[f"A{num}" for num in range(16, 21)],
    keywords_test=["A21", "A56"],
    func_load_ndarray=functools.partial(
        func_load_ndarray,
        dataset=moabb.datasets.Dreyer2023(),
        tmin=0.5,
        tmax=4.5,
        l_freq=8.0,
        h_freq=30.0,
        order_filter=4,
        resample=128,
        label_keys={"event:left": 0, "event:right": 1},
    ),
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
    compile_test=False,
    func_proc_epochs=None,
    early_stopping=early_stopping,
    enable_normalization=enable_normalization,
    name_classifier="eegnet4.2",
    history_fname=(save_base / "history" / f"cross-subject-deeplearning.json"),
    checkpoint_fname=(save_base / "checkpoint" / f"cross-subject-deeplearning.pth"),
    desc="eegnet4.2/drop_prob=0.25",
    enable_wandb_logging=False,
    # wandb_params={
    #    "project": "wandb-project-name",
    #    "name": f"sub-{subject}",
    # },
    seed=seed,
)

for m in range(results.shape[0]):
    print(results.loc[m])
