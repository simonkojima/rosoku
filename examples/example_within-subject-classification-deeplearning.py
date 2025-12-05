"""
Example: Within-subject classification with deep learning
=========================================================
"""

# %%
import functools

from pathlib import Path

import mne
import tag_mne as tm

import moabb.datasets

import torch
import braindecode
import rosoku


# %%
def func_get_model(X, y):
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


def func_load_epochs(keywords, mode, dataset, l_freq, h_freq, order_filter, tmin, tmax):
    subject = keywords[0]
    keywords = keywords[1:]

    sessions = dataset.get_data(subjects=[subject])
    raws = sessions[subject]["0"]

    epochs_list = []

    for (name_run, raw) in raws.items():
        if not True in [k in name_run for k in keywords]:
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


def func_proc_epochs(epochs, mode):
    # do nothing in this example
    return epochs


def convert_epochs_to_ndarray(
        epochs,
        mode,
        label_keys,
):
    X = epochs.get_data()
    y = rosoku.utils.get_labels_from_epochs(epochs, label_keys)

    return X, y


# %%

subject = 56
resample = 128

lr = 1e-3
weight_decay = 1e-2
n_epochs = 100
batch_size = 8
patience = 75
enable_euclidean_alignment = False
enable_normalization = True
device = "cuda" if torch.cuda.is_available() else "cpu"
enable_ddp = False
enable_dp = False

seed = 42

dataset = moabb.datasets.Dreyer2023()

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
    keywords_train=[subject, "R1", "R2"],
    keywords_valid=[subject, "R3"],
    keywords_test=[[subject, "R4", "R5"]],
    func_load_epochs=functools.partial(func_load_epochs, dataset=dataset, l_freq=8.0, h_freq=30.0, order_filter=4,
                                       tmin=dataset.interval[0] + 0.5, tmax=dataset.interval[1]),
    func_proc_epochs=func_proc_epochs,
    func_convert_epochs_to_ndarray=functools.partial(
        convert_epochs_to_ndarray, label_keys=label_keys
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
    early_stopping=early_stopping,
    enable_normalization=enable_normalization,
    name_classifier="eegnet4.2",
    history_fname=(save_base / "history" / f"sub-{subject}.parquet"),
    checkpoint_fname=(save_base / "checkpoint" / f"sub-{subject}.pth"),
    samples_fname=(save_base / "samples" / f"sub-{subject}.parquet"),
    normalization_fname=(save_base / "normalization" / f"sub-{subject}.msgpack"),
    saliency_map_fname=(save_base / "saliency" / f"sub-{subject}.msgpack"),
    label_keys=label_keys,
    seed=seed,
)

for m in range(results.shape[0]):
    print(results.loc[m])

results.to_parquet(save_base / "results_deeplearning_within-subject.parquet")
