"""
Example: Within-subject classification with riemannian classifier
=================================================================
"""

# %%
import functools
from pathlib import Path
import mne
import moabb.datasets
import pyriemann

import rosoku

mne.set_log_level("CRITICAL")


# %%


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
    X = pyriemann.estimation.Covariances().transform(X)
    y = rosoku.utils.get_labels_from_epochs(epochs, label_keys)

    return X, y


# %%

subject = 56

dataset = moabb.datasets.Dreyer2023()
label_keys = {"left_hand": 0, "right_hand": 1}

save_base = Path("~").expanduser() / "rosoku-log"

results = rosoku.conventional(
    keywords_train=[subject, "R1", "R2"],
    keywords_test=[[subject, "R3", "R4", "R5"]],
    func_load_epochs=functools.partial(func_load_epochs, dataset=dataset, l_freq=8.0, h_freq=30.0, order_filter=4,
                                       tmin=dataset.interval[0] + 0.5, tmax=dataset.interval[1]),
    func_proc_epochs=func_proc_epochs,
    func_convert_epochs_to_ndarray=functools.partial(
        convert_epochs_to_ndarray, label_keys=label_keys
    ),
    samples_fname=save_base / "samples.parquet"
)

for m in range(results.shape[0]):
    print(results.loc[m])

results.to_parquet(save_base / "results.parquet")
