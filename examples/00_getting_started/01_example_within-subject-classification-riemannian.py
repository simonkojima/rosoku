"""
Example 01: Within-subject classification with Riemannian classifier
====================================================================

This example demonstrates a **within-subject motor imagery classification**
pipeline using a **Riemannian geometry-based classifier** with ``rosoku``.

We use the **Dreyer2023** dataset and evaluate classification performance on a
single subject, where training and test sets are defined by different recording
runs of the same subject.

The example highlights a *classical EEG decoding pipeline* and serves as a
baseline reference for comparison with deep-learning-based approaches.

Key aspects illustrated in this example include:

- Callback-based loading and preprocessing of raw EEG data using MNE
- Conversion of epoched EEG signals into **SPD covariance matrices**
- Use of a Riemannian classifier implemented via **pyRiemann**
- Simple and transparent experimental design with minimal hyperparameters

The pipeline consists of the following steps:

1. Load and band-pass filter raw EEG recordings
2. Epoch the continuous data and extract EEG channels
3. Estimate covariance matrices from epoched signals
4. Train and evaluate a Riemannian classifier on subject-specific runs
5. Collect performance metrics and per-trial predictions

This example is intentionally kept minimal and interpretable, and can be used
as a starting point or a baseline for more advanced EEG decoding pipelines
implemented with ``rosoku``.
"""

# Authors: Simon Kojima <simon.kojima@inria.fr>
#
# License: BSD (3-clause)

# %%
# Import Packages
# ===============
import functools
from pathlib import Path
import mne
import pyriemann
import rosoku

from moabb.datasets import Dreyer2023


# %%
# Define callback functions
# =========================


def callback_load_epochs(
    items, split, dataset, l_freq, h_freq, order_filter, tmin, tmax
):
    subject = items[0]
    keywords = items[1:]

    sessions = dataset.get_data(subjects=[subject])
    raws_dict = sessions[subject]["0"]

    epochs_list = []

    for name_run, raw in raws_dict.items():
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


def callback_proc_epochs(epochs, split):
    # do nothing in this example
    return epochs


def callback_convert_epochs_to_ndarray(
    epochs,
    split,
    label_keys,
):
    X = epochs.get_data()
    X = pyriemann.estimation.Covariances(estimator="lwf").transform(X)
    y = rosoku.utils.get_labels_from_epochs(epochs, label_keys)

    return X, y


# %%
# Run the Experiment
# ==================

subject = 10

dataset = Dreyer2023()
label_keys = {"left_hand": 0, "right_hand": 1}

save_base = Path("~").expanduser() / "rosoku-log"

results = rosoku.conventional(
    items_train=[subject, "R1", "R2"],
    items_test=[[subject, "R3", "R4", "R5", "R6"]],
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
        callback_convert_epochs_to_ndarray, label_keys=label_keys
    ),
    scoring=["accuracy", "f1"],
    samples_fname=save_base / "samples.parquet",
    additional_values={"subject": subject},
)

# %%
# Print Results
# =============
print(results.to_string())
