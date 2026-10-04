"""
Example 03: Cross-subject classification with Riemannian classifier
===================================================================

This example demonstrates **cross-subject motor imagery classification**
using a **Riemannian geometry-based pipeline** with ``rosoku``.

We use the **Dreyer2023** dataset and train a classifier on data pooled from
multiple subjects, then evaluate it on **unseen subjects**. In contrast to
within-subject decoding, cross-subject classification is more challenging
because EEG distributions vary substantially across users.

Key aspects illustrated in this example include:

- Loading and preprocessing EEG data from **multiple subjects** (MNE-based)
- Feature extraction using **SPD covariance matrices** (pyRiemann)
- Optional transfer-learning step via **Riemannian alignment**
  (domain adaptation across subjects)
- A callback that returns NumPy arrays directly (``callback_load_ndarray``),
  which is convenient for cross-subject pipelines

The pipeline consists of the following steps:

1. Load raw EEG recordings for each subject and apply band-pass filtering
2. Epoch the data and compute covariance matrices per trial
3. (Optional) Apply Riemannian alignment to reduce inter-subject distribution shifts
4. Concatenate trials across training subjects and fit a Riemannian classifier
5. Evaluate the trained model on held-out subjects

This example is intended as a simple, interpretable template for building
cross-subject Riemannian decoding pipelines with ``rosoku``.
"""

# Authors: Simon Kojima <simon.kojima@inria.fr>
#
# License: BSD (3-clause)

# %%
# Import Packages
# ===============
import functools
import numpy as np
import mne
from moabb.datasets import Dreyer2023
import pyriemann
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

            X = pyriemann.estimation.Covariances(estimator="lwf").transform(
                epochs.get_data()
            )

            # Apply Domain Adaptation (Riemannian Alignment)
            X = rosoku.tl.riemannian_alignment(X, scaling=True)

            y = rosoku.utils.get_labels_from_epochs(epochs, label_keys=label_keys)

            X_list.append(X)
            y_list.append(y)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y


# %%
# Run the Experiment
# ==================

label_keys = {"left_hand": 0, "right_hand": 1}
dataset = Dreyer2023()

results = rosoku.conventional(
    items_train=list(range(1, 21)),
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
    additional_values={"example_key": "example_value"},
)

# %%
# Print Results
# =============

print(results.to_string())
