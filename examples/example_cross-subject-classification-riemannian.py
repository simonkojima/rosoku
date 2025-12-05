"""
Example: Cross-subject classification with riemannian classifier
================================================================
"""

# %%
import functools
import numpy as np

import mne
import tag_mne as tm

import moabb.datasets

import pyriemann
import rosoku


# %%
def func_load_ndarray(
        keywords,
        mode,
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

    for keyword in keywords:
        subject = int(keyword[1:])
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

            # Apply Domain Adoptation (Riemannian Alignment)
            X = rosoku.tl.riemannian_alignment(X, scaling=True)

            y = rosoku.utils.get_labels_from_epochs(
                epochs, label_keys=label_keys
            )

            X_list.append(X)
            y_list.append(y)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y


# %%
label_keys = {"left_hand": 0, "right_hand": 1}
dataset = moabb.datasets.Dreyer2023()

results = rosoku.conventional(
    keywords_train=[f"A{num}" for num in range(1, 3)],
    keywords_test=["A21", "A56"],
    func_load_ndarray=functools.partial(
        func_load_ndarray,
        dataset=dataset,
        tmin=dataset.interval[0] + 0.5,
        tmax=dataset.interval[1],
        l_freq=8.0,
        h_freq=30.0,
        order_filter=4,
        label_keys=label_keys,
    ),
)

for m in range(results.shape[0]):
    print(results.loc[m])
