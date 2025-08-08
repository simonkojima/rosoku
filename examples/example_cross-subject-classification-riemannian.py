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

        epochs_subject = epochs_subject.crop(tmin=tmin, tmax=tmax).pick(picks="eeg")

        y_subject = rosoku.utils.get_labels_from_epochs(
            epochs_subject, label_keys=label_keys
        )

        X_subject = pyriemann.estimation.Covariances().transform(
            epochs_subject.get_data()
        )

        X_subject = rosoku.tl.riemannian_alignment(X_subject, scaling=True)

        y.append(y_subject)
        X.append(X_subject)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    return X, y


# %%


def convert_epochs_to_ndarray(
    epochs_train,
    epochs_test,
    label_keys,
):

    X_train = epochs_train.get_data()
    X_test = epochs_test.get_data()

    X_train = pyriemann.estimation.Covariances().transform(X_train)
    X_test = pyriemann.estimation.Covariances().transform(X_test)

    y_train = rosoku.utils.get_labels_from_epochs(epochs_train, label_keys)
    y_test = rosoku.utils.get_labels_from_epochs(epochs_test, label_keys)

    return X_train, X_test, y_train, y_test


# %%
label_keys = {"event:left": 0, "event:right": 1}

results = rosoku.conventional(
    keywords_train=[f"A{num}" for num in range(1, 3)],
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
)

for m in range(results.shape[0]):
    print(results.loc[m])
