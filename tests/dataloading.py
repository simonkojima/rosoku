# %%
import functools

from pathlib import Path

import mne
import tag_mne as tm

import moabb.datasets

import torch
import braindecode
import rosoku


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


def proc_epochs(epochs, mode, tmin=0.5, tmax=4.5):
    epochs = epochs.pick(picks="eeg").crop(tmin=tmin, tmax=tmax)
    return epochs


def load_epochs(keywords, mode, epochs):
    return epochs[keywords]


def convert_epochs_to_ndarray(epochs):
    print(epochs)


if __name__ == "__main__":

    subject = 56
    dataset = moabb.datasets.Dreyer2023()
    sessions = dataset.get_data(subjects=[subject])
    raws = sessions[subject]["0"]

    resample = 128

    save_base = Path("~").expanduser() / "rosoku-log"
    save_base.mkdir(parents=True, exist_ok=True)

    if False:
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
        epochs.save(
            (save_base / f"dataset-{dataset.code}_sub-{subject}_epochs-epo.fif"),
            overwrite=True,
        )

    epochs = mne.read_epochs(
        save_base / f"dataset-{dataset.code}_sub-{subject}_epochs-epo.fif"
    )
    print(epochs)

    keywords_train = ["run:1/event:left"]
    keywords_valid = [
        "run:1/event:right",
        "run:2",
    ]
    keywords_test = ["run:3", "run:4", "run:5"]

    func_load_epochs = functools.partial(load_epochs, epochs=epochs)
    func_load_ndarray = None
    func_proc_epochs = proc_epochs
    func_proc_ndarray = None
    apply_func_proc_per_obj = True
    func_convert_epochs_to_ndarray = convert_epochs_to_ndarray
    compile_test = False

    """
    X_train, X_valid, X_test, y_train, y_valid, y_test = rosoku.utils.load_data(
        keywords_train=keywords_train,
        keywords_valid=keywords_valid,
        keywords_test=keywords_test,
        func_load_epochs=func_load_epochs,
        func_load_ndarray=func_load_ndarray,
        func_proc_epochs=func_proc_epochs,
        func_proc_ndarray=func_proc_ndarray,
        apply_func_proc_per_obj=apply_func_proc_per_obj,
        func_convert_epochs_to_ndarray=func_convert_epochs_to_ndarray,
        compile_test=compile_test,
    )
    """

    X_train, X_valid, X_test, y_train, y_valid, y_test = rosoku.utils.load_data_2(
        keywords_train=keywords_train,
        keywords_valid=keywords_valid,
        keywords_test=keywords_test,
        func_load_epochs=func_load_epochs,
        func_load_ndarray=func_load_ndarray,
        func_proc_epochs=func_proc_epochs,
        func_proc_ndarray=func_proc_ndarray,
        apply_func_proc_per_obj=apply_func_proc_per_obj,
        func_convert_epochs_to_ndarray=func_convert_epochs_to_ndarray,
        compile_test=compile_test,
    )

    print(X_train.shape)
    print(X_valid.shape)
    print(X_test.shape)
