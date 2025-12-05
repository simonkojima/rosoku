import numpy as np


def get_swap_dict(d):
    return {v: k for k, v in d.items()}


def markers_from_events(events, event_id):
    event_id_swap = get_swap_dict(event_id)

    samples = np.array(events)[:, 0]

    markers = list()
    for val in np.array(events)[:, 2]:
        if "marker:" in str(event_id_swap[val]):
            markers.append(str(event_id_swap[val]))
        else:
            markers.append("marker:%s" % str(event_id_swap[val]))

    return samples, markers


def get_labels_from_epochs(epochs, label_keys={"left_hand": 0, "right_hand": 1}):
    y = list()

    _, markers = markers_from_events(epochs.events, epochs.event_id)

    for marker in markers:
        for key, val in label_keys.items():
            if "/" in marker:
                if key in marker.split("/"):
                    y.append(val)
            else:
                if key in marker:
                    y.append(val)

    if len(epochs) != len(y):
        raise RuntimeError(
            f"lenth of epochs is not match with length of y.\n len(epochs): {len(epochs)}, len(y): {len(y)}"
        )

    return np.array(y)


def apply_func_proc(func_proc, func_proc_mode, train, valid, test):
    match func_proc_mode:
        case "per_split":
            train = func_proc(train, "train")
            if valid is not None:
                valid = func_proc(valid, "valid")
            test = [func_proc(obj, "test") for obj in test]
        case "per_function":
            if valid is None:
                train, test = func_proc(train, test)
            else:
                train, valid, test = func_proc(train, valid, test)

        case _:
            raise ValueError(f"func_proc_mode: {func_proc_mode} is unknown")

    return train, valid, test


def convert_epochs_to_ndarray(
        epochs,
        mode,
        label_keys={"event:left": 0, "event:right": 1},
        **kwargs,
):
    X = epochs.get_data(**kwargs)
    y = get_labels_from_epochs(epochs, label_keys)

    return X, y


def load_data(
        keywords_train,
        keywords_valid,
        keywords_test,
        func_load_epochs=None,
        func_load_ndarray=None,
        func_proc_epochs=None,
        func_proc_ndarray=None,
        func_proc_mode="per_split",
        func_convert_epochs_to_ndarray=convert_epochs_to_ndarray,
):
    """
    Load dataset using given keyword lists and optional processing functions.

    Parameters
    ----------
    keywords_train : list
        A list of identifiers used to load the training data.

    keywords_valid : list or None
        A list of identifiers for validation data.
        If ``None``, data will be split into only train and test.

    keywords_test : list
        A list of identifiers for test data.
        For example:
            ``keywords_test=["A29", "A3"]`` → test each dataset separately  
            ``keywords_test=[["A29", "A3"]]`` → merge them as a single test dataset

    func_load_epochs : callable, optional
        Function used to load *epochs* objects.  
        Must be provided if ``func_load_ndarray`` is ``None``.

    func_load_ndarray : callable, optional
        Function used to load raw NumPy arrays.  
        Must be provided if ``func_load_epochs`` is ``None``.

    func_proc_epochs : callable, optional
        Processing function applied to epochs before conversion to ndarray.

    func_proc_ndarray : callable, optional
        Processing function applied after ndarray conversion.

    func_proc_mode : {'per_split', 'per_function'}, default='per_split'
        - ``'per_split'``: ``func_proc`` is applied to each split independently  
          (**train**, **valid**, **test**)  
        - ``'per_function'``: all splits are passed to ``func_proc`` at once

    func_convert_epochs_to_ndarray : callable, default=rosoku.utils.convert_epochs_to_ndarray
        Function that converts epochs → ``(X, y)`` arrays.

    Notes
    -----
    At least one of ``func_load_epochs`` or ``func_load_ndarray`` must be provided.

    If ``func_load_ndarray is None`` the processing pipeline becomes:

    .. code-block:: text

        func_load_epochs()
        ↓
        func_proc_epochs()
        ↓
        func_convert_epochs_to_ndarray()
        ↓
        func_proc_ndarray()

    If ``func_load_epochs is None`` the pipeline becomes simpler:

    .. code-block:: text

        func_load_ndarray()
        ↓
        func_proc_ndarray()

    Returns
    -------
    X_train : ndarray
    X_valid : ndarray or None
    X_test : list of ndarray
    y_train : ndarray
    y_valid : ndarray or None
    y_test : list of ndarray

    """
    if func_load_epochs is None and func_load_ndarray is None:
        raise ValueError("Specify func_load_epochs or func_load_ndarray")

    if func_load_epochs is not None and func_load_ndarray is not None:
        raise ValueError("Either func_load_epochs or func_load_ndarray must be None")

    if keywords_valid is None:
        if isinstance(keywords_train, list) and isinstance(keywords_test, list):
            pass
        else:
            raise ValueError(
                "keywords_train and keywords_test must be instance of list"
            )
    else:
        if (
                isinstance(keywords_train, list)
                and isinstance(keywords_valid, list)
                and isinstance(keywords_test, list)
        ):
            pass
        else:
            raise ValueError(
                "keywords_train, keywords_valid, and keywords_test must be instance of list"
            )

    if func_load_epochs is not None:
        # load epochs
        epochs_train = func_load_epochs(keywords_train, "train")
        if keywords_valid is None:
            epochs_valid = None
        else:
            epochs_valid = func_load_epochs(keywords_valid, "valid")

        epochs_test = []
        for k in keywords_test:
            if isinstance(k, list):
                e = func_load_epochs(k, "test")
                epochs_test.append(e)
            else:
                e = func_load_epochs([k], "test")
                epochs_test.append(e)

        # apply func_proc_epochs
        if func_proc_epochs is not None:
            (epochs_train, epochs_valid, epochs_test) = apply_func_proc(
                func_proc=func_proc_epochs,
                func_proc_mode=func_proc_mode,
                train=epochs_train,
                valid=epochs_valid,
                test=epochs_test,
            )

        # convert epochs to ndarray
        X_train, y_train = func_convert_epochs_to_ndarray(epochs_train, "train")
        if epochs_valid is None:
            X_valid, y_valid = None, None
        else:
            X_valid, y_valid = func_convert_epochs_to_ndarray(epochs_valid, "valid")
        X_test, y_test = [], []
        for e in epochs_test:
            X, y = func_convert_epochs_to_ndarray(e, "test")
            X_test.append(X)
            y_test.append(y)
    else:
        # load ndarray
        X_train, y_train = func_load_ndarray(keywords_train, "train")

        if keywords_valid is None:
            X_valid, y_valid = None, None
        else:
            X_valid, y_valid = func_load_ndarray(keywords_valid, "valid")

        X_test, y_test = [], []
        for k in keywords_test:
            if isinstance(k, list):
                X, y = func_load_ndarray(k, "test")
                X_test.append(X)
                y_test.append(y)
            else:
                X, y = func_load_ndarray([k], "test")
                X_test.append(X)
                y_test.append(y)

    # proc nd array

    if func_proc_ndarray is not None:
        dict_test = [{"X": X, "y": y} for X, y in zip(X_test, y_test)]
        (train, valid, test) = apply_func_proc(
            func_proc=func_proc_ndarray,
            func_proc_mode=func_proc_mode,
            train={"X": X_train, "y": y_train},
            valid={"X": X_valid, "y": y_valid},
            test=dict_test,
        )

        X_train, y_train = train["X"], train["y"]
        X_valid, y_valid = valid["X"], valid["y"]
        X_test, y_test = [d["X"] for d in test], [d["y"] for d in test]

    return X_train, X_valid, X_test, y_train, y_valid, y_test
