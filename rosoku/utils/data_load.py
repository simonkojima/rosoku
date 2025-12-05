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
    """
    Extract trial labels from MNE Epochs based on event markers.

    This function retrieves event markers from ``epochs.events`` and maps them to
    integer class IDs according to ``label_keys``.
    Event strings containing multiple tags (e.g. ``"cue/left_hand"``) are also
    supported — if a key appears anywhere within the marker, it is assigned.

    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs from which events are extracted.
        ``epochs.events`` and ``epochs.event_id`` must be available.

    label_keys : dict, default={"left_hand": 0, "right_hand": 1}
        Mapping from event keyword to integer class label.
        Keys are matched against event marker strings.
        Example:
        ``{"left_hand": 0, "right_hand": 1, "feet": 2}``

    Returns
    -------
    y : np.ndarray of shape (n_epochs,)
        Array of integer labels corresponding to each epoch.

    Raises
    ------
    RuntimeError
        If the extracted label count does not match the number of epochs.

    Notes
    -----
    - Event markers may contain hierarchical names (e.g. ``"cue/left_hand"``).
      In such cases, the marker is split by ``"/"`` and matched against keys.
    - Every epoch must contain exactly one identifiable label.
    - ``len(epochs) == len(y)`` is enforced for consistency.

    Examples
    --------
    >>> y = get_labels_from_epochs(epochs,
    ...     label_keys={"left_hand": 0, "right_hand": 1, "feet": 2})
    >>> y[:10]
    array([0, 1, 1, 0, ...])
    """
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
        label_keys={"left_hand": 0, "right_hand": 1},
        **kwargs,
):
    """
    Convert an MNE Epochs object into NumPy arrays (X, y).

    This function extracts the raw epoch data using ``epochs.get_data()`` and
    generates corresponding class labels by parsing event markers via
    :func:`get_labels_from_epochs`.
    It serves as the **default conversion function** for both
    :func:`rosoku.deeplearning` and :func:`rosoku.conventional`,
    where it is passed as ``func_convert_epochs_to_ndarray``.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE epochs containing EEG/BCI trials. Must include ``events`` and
        ``event_id`` for label extraction.

    mode : {"train", "valid", "test"}
        Provided for pipeline consistency, but not used directly here.
        Allows seamless substitution with user-defined conversion functions
        that may behave differently depending on dataset split.

    label_keys : dict, default={"left_hand": 0, "right_hand": 1}
        Mapping from event string to class ID.
        Passed to :func:`get_labels_from_epochs`.

    **kwargs :
        Additional arguments forwarded to ``epochs.get_data()``, such as:

        ``picks=...`` (channel selection)
        ``tmin, tmax`` (time window)
        ``reject_by_annotation=True`` etc.

    Returns
    -------
    X : np.ndarray of shape (n_epochs, n_channels, n_times)
        Trial data extracted from MNE Epochs.

    y : np.ndarray of shape (n_epochs,)
        Integer class labels derived from event markers.

    Notes
    -----
    - Serves as the default feature-extraction backend for rosoku pipelines.
    - Intended for cases where the user does **not** need custom handcrafted
      features and prefers raw time-domain input for ML/DL.
    - Can be replaced by any user-defined function with the same return format.

    Examples
    --------
    >>> X, y = convert_epochs_to_ndarray(epochs, mode="train")
    >>> X.shape, y.shape
    ((120, 64, 400), (120,))
    """
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
    Load and preprocess datasets for rosoku pipelines using keyword specifications.

    This utility function is the core data loader used by rosoku's
    :func:`conventional` and :func:`deeplearning` pipelines. It takes
    user-defined *keywords* that describe which data to load for the
    train/validation/test splits, calls user-provided loading functions
    (for MNE Epochs or NumPy arrays), optionally applies preprocessing,
    and returns NumPy arrays suitable for machine-learning models.

    Parameters
    ----------
    keywords_train : list
        List of keyword objects specifying which data belong to the
        training split. The structure of each keyword is arbitrary
        (e.g., dicts with subject/session metadata) and is interpreted
        solely by the user-defined loading functions.

    keywords_valid : list or None
        List of keyword objects specifying validation data.
        If ``None``, no separate validation set is loaded and
        ``X_valid``/``y_valid`` will be ``None``.

    keywords_test : list
        List describing test data and how they are grouped. Each element
        can be either:

        - a single keyword object → treated as one test group
        - a list of keyword objects → merged and evaluated as a single test group

        Examples
        --------
        - ``keywords_test = ["A29", "A3"]``
          → two separate test sets (``["A29"]``, ``["A3"]``)

        - ``keywords_test = [["A29", "A3"]]``
          → load both and merge into one test set

    func_load_epochs : callable, optional
        Function used to load data as MNE ``Epochs`` objects.
        Must accept ``(keywords, mode)`` where ``keywords`` is a list of
        keyword objects and ``mode`` is one of ``"train"``, ``"valid"``,
        or ``"test"``. Should return an ``mne.Epochs`` instance (or a
        merged Epochs object).

        Required if ``func_load_ndarray`` is ``None``.

    func_load_ndarray : callable, optional
        Function used to load data directly as NumPy arrays.
        Must accept ``(keywords, mode)`` and return a tuple ``(X, y)``,
        where ``X`` is a NumPy array and ``y`` the corresponding labels.

        Required if ``func_load_epochs`` is ``None``.

    func_proc_epochs : callable, optional
        Preprocessing function applied to Epochs objects *before*
        conversion to NumPy arrays. It is passed through to
        :func:`apply_func_proc` and can be used for tasks such as
        channel selection, cropping, filtering, etc.

    func_proc_ndarray : callable, optional
        Preprocessing function applied to NumPy arrays *(X, y)* after
        conversion from Epochs or direct loading. Also handled by
        :func:`apply_func_proc`.

    func_proc_mode : {"per_split", "all"}, default="per_split"
        Controls how the preprocessing functions are applied:

        - ``"per_split"`` : process train/valid/test splits independently
        - ``"all"`` : pass all splits at once to the processing function
          (exact behavior depends on :func:`apply_func_proc`)

    func_convert_epochs_to_ndarray : callable, default=convert_epochs_to_ndarray
        Function used to convert Epochs objects to ``(X, y)`` arrays.
        By default, uses :func:`convert_epochs_to_ndarray`, which wraps
        ``epochs.get_data()`` and :func:`get_labels_from_epochs`.

    Returns
    -------
    X_train : np.ndarray
        Training data array.

    X_valid : np.ndarray or None
        Validation data array, or ``None`` if ``keywords_valid`` is ``None``.

    X_test : list of np.ndarray
        List of test data arrays, one per test group defined in
        ``keywords_test``.

    y_train : np.ndarray
        Training labels.

    y_valid : np.ndarray or None
        Validation labels, or ``None`` if ``keywords_valid`` is ``None``.

    y_test : list of np.ndarray
        List of label arrays corresponding to each test group in
        ``X_test``.

    Raises
    ------
    ValueError
        If neither nor both of ``func_load_epochs`` and
        ``func_load_ndarray`` are provided, or if keyword lists are
        not lists as required.

    Notes
    -----
    - Exactly one of ``func_load_epochs`` and ``func_load_ndarray``
      must be provided.
    - When ``func_load_epochs`` is used, the pipeline is:

      .. code-block:: text

          func_load_epochs()
          ↓
          func_proc_epochs()        (optional)
          ↓
          func_convert_epochs_to_ndarray()
          ↓
          func_proc_ndarray()       (optional)

    - When ``func_load_ndarray`` is used, the pipeline is:

      .. code-block:: text

          func_load_ndarray()
          ↓
          func_proc_ndarray()       (optional)

    - This function is designed to support flexible dataset definitions
      while keeping the downstream experiment code (training/evaluation)
      agnostic to data-loading details.

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
