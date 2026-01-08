import random
import numpy as np
import torch


def get_swap_dict(d):
    return {v: k for k, v in d.items()}


def markers_from_events(events, event_id):
    event_id_swap = get_swap_dict(event_id)

    samples = np.array(events)[:, 0]

    markers = list()
    for val in np.array(events)[:, 2]:
        markers.append(str(event_id_swap[val]))

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
                if key == marker:
                    y.append(val)

    if len(epochs) != len(y):
        raise RuntimeError(
            f"lenth of epochs is not match with length of y.\n len(epochs): {len(epochs)}, len(y): {len(y)}"
        )

    return np.array(y)


def apply_callback_proc(callback_proc, callback_proc_mode, train, valid, test):
    """
    Apply a preprocessing callback function to dataset splits.

    This function supports two processing modes:

    - ``"per_split"``: The callback function is applied independently to
      each data split (train / valid / test). The callback receives a
      single split and its corresponding split name.

    - ``"joint"``: The callback function is applied jointly to multiple
      splits at once. The callback receives all available splits together
      and may share state across them (e.g., fitting on training data and
      applying the transformation to validation and test data).

    Parameters
    ----------
    callback_proc : callable
        Preprocessing callback function.

        - If ``callback_proc_mode="per_split"``, the signature must be::

              callback_proc(data, split)

          where ``split`` is one of ``{"train", "valid", "test"}``.

        - If ``callback_proc_mode="joint"``, the signature must be either::

              callback_proc(train, test)
              callback_proc(train, valid, test)

          depending on whether validation data are provided.

    callback_proc_mode : {"per_split", "joint"}
        Mode that specifies how the preprocessing callback is applied.

        - ``"per_split"`` applies the callback independently to each split.
        - ``"joint"`` applies the callback jointly to multiple splits.

    train : object
        Training split data.

    valid : object or None
        Validation split data. If ``None``, validation data are assumed to
        be unavailable and are skipped.

    test : list
        Test split data. Each element is processed independently in
        ``"per_split"`` mode, or passed collectively in ``"joint"`` mode.

    Returns
    -------
    train : object
        Processed training split.

    valid : object or None
        Processed validation split.

    test : list
        Processed test split(s).

    Raises
    ------
    ValueError
        If ``callback_proc_mode`` is not one of the supported modes.

    Notes
    -----
    The ``"per_split"`` mode is suitable for stateless or split-local
    preprocessing steps such as filtering, normalization, or feature
    extraction.

    The ``"joint"`` mode is intended for stateful preprocessing that
    requires access to multiple splits simultaneously, such as fitting
    a transformation on training data and applying it consistently to
    validation and test data.

    Examples
    --------
    Apply split-wise preprocessing::

        train, valid, test = apply_callback_proc(
            callback_proc=my_proc,
            callback_proc_mode="per_split",
            train=train,
            valid=valid,
            test=test,
        )

    Apply joint preprocessing (e.g., fit on train, apply to others)::

        train, valid, test = apply_callback_proc(
            callback_proc=my_proc,
            callback_proc_mode="joint",
            train=train,
            valid=valid,
            test=test,
        )
    """
    match callback_proc_mode:
        case "per_split":
            train = callback_proc(train, "train")
            if valid is not None:
                valid = callback_proc(valid, "valid")
            test = [callback_proc(obj, "test") for obj in test]
        case "joint":
            if valid is None:
                train, test = callback_proc(train, test)
            else:
                train, valid, test = callback_proc(train, valid, test)

        case _:
            raise ValueError(f"callback_proc_mode: {callback_proc_mode} is unknown")

    return train, valid, test


def convert_epochs_to_ndarray(
        epochs,
        split,
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

    split : {"train", "valid", "test"}
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
    >>> X, y = convert_epochs_to_ndarray(epochs, split="train")
    >>> X.shape, y.shape
    ((120, 64, 400), (120,))
    """
    X = epochs.get_data(**kwargs)
    y = get_labels_from_epochs(epochs, label_keys)

    return X, y


def ndarray_to_tensor(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        device="cpu",
        dtype=torch.float32,
):
    """
    Convert NumPy arrays to PyTorch tensors and move them to a device.

    This utility function converts training, validation, and test datasets from
    NumPy arrays to PyTorch tensors with appropriate dtypes, and moves them to
    the specified device. It supports both single test sets and grouped test
    sets represented as lists, which is consistent with rosoku's handling of
    multiple test evaluation groups.

    Parameters
    ----------
    X_train : array-like
        Training data array. Typically has shape
        ``(n_samples, n_channels, n_times)`` for EEG data.

    y_train : array-like
        Training labels. Must be convertible to integer class indices.

    X_valid : array-like
        Validation data array with the same structure as ``X_train``.

    y_valid : array-like
        Validation labels.

    X_test : array-like or list of array-like
        Test data array, or a list of test data arrays corresponding to multiple
        test evaluation groups.

    y_test : array-like or list of array-like
        Test labels, or a list of label arrays corresponding to ``X_test``.

    device : {"cpu", "cuda"}, optional
        Device to which the tensors are moved (default: ``"cpu"``).

    Returns
    -------
    X_train_tensor : torch.Tensor
        Training data tensor of type ``torch.float`` on ``device``.

    y_train_tensor : torch.Tensor
        Training label tensor of type ``torch.int64`` on ``device``.

    X_valid_tensor : torch.Tensor
        Validation data tensor of type ``torch.float`` on ``device``.

    y_valid_tensor : torch.Tensor
        Validation label tensor of type ``torch.int64`` on ``device``.

    X_test_tensor : torch.Tensor or list of torch.Tensor
        Test data tensor(s) of type ``torch.float`` on ``device``.
        A list is returned if ``X_test`` is a list.

    y_test_tensor : torch.Tensor or list of torch.Tensor
        Test label tensor(s) of type ``torch.int64`` on ``device``.
        A list is returned if ``y_test`` is a list.

    dtype : torch.dtype, optional
        Data type used when converting NumPy arrays to PyTorch tensors
        (default: ``torch.float32``).

        This argument is forwarded to :func:`ndarray_to_tensor` and controls the
        floating-point precision of the resulting tensors stored in the
        :class:`torch.utils.data.TensorDataset`.

        Typical choices are:

        - ``torch.float32`` : default and recommended for most training scenarios
        - ``torch.float64`` : useful for numerical debugging or high-precision analysis
        - ``torch.float16`` / ``torch.bfloat16`` : may be used for memory reduction,
          but require careful handling and compatible models/devices

        Note:
         - ``dtype`` affects only the feature tensors ``X``. Label tensors ``y`` are
           converted to integer types as required by PyTorch loss functions.
         - Changing ``dtype`` can affect numerical stability and reproducibility,
           especially when combined with GPU acceleration.

    Notes
    -----
    - All feature arrays are converted to ``torch.float`` and all labels to
      ``torch.int64``, which is compatible with common PyTorch loss functions
      such as :class:`torch.nn.CrossEntropyLoss`.
    - When ``X_test`` and ``y_test`` are lists, the returned test tensors preserve
      this list structure, enabling grouped test evaluation in downstream
      pipelines (e.g., in :func:`rosoku.deeplearning`).

    See Also
    --------
    rosoku.deeplearning :
        Deep-learning pipeline that consumes tensors produced by this function.

    Examples
    --------
    Convert NumPy arrays to tensors on GPU::

        Xtr_t, ytr_t, Xva_t, yva_t, Xte_t, yte_t = ndarray_to_tensor(
            X_train, y_train,
            X_valid, y_valid,
            X_test, y_test,
            device="cuda",
        )
    """
    import torch

    X_train_tensor = torch.tensor(X_train, dtype=dtype).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64).to(device)

    if X_valid is not None:
        X_valid_tensor = torch.tensor(X_valid, dtype=dtype).to(device)
        y_valid_tensor = torch.tensor(y_valid, dtype=torch.int64).to(device)
    else:
        X_valid_tensor = y_valid_tensor = None

    if isinstance(X_test, list):
        X_test_tensor = [torch.tensor(X, dtype=dtype).to(device) for X in X_test]
        y_test_tensor = [torch.tensor(y, dtype=torch.int64).to(device) for y in y_test]
    else:
        X_test_tensor = torch.tensor(X_test, dtype=dtype).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.int64).to(device)

    return (
        X_train_tensor,
        y_train_tensor,
        X_valid_tensor,
        y_valid_tensor,
        X_test_tensor,
        y_test_tensor,
    )


def tensor_to_dataset(
        X_train_tensor,
        y_train_tensor,
        X_valid_tensor,
        y_valid_tensor,
        X_test_tensor,
        y_test_tensor,
):
    """
    Convert PyTorch tensors into TensorDataset objects.

    This utility function wraps training, validation, and test tensors into
    :class:`torch.utils.data.TensorDataset` objects. It supports both single
    test datasets and grouped test datasets represented as lists, which is
    consistent with rosoku's handling of multiple test evaluation groups.

    Parameters
    ----------
    X_train_tensor : torch.Tensor
        Training data tensor.

    y_train_tensor : torch.Tensor
        Training label tensor.

    X_valid_tensor : torch.Tensor
        Validation data tensor.

    y_valid_tensor : torch.Tensor
        Validation label tensor.

    X_test_tensor : torch.Tensor or list of torch.Tensor
        Test data tensor, or a list of test data tensors corresponding to
        multiple test evaluation groups.

    y_test_tensor : torch.Tensor or list of torch.Tensor
        Test label tensor, or a list of label tensors corresponding to
        ``X_test_tensor``.

    Returns
    -------
    dataset_train : torch.utils.data.TensorDataset
        Dataset containing training data and labels.

    dataset_valid : torch.utils.data.TensorDataset
        Dataset containing validation data and labels.

    dataset_test : torch.utils.data.TensorDataset or list of torch.utils.data.TensorDataset
        Test dataset(s). A list is returned if ``X_test_tensor`` is a list,
        preserving the grouping of test evaluation sets.

    Notes
    -----
    - Each returned dataset yields ``(data, label)`` pairs when iterated.
    - When ``X_test_tensor`` is a list, the list structure is preserved in the
      returned ``dataset_test`` to enable grouped test evaluation in downstream
      pipelines (e.g., in :func:`rosoku.deeplearning`).

    See Also
    --------
    ndarray_to_tensor :
        Convert NumPy arrays to PyTorch tensors.

    torch.utils.data.TensorDataset :
        PyTorch dataset class used to wrap tensors.

    Examples
    --------
    Convert tensors to datasets::

        dataset_train, dataset_valid, dataset_test = tensor_to_dataset(
            X_train_tensor, y_train_tensor,
            X_valid_tensor, y_valid_tensor,
            X_test_tensor, y_test_tensor,
        )

    Use grouped test datasets::

        for ds in dataset_test:
            loader = DataLoader(ds, batch_size=64)
            ...
    """
    import torch

    dataset_train = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)

    if X_valid_tensor is not None:
        dataset_valid = torch.utils.data.TensorDataset(X_valid_tensor, y_valid_tensor)
    else:
        dataset_valid = None

    if isinstance(X_test_tensor, list):
        dataset_test = [
            torch.utils.data.TensorDataset(X, y)
            for X, y in zip(X_test_tensor, y_test_tensor)
        ]
    else:
        dataset_test = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

    return dataset_train, dataset_valid, dataset_test


def dataset_to_dataloader(
        dataset_train,
        dataset_valid,
        dataset_test,
        batch_size,
        num_workers=0,
        seed=None,
        generator=None,
):
    if (num_workers > 0) and (seed is not None or generator is not None):

        def func_worker_init(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

    else:
        func_worker_init = None

    if (generator is None) and (seed is not None):
        generator = torch.Generator()
        generator.manual_seed(seed)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=func_worker_init,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    def make_non_shuffle_dl(ds):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            generator=generator,
            worker_init_fn=func_worker_init,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

    if dataset_valid is not None:
        dataloader_valid = make_non_shuffle_dl(dataset_valid)
    else:
        dataloader_valid = None

    if isinstance(dataset_test, list):
        dataloader_test = [make_non_shuffle_dl(dataset) for dataset in dataset_test]
    else:
        dataloader_test = make_non_shuffle_dl(dataset_test)

    return dataloader_train, dataloader_valid, dataloader_test


def ndarray_to_dataloader(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        batch_size,
        device="cpu",
        dtype=torch.float32,
        num_workers=0,
        seed=None,
        generator=None,
):
    """
    Convert NumPy arrays to PyTorch DataLoaders.

    This is a convenience wrapper that converts NumPy arrays to PyTorch tensors,
    wraps them into :class:`torch.utils.data.TensorDataset` objects, and constructs
    :class:`torch.utils.data.DataLoader` instances for training, validation, and
    test splits.

    Grouped test evaluation is supported: if ``X_test`` / ``y_test`` are provided
    as lists, the returned test DataLoaders preserve the list structure, enabling
    evaluation of multiple test groups.

    Parameters
    ----------
    X_train : array-like
        Training data array. Typically has shape
        ``(n_samples, n_channels, n_times)`` for EEG data.

    y_train : array-like
        Training labels. Must be convertible to integer class indices.

    X_valid : array-like
        Validation data array with the same structure as ``X_train``.

    y_valid : array-like
        Validation labels.

    X_test : array-like or list of array-like
        Test data array, or a list of test data arrays corresponding to multiple
        test evaluation groups.

    y_test : array-like or list of array-like
        Test labels, or a list of label arrays corresponding to ``X_test``.

    batch_size : int
        Mini-batch size used for all DataLoaders.

    device : {"cpu", "cuda"}, optional
        Device to which the tensors are moved before dataset/DataLoader creation
        (default: ``"cpu"``).

    dtype : torch.dtype, optional
        Data type used when converting NumPy arrays to PyTorch tensors
        (default: ``torch.float32``).

        This argument is forwarded to :func:`ndarray_to_tensor` and controls the
        floating-point precision of the resulting tensors stored in the
        :class:`torch.utils.data.TensorDataset`.

        Typical choices are:

        - ``torch.float32`` : default and recommended for most training scenarios
        - ``torch.float64`` : useful for numerical debugging or high-precision analysis
        - ``torch.float16`` / ``torch.bfloat16`` : may be used for memory reduction,
          but require careful handling and compatible models/devices

        Note:
         - ``dtype`` affects only the feature tensors ``X``. Label tensors ``y`` are
           converted to integer types as required by PyTorch loss functions.
         - Changing ``dtype`` can affect numerical stability and reproducibility,
           especially when combined with GPU acceleration.

    num_workers : int, optional
        Number of worker processes used by each DataLoader (default: 0).

        For **maximum reproducibility**, it is recommended to keep
        ``num_workers=0``, as multi-process data loading can introduce
        non-determinism depending on the dataset, transformations, and system
        configuration.

    seed : int or None, optional
        Random seed used to control deterministic behavior in data loading.
        This value is forwarded to ``dataset_to_dataloader`` and used to seed
        PyTorch, NumPy, and Python RNGs where applicable.

    generator : None | int | torch.Generator, optional
        Random generator controlling the DataLoader sampling order.

        - If ``None``, no explicit generator is used.
        - If ``int``, a new :class:`torch.Generator` is created and seeded.
        - If :class:`torch.Generator`, it is used directly.

    Returns
    -------
    dataloader_train : torch.utils.data.DataLoader
        DataLoader for the training dataset.

    dataloader_valid : torch.utils.data.DataLoader
        DataLoader for the validation dataset.

    dataloader_test : torch.utils.data.DataLoader or list of torch.utils.data.DataLoader
        DataLoader(s) for the test dataset(s). A list is returned if ``X_test`` is
        a list.

    See Also
    --------
    ndarray_to_tensor :
        Convert NumPy arrays to PyTorch tensors.

    tensor_to_dataset :
        Wrap tensors into TensorDataset objects.

    dataset_to_dataloader :
        Construct DataLoaders with explicit control over seeding and workers.

    Notes
    -----
    - This function is a thin wrapper around
      ``ndarray_to_tensor → tensor_to_dataset → dataset_to_dataloader``.
    - For strict reproducibility (e.g., debugging, method comparison, or paper
      experiments), it is recommended to use:

      - ``num_workers = 0``
      - a fixed ``seed``

    - Using ``num_workers > 0`` can improve throughput but may introduce subtle
      non-determinism even when seeds are set.

    Examples
    --------
    Standard (reproducible) usage::

        dl_train, dl_valid, dl_test = ndarray_to_dataloader(
            X_train, y_train,
            X_valid, y_valid,
            X_test, y_test,
            batch_size=64,
            num_workers=0,
            seed=42,
        )

    Grouped test evaluation::

        dl_train, dl_valid, dl_test = ndarray_to_dataloader(
            X_train, y_train,
            X_valid, y_valid,
            [X_test_ses1, X_test_ses2],
            [y_test_ses1, y_test_ses2],
            batch_size=64,
        )
    """
    (
        X_train_tensor,
        y_train_tensor,
        X_valid_tensor,
        y_valid_tensor,
        X_test_tensor,
        y_test_tensor,
    ) = ndarray_to_tensor(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        device=device,
        dtype=dtype,
    )

    (dataset_train, dataset_valid, dataset_test) = tensor_to_dataset(
        X_train_tensor,
        y_train_tensor,
        X_valid_tensor,
        y_valid_tensor,
        X_test_tensor,
        y_test_tensor,
    )

    return dataset_to_dataloader(
        dataset_train,
        dataset_valid,
        dataset_test,
        num_workers=num_workers,
        batch_size=batch_size,
        seed=seed,
        generator=generator,
    )


def load_data(
        items_train,
        items_valid,
        items_test,
        callback_load_epochs=None,
        callback_load_ndarray=None,
        callback_proc_epochs=None,
        callback_proc_ndarray=None,
        callback_proc_mode="per_split",
        callback_convert_epochs_to_ndarray=convert_epochs_to_ndarray,
):
    """
    Load and preprocess datasets for rosoku pipelines using item specifications.

    This utility is the core data loader used by rosoku's :func:`conventional` and
    :func:`deeplearning` pipelines. It takes user-defined *items* that describe which
    data to load for the train/validation/test splits, calls user-provided loading
    functions (for MNE Epochs or NumPy arrays), optionally applies preprocessing, and
    returns NumPy arrays suitable for downstream models.

    In rosoku, an *item* is an arbitrary user-defined object (often a dict) that
    encodes how to locate/load a subset of data (e.g., subject/session/run metadata).
    Items are interpreted only by the user callbacks.

    Parameters
    ----------
    items_train : list
        List of items describing the training split.

    items_valid : list or None
        List of items describing the validation split. If ``None``, no validation set
        is loaded and ``X_valid``/``y_valid`` are returned as ``None``.

    items_test : list
        Test split specification with optional grouping. Each element defines **one**
        test evaluation group and can be either:

        - a single item (e.g., ``"A29"`` or ``{"sub": 1, "ses": 2}``)
        - a list of items (e.g., ``["A29", "A3"]``) to be loaded/merged as a single group

        Note:
        Internally, any single item in ``items_test`` is wrapped to a list
        ``[item]`` before calling the loading callback, so the loader always receives
        a list of items.

    callback_load_epochs : callable | None, optional
        Loader returning an :class:`mne.Epochs` instance.

        The callback **must** have the signature::

            callback_load_epochs(items, split)

        where ``items`` is a list of items and ``split`` is one of
        ``{"train", "valid", "test"}``.

        The callback should implement any merging/concatenation logic needed for
        grouped items (e.g., using :func:`mne.concatenate_epochs`).

        Exactly one of ``callback_load_epochs`` and ``callback_load_ndarray`` must be
        provided.

    callback_load_ndarray : callable | None, optional
        Loader returning arrays directly.

        The callback **must** have the signature::

            callback_load_ndarray(items, split)

        and return ``(X, y)``, where ``X`` is an array-like feature tensor and ``y``
        are the corresponding labels.

        Exactly one of ``callback_load_epochs`` and ``callback_load_ndarray`` must be
        provided.

    callback_proc_epochs : callable | None, optional
        Optional preprocessing applied to Epochs objects *before* conversion to arrays.
        It is invoked via ``apply_callback_proc``. Typical use cases include channel
        selection, cropping, filtering, artifact rejection, etc.

    callback_proc_ndarray : callable | None, optional
        Optional preprocessing applied to ndarray data after conversion (or direct
        ndarray loading). It is invoked via ``apply_callback_proc``.

    callback_proc_mode : {"per_split", "joint"}, optional
        Controls how preprocessing callbacks are applied, as interpreted by
        ``apply_callback_proc``.

        - ``"per_split"``:
            The preprocessing callback is applied independently to each data
            split (train / valid / test).

            The callback function must have the signature::

                callback_proc(data, split)

            where ``split`` is one of ``{"train", "valid", "test"}``.

            The callback must return the processed version of ``data``.
            No information is shared across splits in this mode.

        - ``"joint"``:
            The preprocessing callback is applied jointly to multiple splits
            at once, allowing shared state across splits.

            If validation data are not provided (``valid is None``), the callback
            must have the signature::

                callback_proc(train, test)

            and must return::

                train, test

            If validation data are provided, the callback must have the signature::

                callback_proc(train, valid, test)

            and must return::

                train, valid, test

            This mode is intended for stateful preprocessing steps such as
            fitting a transformation on training data and applying it
            consistently to validation and test data.

    callback_convert_epochs_to_ndarray : callable, optional
        Function used to convert Epochs objects to arrays.
        Called as ``callback_convert_epochs_to_ndarray(epochs, split)``.
        By default, uses :func:`convert_epochs_to_ndarray`.

    Returns
    -------
    X_train : numpy.ndarray
        Training data array.

    X_valid : numpy.ndarray or None
        Validation data array, or ``None`` if ``items_valid`` is ``None``.

    X_test : list of numpy.ndarray
        List of test data arrays, one per test evaluation group.

    y_train : numpy.ndarray
        Training labels.

    y_valid : numpy.ndarray or None
        Validation labels, or ``None`` if ``items_valid`` is ``None``.

    y_test : list of numpy.ndarray
        List of label arrays, one per test evaluation group.

    Raises
    ------
    ValueError
        If neither or both of ``callback_load_epochs`` and ``callback_load_ndarray`` are
        provided.
    ValueError
        If ``items_train`` / ``items_valid`` / ``items_test`` are not lists as required.

    Notes
    -----
    - Exactly one of ``callback_load_epochs`` and ``callback_load_ndarray`` must be
      provided.

    - When ``callback_load_epochs`` is used, the pipeline is::

          callback_load_epochs(items_*, split)
          -> callback_proc_epochs(...)             (optional, via apply_callback_proc)
          -> callback_convert_epochs_to_ndarray(..., split)
          -> callback_proc_ndarray(...)            (optional, via apply_callback_proc)

    - When ``callback_load_ndarray`` is used, the pipeline is::

          callback_load_ndarray(items_*, split)
          -> callback_proc_ndarray(...)            (optional, via apply_callback_proc)

    Examples
    --------
    Load from ndarrays (single-item groups)::

        def load_xy(items, split):
            item = items[0]
            X = np.load(item["X"])
            y = np.load(item["y"])
            return X, y

        X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(
            items_train=[{"X": "Xtr.npy", "y": "ytr.npy"}],
            items_valid=None,
            items_test=[{"X": "Xte.npy", "y": "yte.npy"}],
            callback_load_ndarray=load_xy,
        )

    Group multiple items into a single merged test set::

        def load_epochs(items, split):
            epochs_list = [mne.read_epochs(it["fname"]) for it in items]
            return mne.concatenate_epochs(epochs_list)

        X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(
            items_train=[{"fname": "sub-01_ses-01-epo.fif"}],
            items_valid=None,
            items_test=[[{"fname": "sub-01_ses-02-epo.fif"},
                         {"fname": "sub-01_ses-03-epo.fif"}]],
            callback_load_epochs=load_epochs,
        )
    """
    if callback_load_epochs is None and callback_load_ndarray is None:
        raise ValueError("Specify callback_load_epochs or callback_load_ndarray")

    if callback_load_epochs is not None and callback_load_ndarray is not None:
        raise ValueError(
            "Either callback_load_epochs or callback_load_ndarray must be None"
        )

    if items_valid is None:
        if isinstance(items_train, list) and isinstance(items_test, list):
            pass
        else:
            raise ValueError("items_train and items_test must be instance of list")
    else:
        if (
                isinstance(items_train, list)
                and isinstance(items_valid, list)
                and isinstance(items_test, list)
        ):
            pass
        else:
            raise ValueError(
                "items_train, items_valid, and items_test must be instance of list"
            )

    if callback_load_epochs is not None:
        # load epochs
        epochs_train = callback_load_epochs(items_train, "train")
        if items_valid is None:
            epochs_valid = None
        else:
            epochs_valid = callback_load_epochs(items_valid, "valid")

        epochs_test = []
        for item in items_test:
            if isinstance(item, list):
                e = callback_load_epochs(item, "test")
                epochs_test.append(e)
            else:
                e = callback_load_epochs([item], "test")
                epochs_test.append(e)

        # apply callback_proc_epochs
        if callback_proc_epochs is not None:
            (epochs_train, epochs_valid, epochs_test) = apply_callback_proc(
                callback_proc=callback_proc_epochs,
                callback_proc_mode=callback_proc_mode,
                train=epochs_train,
                valid=epochs_valid,
                test=epochs_test,
            )

        # convert epochs to ndarray
        X_train, y_train = callback_convert_epochs_to_ndarray(epochs_train, "train")
        if epochs_valid is None:
            X_valid, y_valid = None, None
        else:
            X_valid, y_valid = callback_convert_epochs_to_ndarray(epochs_valid, "valid")
        X_test, y_test = [], []
        for e in epochs_test:
            X, y = callback_convert_epochs_to_ndarray(e, "test")
            X_test.append(X)
            y_test.append(y)
    else:
        # load ndarray
        X_train, y_train = callback_load_ndarray(items_train, "train")

        if items_valid is None:
            X_valid, y_valid = None, None
        else:
            X_valid, y_valid = callback_load_ndarray(items_valid, "valid")

        X_test, y_test = [], []
        for item in items_test:
            if isinstance(item, list):
                X, y = callback_load_ndarray(item, "test")
                X_test.append(X)
                y_test.append(y)
            else:
                X, y = callback_load_ndarray([item], "test")
                X_test.append(X)
                y_test.append(y)

    # proc nd array

    if callback_proc_ndarray is not None:
        dict_test = [{"X": X, "y": y} for X, y in zip(X_test, y_test)]
        (train, valid, test) = apply_callback_proc(
            callback_proc=callback_proc_ndarray,
            callback_proc_mode=callback_proc_mode,
            train={"X": X_train, "y": y_train},
            valid={"X": X_valid, "y": y_valid},
            test=dict_test,
        )

        X_train, y_train = train["X"], train["y"]
        X_valid, y_valid = valid["X"], valid["y"]
        X_test, y_test = [d["X"] for d in test], [d["y"] for d in test]

    return X_train, X_valid, X_test, y_train, y_valid, y_test
