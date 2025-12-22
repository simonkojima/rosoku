import time
import random
import json
import msgpack

import numpy as np

import torch

import pandas as pd

import rosoku.preprocessing
from . import utils
from . import preprocessing
from . import attribution

from .utils.core import _train_epoch, _add_values_to_df


def setup_optimizer(optimizer, optimizer_params, model):
    if optimizer_params is not None:
        optimizer = optimizer(
            params=model.parameters(),
            **optimizer_params,
        )
    else:
        optimizer = optimizer(params=model.parameters())

    return optimizer


def setup_scheduler(scheduler, scheduler_params, optimizer):
    if scheduler is not None:
        if scheduler_params is not None:
            scheduler = scheduler(
                optimizer=optimizer,
                **scheduler_params,
            )
        else:
            scheduler = scheduler(optimizer=optimizer)

    return scheduler


def deeplearning_train(
    dataloader_train,
    dataloader_valid,
    n_epochs,
    model,
    criterion,
    device,
    optimizer=None,
    scheduler=None,
    early_stopping=None,
    callback_early_stopping=None,
    enable_wandb_logging=False,
    wandb_params=None,
    checkpoint_fname=None,
    history_fname=None,
    min_delta=0,
):
    if enable_wandb_logging:
        import wandb

    if (callback_early_stopping is not None) and (early_stopping is not None):
        raise ValueError(
            "Cannot specify both callback_early_stopping and early_stopping"
        )

    if early_stopping is not None:
        early_stopping.initialize()

    history = {
        "epoch": list(),
        "train_loss": list(),
        "valid_loss": list(),
        "train_acc": list(),
        "valid_acc": list(),
    }

    loss_best = {"value": float("inf")}

    if enable_wandb_logging:
        wandb.init(**wandb_params)

    tic = time.time()
    for epoch in range(n_epochs):
        train_loss, valid_loss = _train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            min_delta=min_delta,
            dataloader_train=dataloader_train,
            dataloader_valid=dataloader_valid,
            epoch=epoch,
            device=device,
            loss_best=loss_best,
            history=history,
            checkpoint_fname=checkpoint_fname,
            enable_wandb=enable_wandb_logging,
        )

        if early_stopping is not None:
            if early_stopping(valid_loss):
                print(f"Early stopping was triggered: epoch #{epoch + 1}")
                break

        if callback_early_stopping is not None:
            if callback_early_stopping(
                {"train_loss": train_loss, "valid_loss": valid_loss, "epoch": epoch + 1}
            ):
                print(f"Early stopping was triggered: epoch #{epoch + 1}")
                break

    toc = time.time()
    elapsed_time = toc - tic
    print(f"Elapsed Time: {elapsed_time:.2f}s")

    if history_fname is not None:
        df_save = pd.DataFrame(history)
        df_save.to_parquet(history_fname)

    return model


def run_experiment(
    num_workers,
    device,
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test,
    criterion,
    batch_size,
    n_epochs,
    optimizer,
    kwargs,
):
    import torch

    # parse kwargs

    optimizer_params = kwargs.get("optimizer_params", None)
    model = kwargs.get("model", None)
    callback_early_stopping = kwargs.get("callback_early_stopping", None)
    scheduler = kwargs.get("scheduler", None)
    scheduler_params = kwargs.get("scheduler_params", None)
    min_delta = kwargs.get("min_delta", 0)
    enable_wandb_logging = kwargs.get("enable_wandb_logging", False)
    wandb_params = kwargs.get("wandb_params", None)
    checkpoint_fname = kwargs.get("checkpoint_fname", None)
    history_fname = kwargs.get("history_fname", None)
    early_stopping = kwargs.get("early_stopping", None)
    dtype = kwargs.get("dtype", torch.float32)
    seed = kwargs.get("seed", None)

    if model is None:
        raise RuntimeError("model is None")

    # create dataloader

    (dataloader_train, dataloader_valid, _) = utils.ndarray_to_dataloader(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        device="cpu",
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        generator=None,
        dtype=dtype,
    )

    # setup optimizer
    optimizer = setup_optimizer(optimizer, optimizer_params, model)

    # setup scheduler
    scheduler = setup_scheduler(scheduler, scheduler_params, optimizer)

    # setup early stopping
    if isinstance(early_stopping, int):
        early_stopping = utils.EarlyStopping(
            patience=early_stopping, min_delta=min_delta
        )

    model = deeplearning_train(
        dataloader_train=dataloader_train,
        dataloader_valid=dataloader_valid,
        n_epochs=n_epochs,
        model=model,
        criterion=criterion,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        enable_wandb_logging=enable_wandb_logging,
        wandb_params=wandb_params,
        checkpoint_fname=checkpoint_fname,
        history_fname=history_fname,
        early_stopping=early_stopping,
        callback_early_stopping=callback_early_stopping,
        min_delta=min_delta,
    )


def deeplearning(
    items_train,
    items_valid,
    items_test,
    callback_load_epochs=None,
    callback_load_ndarray=None,
    criterion=torch.nn.CrossEntropyLoss(),
    batch_size=64,
    n_epochs=500,
    optimizer=torch.optim.AdamW,
    *,
    callback_proc_mode="per_split",
    callback_proc_epochs=None,
    callback_proc_ndarray=None,
    callback_convert_epochs_to_ndarray=utils.convert_epochs_to_ndarray,
    callback_get_logits=None,
    callback_get_preds=None,
    callback_get_probas=None,
    callback_get_model=None,
    callback_early_stopping=None,
    callback_normalization=rosoku.preprocessing.normalize,
    optimizer_params=None,
    model=None,
    scheduler=None,
    scheduler_params=None,
    min_delta=0,
    device="cpu",
    num_workers=0,
    scoring="accuracy",
    scoring_name=None,
    enable_wandb_logging=False,
    wandb_params=None,
    checkpoint_fname=None,
    history_fname=None,
    samples_fname=None,
    normalization_fname=None,
    saliency_map_fname=False,
    early_stopping=None,
    model_name=None,
    use_deterministic_algorithms=False,
    deterministic_warn_only=False,
    label_keys=None,
    seed=None,
    dtype=torch.float32,
    additional_values=None,
):
    """
    Run a deep-learning EEG/BCI experiment using PyTorch models.

    This function provides an end-to-end pipeline for training and evaluating a
    PyTorch model on EEG/BCI datasets using rosoku's item-based data abstraction.
    It supports loading data via user callbacks (MNE Epochs or NumPy arrays),
    optional preprocessing and normalization, model training, grouped test
    evaluation, saliency-map computation, and exporting results.

    Compared to high-level frameworks, this function exposes explicit hooks for:

    - model creation (``callback_get_model``),
    - inference extraction (logits / predictions / probabilities),
    - early stopping control,
    - reproducibility control (seed and deterministic algorithms),

    while still handling the full experimental loop.

    Parameters
    ----------
    items_train : list
        List of items describing the training split. Each item is a user-defined
        object (often a dict) interpreted only by the data-loading callbacks.

    items_valid : list or None
        List of items describing the validation split. If ``None``, no validation
        data are used.

        Notes:
        If ``items_valid`` is ``None``, the built-in patience-style
        ``early_stopping`` is not allowed (see ``early_stopping``).

    items_test : list
        List defining test evaluation groups. Each element can be either:

        - a single item: treated as one test group
        - a list of items: loaded and merged as one test group

        Notes:
        Grouping is handled by :func:`rosoku.utils.load_data`. Each element of
        ``items_test`` corresponds to exactly one evaluation group (i.e., one row
        per metric set in the returned results, per test group).

    callback_load_epochs : callable | None, optional
        Loader returning an :class:`mne.Epochs` instance.

        The callback **must** have the signature::

            callback_load_epochs(items, split)

        where ``items`` is a list of items and ``split`` is one of
        ``{"train", "valid", "test"}``.

        For grouped items (e.g., a test group with multiple sessions), the callback
        should implement merging/concatenation logic (e.g.,
        :func:`mne.concatenate_epochs`).

        Exactly one of ``callback_load_epochs`` and ``callback_load_ndarray`` must be
        provided (enforced in :func:`rosoku.utils.load_data`).

    callback_load_ndarray : callable | None, optional
        Loader returning ``(X, y)`` arrays.

        The callback **must** have the signature::

            callback_load_ndarray(items, split)

        and return a tuple ``(X, y)`` where ``X`` is array-like and ``y`` are labels.

        Exactly one of ``callback_load_epochs`` and ``callback_load_ndarray`` must be
        provided (enforced in :func:`rosoku.utils.load_data`).

    criterion : torch.nn.Module, optional
        Loss function used for training (default:
        :class:`torch.nn.CrossEntropyLoss`).

    batch_size : int, optional
        Mini-batch size used for training and inference.

    n_epochs : int, optional
        Maximum number of training epochs.

    optimizer : type, optional
        Optimizer class (not an instance), e.g. :class:`torch.optim.AdamW`.

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

    callback_proc_epochs : callable | None, optional
        Optional preprocessing applied to Epochs objects before conversion to arrays.

    callback_proc_ndarray : callable | None, optional
        Optional preprocessing applied to ndarray data via ``apply_callback_proc``.

    callback_convert_epochs_to_ndarray : callable, optional
        Converter from Epochs to ``(X, y)`` arrays.
        Called as ``callback_convert_epochs_to_ndarray(epochs, split)``.

    callback_get_logits : callable | None, optional
        Optional hook to compute logits during inference.

        If provided, it is called as::

            callback_get_logits(model, X)

        If ``None``, logits are computed by the default forward pass::

            logits = model(X)

        The returned ``logits`` must be shaped ``(n_samples, n_classes)``.

    callback_get_preds : callable | None, optional
        Optional hook to compute predicted labels during inference.

        If provided, it is called as::

            callback_get_preds(model, X)

        If ``None``, predictions are computed from logits as::

            preds = torch.argmax(logits, dim=1)

        The returned ``preds`` must be a 1D tensor/array of length ``n_samples``.

    callback_get_probas : callable | None, optional
        Optional hook to compute class probabilities during inference.

        If provided, it is called as::

            callback_get_probas(model, X)

        If ``None``, probabilities are computed from logits as::

            probas = torch.nn.functional.softmax(logits, dim=1)

        The returned ``probas`` must be shaped ``(n_samples, n_classes)``.

    callback_get_model : callable | None, optional
        Factory function returning a ``torch.nn.Module`` instance.

        If ``model`` is ``None``, this callback **must** be provided and is called as::

            callback_get_model(X_train, y_train)

        where ``X_train`` and ``y_train`` are arrays returned by
        :func:`rosoku.utils.load_data`.

        If ``model`` is provided explicitly, this callback is ignored.

    callback_early_stopping : callable | None, optional
        User-defined early stopping callback.

        If provided, this callback is evaluated once per epoch during training.
        It must accept a single dictionary describing the current training state
        and return a boolean value:

        - ``True``  → stop training (early stopping is triggered)
        - ``False`` → continue training

        The callback is invoked as::

            callback_early_stopping(state)

        where ``state`` is a dictionary with the following keys:

        ``"train_loss"`` : float
            Training loss at the current epoch.

        ``"valid_loss"`` : float
            Validation loss at the current epoch.

        ``"epoch"`` : int
            Current epoch number (1-based).

        When the callback returns ``True``, the training loop is immediately
        terminated after the current epoch.

        Notes:

        - The callback is evaluated after both training and validation losses
          have been computed for the epoch.
        - The callback itself is responsible for maintaining any internal state
          (e.g., best loss, patience counters).
        - ``callback_early_stopping`` and ``early_stopping`` cannot be specified
          at the same time.
        - This callback can be used even when ``items_valid`` is ``None``,
          provided the callback logic does not rely on ``"valid_loss"``.

    callback_normalization : callable | None, optional
        Normalization function applied to train/validation/test arrays.

        By default, ``rosoku.preprocessing.normalize`` is used, which applies
        z-score normalization based on the training data and returns
        ``(X_train, X_valid, X_test, mean, std)``.

        If provided, this callback is invoked **after data loading and preprocessing**
        and **before model creation and training**. It is responsible for applying
        normalization (e.g., z-score normalization) consistently across splits.

        The callback **must** be callable and is invoked as::

            callback_normalization(X_train, X_valid, X_test)

        The return value must be either:

        - ``(X_train, X_valid, X_test)``
        - ``(X_train, X_valid, X_test, mean, std)``

        where:

        ``X_train`` : numpy.ndarray
            Normalized training data.

        ``X_valid`` : numpy.ndarray or None
            Normalized validation data.

        ``X_test`` : list of numpy.ndarray
            Normalized test data (one array per test group).

        ``mean`` / ``std`` : numpy.ndarray, optional
            Normalization parameters. These are required **only if**
            ``normalization_fname`` is provided.

        Behavior details:

        - If ``callback_normalization`` is ``None``, no normalization is applied.
        - If ``normalization_fname`` is provided, the callback **must** return
          ``mean`` and ``std`` in addition to the normalized arrays.
        - ``mean`` and ``std`` are saved to ``normalization_fname`` using msgpack.
        - rosoku does not enforce a specific normalization strategy; the callback
          fully defines the behavior.

        Recommended usage:

        - Compute statistics (mean/std) **only from training data** and apply them
          to validation and test data to avoid data leakage.

    optimizer_params : dict | None, optional
        Keyword arguments forwarded to the optimizer constructor.

    model : torch.nn.Module | None, optional
        Pre-instantiated model. If ``None``, ``callback_get_model`` must be provided.

    scheduler : type | None, optional
        Learning-rate scheduler class (not an instance).

    scheduler_params : dict | None, optional
        Keyword arguments forwarded to the scheduler constructor.

    min_delta : float, optional
        Minimum improvement threshold used for **model improvement detection**.

        This parameter is used in two places depending on the configuration:

        1. **Checkpoint saving**
           When ``checkpoint_fname`` is provided, the training routine tracks the
           best validation loss and saves model parameters only if the monitored
           loss improves by at least ``min_delta`` compared to the previously best
           value.

           A new checkpoint is written if::

               current_loss < best_loss - min_delta

        2. **Early stopping (when ``early_stopping`` is an integer)**
           When ``early_stopping`` is specified as an integer, ``min_delta`` is
           forwarded to :class:`rosoku.utils.EarlyStopping` and used as the minimum
           required improvement to reset the patience counter.

    device : {"cpu", "cuda"}, optional
        Device used for training and inference.

    num_workers : int, optional
        Number of worker processes used by PyTorch DataLoaders.

        **For maximum reproducibility, it is strongly recommended to use
        ``num_workers=0``.**

        Using multiple workers may introduce non-determinism depending on the
        dataset, transforms, and system configuration.

    scoring : str | callable | list of (str or callable), optional
        Scoring metric(s) computed on each test group.

        If a string, it is resolved via :func:`sklearn.metrics.get_scorer` and the
        underlying ``_score_func`` is used.
        If a callable, it must have signature ``scoring(y_true, y_pred)``.

    scoring_name : str | list of str | None, optional
        Column names corresponding to ``scoring``. If ``None``, names are inferred:
        strings keep their name, callables become ``"callable"``.

    enable_wandb_logging : bool, optional
        If True, log metrics and predictions to Weights & Biases.

    wandb_params : dict | None, optional
        Parameters forwarded to ``wandb.init`` (handled inside training / caller code).

    checkpoint_fname : path-like | None, optional
        If provided, a checkpoint is loaded before test-time inference and
        ``model_state_dict`` is restored.

        Recommended extension: ``.pth``.

    history_fname : path-like | None, optional
        If provided, training history is saved by the training routine.

        Recommended extension: ``.parquet`` (pandas DataFrame).

    samples_fname : path-like | None, optional
        If provided, writes sample-level predictions to a Parquet file.

        Recommended extension: ``.parquet`` (pandas DataFrame).

    normalization_fname : path-like | None, optional
        If provided and ``enable_normalization=True``, saves normalization parameters
        (mean/std) via msgpack.

        Recommended extension: ``.msgpack``.

    saliency_map_fname : path-like | None, optional
        If provided, computes saliency maps for each test group and class and saves
        them via msgpack.

        Recommended extension: ``.msgpack``.

    early_stopping : int | None, optional
        Patience-style early stopping parameter interpreted by the training routine.

        It will raise ValueError, if ``items_valid`` is ``None`` and ``early_stopping`` is provided.
        In that case, use ``callback_early_stopping`` instead.

    model_name : str | None, optional
        Model name recorded in outputs. Defaults to ``model.__class__.__name__``.

    use_deterministic_algorithms : bool, optional
        If True, enforce deterministic algorithms via
        :func:`torch.use_deterministic_algorithms`.

        When enabled, PyTorch may raise a :class:`RuntimeError` if an operation does
        not have a deterministic implementation on the current backend/device.

    deterministic_warn_only : bool, optional
        Passed to :func:`torch.use_deterministic_algorithms` as ``warn_only``.

        If True, PyTorch will emit warnings instead of raising errors when it
        encounters non-deterministic operations (useful for debugging and gradually
        hardening reproducibility).

    label_keys : dict | None, optional
        Mapping from class labels to integer IDs, used for saliency map computation.
        If ``None``, labels may be inferred from ``y_test`` (implementation-dependent).

    seed : int | None, optional
        Random seed controlling NumPy, Python, and PyTorch RNGs. When provided,
        deterministic backend settings are enabled:

        - ``torch.backends.cudnn.deterministic = True``
        - ``torch.backends.cudnn.benchmark = False``
        - TF32 disabled for matmul and cuDNN

    dtype : torch.dtype, optional
        Default floating-point dtype forwarded to the training routine via ``kwargs``.
        Useful when you need to force ``float32`` (recommended for stability) or
        experiment with other dtypes.

    additional_values : dict | None, optional
        Extra metadata appended as columns to output DataFrames (summary and samples).

    Returns
    -------
    df : pandas.DataFrame
        Summary results with one row per test group. Includes JSON-serialized
        ``items_train`` / ``items_valid`` / ``items_test`` specifications, scoring
        metrics, and the model name.

    Notes
    -----
    Reproducibility recommendations (strict mode)
        - set ``seed`` to a fixed integer
        - set ``use_deterministic_algorithms=True`` (optionally start with
          ``deterministic_warn_only=True`` to identify offending ops)
        - use ``num_workers=0``
        - avoid non-deterministic preprocessing / augmentation steps

    Output file extensions
        rosoku does **not** automatically append file extensions.
        Please explicitly include extensions in filenames. Recommended extensions:

        - ``history_fname``        → ``.parquet``  (pandas DataFrame)
        - ``checkpoint_fname``     → ``.pth``      (PyTorch checkpoint)
        - ``samples_fname``        → ``.parquet``  (pandas DataFrame)
        - ``normalization_fname``  → ``.msgpack``  (msgpack-serialized dict)
        - ``saliency_map_fname``   → ``.msgpack``  (msgpack-serialized dict)
    """
    if enable_wandb_logging:
        import wandb

    if (callback_early_stopping is not None) and (early_stopping is not None):
        raise ValueError(
            "Cannot specify both callback_early_stopping and early_stopping"
        )

    if (items_valid is None) and (early_stopping is not None):
        raise ValueError(
            "Cannot specify early_stopping without items_valid. Use callback_early_stpping instead"
        )

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if use_deterministic_algorithms:
        torch.use_deterministic_algorithms(
            use_deterministic_algorithms, warn_only=deterministic_warn_only
        )

    # load data
    X_train, X_valid, X_test, y_train, y_valid, y_test = utils.load_data(
        items_train=items_train,
        items_valid=items_valid,
        items_test=items_test,
        callback_load_epochs=callback_load_epochs,
        callback_load_ndarray=callback_load_ndarray,
        callback_proc_epochs=callback_proc_epochs,
        callback_proc_ndarray=callback_proc_ndarray,
        callback_proc_mode=callback_proc_mode,
        callback_convert_epochs_to_ndarray=callback_convert_epochs_to_ndarray,
    )

    if len(items_test) != len(X_test):
        raise RuntimeError("len(items_test) != len(X_test)")

    # data normalization
    if callback_normalization is not None:
        if not callable(callback_normalization):
            raise ValueError("callback_normalization must be callable")

        normalization_tuple = callback_normalization(X_train, X_valid, X_test)

        X_train = normalization_tuple[0]
        X_valid = normalization_tuple[1]
        X_test = normalization_tuple[2]

        if normalization_fname is not None:
            if len(normalization_tuple) != 5:
                raise RuntimeError(
                    "callback_normalization should return X_train, X_valid, X_test, mean, std, if normalization_fname is provided"
                )

            normalization_mean = normalization_tuple[3]
            normalization_std = normalization_tuple[4]

            normalization_dict = {
                "mean": normalization_mean.squeeze().tolist(),
                "std": normalization_std.squeeze().tolist(),
            }

            with open(normalization_fname, "wb") as f:
                msgpack.pack(normalization_dict, f)

    if model is None:
        if callback_get_model is None:
            raise ValueError("callback_get_model must be provided if model is None")
        model = callback_get_model(X_train, y_train)

    model.to(device)

    if model_name is None:
        model_name = model.__class__.__name__

    kwargs = {
        "optimizer_params": optimizer_params,
        "model": model,
        "callback_early_stopping": callback_early_stopping,
        "scheduler": scheduler,
        "scheduler_params": scheduler_params,
        "min_delta": min_delta,
        "enable_wandb_logging": enable_wandb_logging,
        "wandb_params": wandb_params,
        "checkpoint_fname": checkpoint_fname,
        "history_fname": history_fname,
        "early_stopping": early_stopping,
        "dtype": dtype,
        "seed": seed,
    }

    run_experiment(
        num_workers=num_workers,
        device=device,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        criterion=criterion,
        batch_size=batch_size,
        n_epochs=n_epochs,
        optimizer=optimizer,
        kwargs=kwargs,
    )

    if not isinstance(scoring, list):
        scoring = [scoring]

    if scoring_name is None:
        scoring_name = []
        for idx, scoring_ in enumerate(scoring):
            if isinstance(scoring_, str):
                scoring_name.append(scoring_)
            elif callable(scoring_):
                scoring_name.append("callable")
            else:
                scoring_name.append("unknown_scoring")

    if not isinstance(scoring_name, list):
        scoring_name = [scoring_name]

    if len(scoring) != len(scoring_name):
        raise RuntimeError("len(scoring) != len(scoring_name)")

    for idx, scoring_ in enumerate(scoring):
        if isinstance(scoring_, str):
            from sklearn.metrics import get_scorer

            scoring_ = get_scorer(scoring_)._score_func
        elif callable(scoring_):
            # do nothing
            pass
        else:
            raise ValueError(f"Invalid scoring: {scoring_}")
        scoring[idx] = scoring_

    # classify test data
    if checkpoint_fname is not None:
        checkpoint = torch.load(checkpoint_fname, map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])

    (_, _, dataloader_test) = utils.ndarray_to_dataloader(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        device="cpu",
        batch_size=batch_size,
        seed=seed,
        generator=None,
    )

    if not isinstance(dataloader_test, list):
        dataloader_test = [dataloader_test]

    model.eval()

    if saliency_map_fname is not None:
        saliency_data = {}
        for idx, dataloader in enumerate(dataloader_test):
            saliency_data[idx] = {}
            if label_keys is None:
                classes = np.unique(y_test).tolist()

                label_keys = {f"{c}": c for c in classes}

            for class_label, c in label_keys.items():
                s = attribution.saliency_map(model, dataloader, device, class_index=c)

                s = s.tolist()

                saliency_data[idx][class_label] = s

        with open(saliency_map_fname, "wb") as f:
            msgpack.pack(saliency_data, f)

    df_list = []
    samples_list = []
    with torch.no_grad():
        for idx, (dataloader, items_test_single) in enumerate(
            zip(dataloader_test, items_test)
        ):

            preds, labels, logits, probas = utils.get_predictions(
                model,
                dataloader,
                device=device,
                callback_get_logits=callback_get_logits,
                callback_get_preds=callback_get_preds,
                callback_get_probas=callback_get_probas,
            )

            scores = []
            for scoring_ in scoring:
                scores.append(scoring_(labels, preds))

            df_results = pd.DataFrame()
            df_results["items_train"] = [json.dumps(items_train)]
            df_results["items_valid"] = [json.dumps(items_valid)]
            df_results["items_test"] = [json.dumps(items_test_single)]

            wandb_log = {}
            for scoring_name_, score in zip(scoring_name, scores):
                df_results[scoring_name_] = [score]
                wandb_log[f"tset/{scoring_name_}"] = score

            samples = pd.DataFrame()
            samples["labels"] = labels
            samples["preds"] = preds
            for idx in range(probas.shape[1]):
                samples[f"probas_{idx}"] = probas[:, idx]
            for idx in range(logits.shape[1]):
                samples[f"logits_{idx}"] = logits[:, idx]
            samples["model"] = [model_name for _ in range(len(samples))]
            if additional_values is not None:
                samples = _add_values_to_df(samples, additional_values)

            if enable_wandb_logging:
                table = wandb.Table(columns=["id", "labels", "preds"])
                for idx, (label, pred) in enumerate(zip(labels, preds)):
                    table.add_data(idx, label, pred)

                wandb_log.update({"predictions": table})
                wandb.log(wandb_log)

            samples_list.append(samples)
            df_list.append(df_results)

    df = pd.concat(df_list, axis=0, ignore_index=True)
    df["model"] = [model_name for _ in range(len(df))]
    if additional_values is not None:
        df = _add_values_to_df(df, additional_values)

    if samples_fname is not None:
        samples = pd.concat(samples_list, axis=0, ignore_index=True)
        samples.to_parquet(samples_fname)

    if enable_wandb_logging:
        wandb.finish()

    return df
