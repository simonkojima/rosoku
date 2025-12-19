import time
import random
import json
import msgpack

import numpy as np

import torch

import pandas as pd

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
        enable_wandb_logging=False,
        wandb_params=None,
        checkpoint_fname=None,
        history_fname=None,
        rank=0,
):
    if enable_wandb_logging:
        import wandb

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
        valid_loss = _train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
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
    callback_get_model = kwargs.get("callback_get_model", None)
    scheduler = kwargs.get("scheduler", None)
    scheduler_params = kwargs.get("scheduler_params", None)
    enable_wandb_logging = kwargs.get("enable_wandb_logging", False)
    wandb_params = kwargs.get("wandb_params", None)
    checkpoint_fname = kwargs.get("checkpoint_fname", None)
    history_fname = kwargs.get("history_fname", None)
    early_stopping = kwargs.get("early_stopping", None)
    seed = kwargs.get("seed", None)

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
    )

    # setup model

    if callback_get_model is not None:
        model = callback_get_model(X_train, y_train)

    if model is None:
        raise RuntimeError("model is None")

    model.to(device)

    # setup optimizer
    optimizer = setup_optimizer(optimizer, optimizer_params, model)

    # setup scheduler
    scheduler = setup_scheduler(scheduler, scheduler_params, optimizer)

    # setup early stopping
    if isinstance(early_stopping, int):
        early_stopping = utils.EarlyStopping(patience=early_stopping)

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
        optimizer_params=None,
        model=None,
        scheduler=None,
        scheduler_params=None,
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
        enable_normalization=False,
        label_keys=None,
        seed=None,
        additional_values=None,
):
    """
    Run a deep-learning EEG/BCI experiment using PyTorch models.

    This function provides an end-to-end pipeline for training and evaluating a
    PyTorch model on EEG/BCI datasets using rosoku's item-based data abstraction.
    It supports loading data via user callbacks (MNE Epochs or NumPy arrays),
    optional preprocessing and normalization, model training, grouped test
    evaluation, saliency-map computation, and result export.

    Compared to high-level frameworks, this function exposes explicit hooks for
    model creation, prediction extraction (logits / predictions / probabilities),
    and reproducibility control, while still handling the full experimental loop.

    Parameters
    ----------
    items_train : list
        List of items describing the training split. Each item is a user-defined
        object (often a dict) interpreted only by the data-loading callbacks.

    items_valid : list or None
        List of items describing the validation split. If ``None``, no validation
        data are used.

    items_test : list
        List defining test evaluation groups. Each element can be either:

        - a single item → treated as one test group
        - a list of items → loaded and merged as one test group

    callback_load_epochs : callable | None, optional
        Loader returning an :class:`mne.Epochs` instance.
        Must have signature ``callback_load_epochs(items, split)`` where ``split`` is
        one of ``{"train", "valid", "test"}``.

    callback_load_ndarray : callable | None, optional
        Loader returning ``(X, y)`` arrays.
        Must have signature ``callback_load_ndarray(items, split)``.

    criterion : torch.nn.Module, optional
        Loss function used for training (default:
        :class:`torch.nn.CrossEntropyLoss`).

    batch_size : int, optional
        Mini-batch size used for training and inference.

    n_epochs : int, optional
        Number of training epochs.

    optimizer : type, optional
        Optimizer class (not an instance), e.g. :class:`torch.optim.AdamW`.

    callback_proc_mode : {"per_split", "all"}, optional
        Strategy controlling how preprocessing callbacks are applied.

    callback_proc_epochs : callable | None, optional
        Optional preprocessing applied to Epochs objects before conversion to arrays.

    callback_proc_ndarray : callable | None, optional
        Optional preprocessing applied to NumPy arrays via ``apply_callback_proc``.

    callback_convert_epochs_to_ndarray : callable, optional
        Converter from Epochs to ``(X, y)`` arrays. Called as
        ``callback_convert_epochs_to_ndarray(epochs, split)``.

    callback_get_logits : callable | None, optional
        Hook to extract logits during inference.
        If ``None``, logits are obtained via ``model(X)``.

    callback_get_preds : callable | None, optional
        Hook to compute predicted labels.
        If ``None``, predictions are computed as ``argmax(logits, dim=1)``.

    callback_get_probas : callable | None, optional
        Hook to compute class probabilities.
        If ``None``, probabilities are computed using softmax over logits.

    callback_get_model : callable | None, optional
        Factory function returning a ``torch.nn.Module``.
        Called as ``callback_get_model(X_train, y_train)`` if ``model`` is ``None``.

    optimizer_params : dict | None, optional
        Keyword arguments passed to the optimizer constructor.

    model : torch.nn.Module | None, optional
        Pre-instantiated model. If ``None``, ``callback_get_model`` must be provided.

    scheduler : type | None, optional
        Learning-rate scheduler class.

    scheduler_params : dict | None, optional
        Keyword arguments passed to the scheduler constructor.

    device : {"cpu", "cuda"}, optional
        Device used for training and inference.

    num_workers : int, optional
        Number of worker processes used by PyTorch DataLoaders.

        **For maximum reproducibility, it is strongly recommended to use
        ``num_workers=0``.**

        Using multiple workers may introduce non-determinism depending on the
        dataset, transformations, and system configuration.

    scoring : str | callable | list of (str or callable), optional
        Scoring metric(s) computed on each test group.

    scoring_name : str | list of str | None, optional
        Column names corresponding to ``scoring``.

    enable_wandb_logging : bool, optional
        If True, log metrics and predictions to Weights & Biases.

    wandb_params : dict | None, optional
        Parameters forwarded to ``wandb.init``.

    checkpoint_fname : path-like | None, optional
        Path to a checkpoint file loaded before test-time inference.

    history_fname : path-like | None, optional
        File path for saving training history.

    samples_fname : path-like | None, optional
        If provided, writes sample-level predictions (labels, preds, logits, probas)
        to a Parquet file.

    normalization_fname : path-like | None, optional
        If provided and ``enable_normalization=True``, saves normalization parameters.

    saliency_map_fname : path-like | None, optional
        If provided, computes saliency maps for each test group and class and saves
        them via msgpack.

    early_stopping : int | callable | None, optional
        Early stopping controller or patience parameter.

    model_name : str | None, optional
        Model name recorded in outputs. Defaults to ``model.__class__.__name__``.

    enable_normalization : bool, optional
        If True, apply z-score normalization to train/valid/test arrays.

    label_keys : dict | None, optional
        Mapping from class labels to integer IDs, used for saliency map computation.

    seed : int | None, optional
        Random seed controlling NumPy, Python, and PyTorch RNGs.

        When provided, the following deterministic settings are enabled:

        - ``torch.backends.cudnn.deterministic = True``
        - ``torch.backends.cudnn.benchmark = False``
        - TF32 disabled for matmul and cuDNN

    additional_values : dict | None, optional
        Extra metadata appended as columns to output DataFrames.

    Returns
    -------
    df : pandas.DataFrame
        Summary results with one row per test group. Includes JSON-serialized
        ``items_train`` / ``items_valid`` / ``items_test`` specifications, scoring
        metrics, and the model name.

    Notes
    -----
    - Grouped test evaluation is controlled entirely by ``items_test``.
    - For strict reproducibility (recommended for method comparison and papers):

      - set ``seed`` to a fixed integer
      - use ``num_workers = 0``
      - avoid non-deterministic preprocessing steps

    - Saliency maps are computed post-training using input gradients averaged over
      samples of the specified class.

    - Output file formats are determined solely by the filenames provided by the user.
      rosoku does **not** automatically append file extensions.

      Please ensure that you explicitly specify the desired file extension when
      providing output paths. The recommended extensions are:

      - ``history_fname``        → ``.parquet``   (pandas DataFrame)
      - ``checkpoint_fname``     → ``.pth``       (PyTorch checkpoint)
      - ``samples_fname``        → ``.parquet``   (pandas DataFrame)
      - ``normalization_fname``  → ``.msgpack``   (msgpack-serialized dict)
      - ``saliency_map_fname``   → ``.msgpack``   (msgpack-serialized dict)

      If no extension (or an unexpected one) is provided, the file will still be
      written, but its format may not be correctly inferred by downstream tools.

    Examples
    --------
    Minimal usage with a model factory::

        def get_model(X_train, y_train):
            return MyNet(
                n_ch=X_train.shape[1],
                n_times=X_train.shape[2],
                n_classes=len(np.unique(y_train)),
            )

        df = deeplearning(
            items_train=[{"sub": 1}],
            items_valid=None,
            items_test=[[{"sub": 1, "ses": 2}]],
            callback_load_ndarray=load_xy,
            callback_get_model=get_model,
            seed=42,
            num_workers=0,
        )
    """
    if enable_wandb_logging:
        import wandb

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

        torch.use_deterministic_algorithms(True)

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
    if enable_normalization:
        X_train, X_valid, X_test, normalization_mean, normalization_std = (
            preprocessing.normalize(X_train, X_valid, X_test, return_params=True)
        )

    kwargs = {
        "optimizer_params": optimizer_params,
        "model": model,
        "callback_get_model": callback_get_model,
        "scheduler": scheduler,
        "scheduler_params": scheduler_params,
        "enable_wandb_logging": enable_wandb_logging,
        "wandb_params": wandb_params,
        "checkpoint_fname": checkpoint_fname,
        "history_fname": history_fname,
        "early_stopping": early_stopping,
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

    if model is None:
        model = callback_get_model(X_train, y_train)

    model.to(device)

    if model_name is None:
        model_name = model.__class__.__name__

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

            if normalization_fname is not None:
                normalization_dict = {
                    "mean": normalization_mean.squeeze().tolist(),
                    "std": normalization_std.squeeze().tolist(),
                }

                with open(normalization_fname, "wb") as f:
                    msgpack.pack(normalization_dict, f)

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
