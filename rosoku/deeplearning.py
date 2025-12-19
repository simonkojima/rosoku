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
        enable_ddp=False,
        enable_dp=False,
        sampler_train=None,
        rank=0,
):
    if enable_wandb_logging:
        if (enable_ddp and rank == 0) or (enable_ddp is False):
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
        if (enable_ddp and rank == 0) or (enable_ddp is False):
            wandb.init(**wandb_params)

    tic = time.time()
    for epoch in range(n_epochs):
        if enable_ddp:
            sampler_train.set_epoch(epoch)
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
            enable_dp=enable_dp,
            enable_ddp=enable_ddp,
            rank=rank,
        )

        if early_stopping is not None:
            if enable_ddp is False and early_stopping(valid_loss):
                print(f"Early stopping was triggered: epoch #{epoch + 1}")
                break
            elif enable_ddp:
                should_stop = False
                if rank == 0:
                    should_stop = early_stopping(valid_loss)
                should_stop_tensor = torch.tensor(
                    should_stop, dtype=torch.uint8, device=device
                )
                torch.distributed.broadcast(should_stop_tensor, src=0)
                should_stop = bool(should_stop_tensor.item())

                if should_stop:
                    if rank == 0:
                        print(f"Early stopping was triggered: epoch #{epoch + 1}")
                    break

    toc = time.time()
    elapsed_time = toc - tic
    if rank == 0:
        print(f"Elapsed Time: {elapsed_time:.2f}s")

    if history_fname is not None and rank == 0:
        df_save = pd.DataFrame(history)
        df_save.to_parquet(history_fname)

    return model


def main(
        enable_ddp,
        enable_dp,
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

    # setup DDP
    if enable_ddp:

        params = utils.get_ddp_params()

        rank = params["rank"]
        local_rank = params["local_rank"]
        world_size = params["world_size"]
        master_addr = params["master_addr"]
        master_port = params["master_port"]

        print(f"rank: {rank}, world_size: {world_size}, local_rank: {local_rank}")
        print(f"MASTER_ADDR: {master_addr}, MASTER_PORT: {master_port}")

        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
        )

        if torch.distributed.is_initialized():
            print(f"[Rank {rank}] Distributed initialized: OK")
        else:
            print(f"[Rank {rank}] Distributed not initialized: NG")
            raise RuntimeError(f"[Rank {rank}] Distributed not initialized: NG")

        device = torch.device(f"cuda:{local_rank}")
    else:
        # non DDP
        rank = 0

    # create dataloader

    if enable_ddp:
        (dataloader_train, dataloader_valid, _, sampler_train) = utils.nd_to_dataloader(
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test,
            device="cpu",
            batch_size=batch_size,
            enable_DS=True,
            DS_params={
                "world_size": world_size,
                "rank": rank,
            },
            generator=seed,
        )
    else:

        (dataloader_train, dataloader_valid, _) = utils.ndarray_to_dataloader(
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test,
            device="cpu",
            batch_size=batch_size,
            enable_DS=False,
            generator=seed,
        )
        sampler_train = None

    # setup model

    if callback_get_model is not None:
        model = callback_get_model(X_train, y_train)

    if model is None:
        raise RuntimeError("model is None")

    model.to(device)

    if enable_dp:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        else:
            raise RuntimeError(
                "You need to have more than one GPU when enable_dp = True."
            )

    if enable_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )

    # setup optimizer
    optimizer = setup_optimizer(optimizer, optimizer_params, model)

    # setup scheduler
    scheduler = setup_scheduler(scheduler, scheduler_params, optimizer)

    # setup early stopping
    if isinstance(early_stopping, int):
        early_stopping = utils.EarlyStopping(patience=early_stopping)

    try:
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
            enable_ddp=enable_ddp,
            enable_dp=enable_dp,
            sampler_train=sampler_train,
            rank=rank,
        )
    finally:
        if enable_ddp:
            torch.distributed.destroy_process_group()


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
        enable_ddp=False,
        enable_dp=False,
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
        desc=None,
        additional_values=None,
):
    """
    Run a general-purpose deep-learning pipeline for EEG/BCI experiments.

    This function provides an end-to-end workflow to train and evaluate a PyTorch
    model on EEG/BCI datasets. It supports loading data as MNE Epochs or NumPy
    arrays via user callbacks, optional preprocessing and normalization, training
    with a configurable optimizer (and optional LR scheduler), and evaluation on
    grouped test sets.

    Several hooks are provided to customize model creation and prediction extraction
    (logits/predictions/probabilities) without changing the core pipeline.

    Test data can be evaluated in user-defined groups: each element of ``items_test``
    represents one evaluation group and can contain one or multiple items (e.g., to
    merge multiple sessions into a single test set before scoring).

    Parameters
    ----------
    items_train : list
        Item objects describing how to load the training data. The content is
        user-defined and interpreted by ``callback_load_epochs`` or
        ``callback_load_ndarray``.

    items_valid : list
        Item objects describing how to load the validation data.

    items_test : list of list
        Item objects describing how to load the test data, grouped for evaluation.
        Each inner list defines one evaluation group.

    callback_load_epochs : callable | None, optional
        Loader returning an :class:`mne.Epochs` instance. Must have signature
        ``callback_load_epochs(item, split)`` where ``split`` is one of
        ``{"train", "valid", "test"}``.

    callback_load_ndarray : callable | None, optional
        Loader returning a tuple ``(X, y)``. Must have signature
        ``callback_load_ndarray(item, split)`` where ``split`` is one of
        ``{"train", "valid", "test"}``.

    criterion : torch.nn.Module, optional
        Loss function instance used for training (default:
        :class:`torch.nn.CrossEntropyLoss`).

    batch_size : int, optional
        Mini-batch size used for training and inference.

    n_epochs : int, optional
        Number of training epochs.

    optimizer : type, optional
        Optimizer class (not an instance), e.g. :class:`torch.optim.AdamW`.

    callback_proc_mode : {"per_split", "all"}, optional
        Strategy for preprocessing across splits, as interpreted by
        ``utils.load_data``.

        Typical meanings are:

        - ``"per_split"``: process train/valid/test independently.
        - ``"all"``: process jointly (exact behavior depends on ``utils.load_data``).

    callback_proc_epochs : callable | None, optional
        Optional preprocessing applied to loaded Epochs (e.g., picking channels,
        cropping, filtering), as used by ``utils.load_data``.

    callback_proc_ndarray : callable | None, optional
        Optional preprocessing applied to array data, as used by ``utils.load_data``.

    callback_convert_epochs_to_ndarray : callable, optional
        Converter used when loading Epochs. By default,
        ``utils.convert_epochs_to_ndarray``.

    callback_get_logits : callable | None, optional
        Optional hook to extract logits from the model during inference.

        If provided, it is called as::

            callback_get_logits(model, X)

        If ``None``, logits are obtained by a direct forward pass::

            logits = model(X)

        The returned ``logits`` must be a 2D tensor/array of shape
        ``(n_samples, n_classes)``.

    callback_get_preds : callable | None, optional
        Optional hook to compute predicted class labels during inference.

        If provided, it is called as::

            callback_get_preds(model, X)

        If ``None``, predicted labels are computed from logits as::

            preds = torch.argmax(logits, dim=1)

        The returned ``preds`` must be a 1D tensor/array of length ``n_samples``.

    callback_get_probas : callable | None, optional
        Optional hook to compute class probabilities during inference.

        If provided, it is called as::

            callback_get_probas(model, X)

        If ``None``, probabilities are computed from logits as::

            probas = torch.nn.functional.softmax(logits, dim=1)

        The returned ``probas`` must be a 2D tensor/array of shape
        ``(n_samples, n_classes)``.

    callback_get_model : callable | None, optional
        Factory function that returns a ``torch.nn.Module`` instance.
        Used when ``model=None``.

        If provided, it is called as::

            callback_get_model(X_train, y_train)

        where ``X_train`` and ``y_train`` are the training arrays returned by
        ``utils.load_data``.

        This callback is useful when the model architecture depends on properties
        of the training data (e.g., number of channels, number of time samples,
        or number of classes).

    optimizer_params : dict | None, optional
        Keyword arguments passed to the optimizer constructor.

    model : torch.nn.Module | None, optional
        Pre-instantiated model. If provided, ``callback_get_model`` is ignored.

    scheduler : type | None, optional
        Learning-rate scheduler class (not an instance). If provided, it is
        configured inside the training routine.

    scheduler_params : dict | None, optional
        Keyword arguments passed to the scheduler constructor.

    device : {"cpu", "cuda"}, optional
        Device used for training and inference when DDP/DP is disabled.

    enable_ddp : bool, optional
        If True, enable DistributedDataParallel training. Requires ``device="cuda"``.
        DDP process parameters are obtained via ``utils.get_ddp_params``.

    enable_dp : bool, optional
        If True, enable DataParallel training. Cannot be True at the same time as
        ``enable_ddp``. Requires ``device="cuda"``.

    num_workers : int, optional
        Number of DataLoader workers per process/GPU. Effective only when
        ``enable_ddp=True``.

    scoring : str | callable | list of (str or callable), optional
        Scoring specification(s) computed on each test group.
        If a string, it is resolved with :func:`sklearn.metrics.get_scorer` and the
        underlying ``_score_func`` is used.
        If a callable, it must have signature ``scoring(y_true, y_pred)`` and return
        a scalar.

    scoring_name : str | list of str | None, optional
        Column name(s) for the returned scores. If ``None``, names are inferred:
        strings keep their name, callables become ``"callable"`` (and other types
        become ``"unknown_scoring"``). Must match ``scoring`` length.

    enable_wandb_logging : bool, optional
        If True, log metrics and predictions to Weights & Biases. In DDP, logging is
        performed only on rank 0.

    wandb_params : dict | None, optional
        Keyword arguments passed to ``wandb.init``.

    checkpoint_fname : path-like | None, optional
        If provided, loads a checkpoint before test-time inference and restores
        ``model_state_dict``. Typically ends with ``.pth``.

    history_fname : path-like | None, optional
        File path for saving training history, as handled by the training routine.

    samples_fname : path-like | None, optional
        If provided, writes sample-level outputs to this path in Parquet format.
        The file includes true labels, predicted labels, per-class probabilities,
        per-class logits, and the model name (plus ``additional_values`` if given).

    normalization_fname : path-like | None, optional
        If provided and ``enable_normalization=True``, saves normalization parameters
        (mean/std) via msgpack.

    saliency_map_fname : path-like | None, optional
        If provided, computes saliency maps for each test group and each class and
        saves them via msgpack.

    early_stopping : int | callable | None, optional
        Early stopping controller or patience parameter, as interpreted by the
        training routine.

    model_name : str | None, optional
        Name recorded in the outputs. If ``None``, defaults to
        ``model.__class__.__name__``.

    enable_normalization : bool, optional
        If True, apply z-score normalization to train/valid/test arrays using
        ``preprocessing.normalize``. When enabled, normalization parameters can be
        saved with ``normalization_fname``.

    label_keys : dict | None, optional
        Mapping from class label strings to integer IDs. Used for saliency map
        computation. If ``None``, it is inferred from unique values in ``y_test``.

    seed : int | None, optional
        Random seed for NumPy/Python/PyTorch. When provided, deterministic CuDNN
        settings are enabled.

    desc : str | None, optional
        Optional description forwarded to the training routine (e.g., for logging).

    additional_values : dict | None, optional
        Extra metadata appended as columns to the output DataFrame (and also to the
        sample-level table if ``samples_fname`` is provided).

    Returns
    -------
    df : pandas.DataFrame
        Summary results with one row per test group. Includes JSON-serialized
        ``items_train`` / ``items_valid`` / ``items_test`` strings, one column per
        requested scoring metric, and a ``"model"`` column.

    Notes
    -----
    - ``items_test`` grouping controls evaluation granularity: each inner list is
      treated as one test set after loading/merging by ``utils.load_data``.
    - If a scoring string is provided, this function uses
      ``sklearn.metrics.get_scorer(scoring)._score_func`` rather than calling the
      scorer object; ensure the callable matches your intended behavior.
    - In DDP mode, W&B logging is performed only on rank 0.
    - Saliency map computation runs over each test group and each class index.

    Examples
    --------
    Provide a model factory that depends on the input shape::

        def get_model(X_train, y_train):
            n_ch = X_train.shape[1]
            n_t = X_train.shape[2]
            n_classes = len(np.unique(y_train))
            return MyNet(n_ch=n_ch, n_times=n_t, n_classes=n_classes)

        df = deeplearning(
            items_train=[{"sub": 1, "ses": 1}],
            items_valid=[{"sub": 1, "ses": 2}],
            items_test=[[{"sub": 1, "ses": 3}]],
            callback_load_ndarray=load_xy,
            callback_get_model=get_model,
            device="cuda",
            n_epochs=200,
            scoring=["accuracy", "balanced_accuracy"],
        )
    """
    if enable_ddp:
        params = utils.get_ddp_params()

    if enable_wandb_logging:
        if (enable_ddp and params["rank"] == 0) or (enable_ddp is False):
            import wandb

    if enable_ddp and enable_dp:
        raise ValueError(
            "enable_ddp and enable_dp cannot be True at the same time. Choose one."
        )

    if (enable_ddp and device != "cuda") or (enable_dp and device != "cuda"):
        raise ValueError(
            "device have to be 'cuda' when enable_ddp = True or enable_dp = True."
        )

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        g = torch.Generator()
        g.manual_seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
        "callback_proc_epochs": callback_proc_epochs,
        "enable_wandb_logging": enable_wandb_logging,
        "wandb_params": wandb_params,
        "checkpoint_fname": checkpoint_fname,
        "history_fname": history_fname,
        "early_stopping": early_stopping,
        "seed": seed,
        "desc": desc,
    }

    if enable_ddp:
        main(
            enable_ddp=enable_ddp,
            enable_dp=enable_dp,
            num_workers=num_workers,
            device=None,
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
    else:
        main(
            enable_ddp=enable_ddp,
            enable_dp=enable_dp,
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
        enable_DS=False,
        generator=seed,
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
                if (enable_ddp and params["rank"] == 0) or (enable_ddp is False):
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

        if enable_ddp:
            ddp_params = utils.get_ddp_params()
            rank = ddp_params["rank"]

        if (enable_ddp and rank == 0) or (enable_ddp is False):
            wandb.finish()

    return df
