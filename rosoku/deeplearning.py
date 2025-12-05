import os
import time
import random
import json
import msgpack

import numpy as np
import sklearn

import torch

import pandas as pd

from . import utils
from . import preprocessing
from . import attribution


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


def load_data(
        subjects, func_get_fnames, func_proc_epochs, label_keys, enable_euclidean_alignment
):
    X = []
    y = []

    for subject in subjects:

        files = func_get_fnames(subject)

        epochs = utils.load_epochs(files, True)

        if func_proc_epochs is not None:
            epochs = func_proc_epochs(epochs)

        y.append(np.array(utils.get_labels_from_epochs(epochs, label_keys)))

        X_single = epochs.get_data()

        if enable_euclidean_alignment:
            from . import tl

            X_single = tl.euclidean_alignment(X_single)

        X.append(X_single)

    return X, y


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
        valid_loss = utils.train_epoch(
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
        df_save.to_parquet(f"{history_fname}.parquet")
        # df_save.to_html(f"{history_fname}.html")

    return model


def load_data(
        subjects, func_get_fnames, func_proc_epochs, label_keys, enable_euclidean_alignment
):
    X = []
    y = []

    for subject in subjects:

        files = func_get_fnames(subject)

        epochs = utils.load_epochs(files, True)

        if func_proc_epochs is not None:
            epochs = func_proc_epochs(epochs)

        y.append(np.array(utils.get_labels_from_epochs(epochs, label_keys)))

        X_single = epochs.get_data()

        if enable_euclidean_alignment:
            from . import tl

            X_single = tl.euclidean_alignment(X_single)

        X.append(X_single)

    return X, y


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
    func_get_model = kwargs.get("func_get_model", None)
    scheduler = kwargs.get("scheduler", None)
    scheduler_params = kwargs.get("scheduler_params", None)
    enable_wandb_logging = kwargs.get("enable_wandb_logging", False)
    wandb_params = kwargs.get("wandb_params", None)
    checkpoint_fname = kwargs.get("checkpoint_fname", None)
    history_fname = kwargs.get("history_fname", None)
    early_stopping = kwargs.get("early_stopping", None)
    name_classifier = kwargs.get("name_classifier", None)
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
                "num_workers": num_workers,
                "rank": rank,
            },
            generator=seed,
        )
    else:

        (dataloader_train, dataloader_valid, _) = utils.nd_to_dataloader(
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

    if func_get_model is not None:
        model = func_get_model(X_train, y_train)

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

    if name_classifier is None:
        name_classifier = model.__class__.__name__

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
        keywords_train,
        keywords_valid,
        keywords_test,
        func_load_epochs=None,
        func_load_ndarray=None,
        criterion=torch.nn.CrossEntropyLoss(),
        batch_size=64,
        n_epochs=500,
        optimizer=torch.optim.AdamW,
        *,
        func_proc_mode="per_split",
        func_proc_epochs=None,
        func_proc_ndarray=None,
        func_convert_epochs_to_ndarray=utils.convert_epochs_to_ndarray,
        optimizer_params=None,
        model=None,
        func_get_model=None,
        scheduler=None,
        scheduler_params=None,
        device="cpu",
        enable_ddp=False,
        enable_dp=False,
        num_workers=0,
        enable_wandb_logging=False,
        wandb_params=None,
        checkpoint_fname=None,
        history_fname=None,
        samples_fname=None,
        normalization_fname=None,
        saliency_map_fname=False,
        early_stopping=None,
        name_classifier=None,
        enable_normalization=False,
        label_keys=None,
        seed=None,
        desc=None,
        additional_values=None,
):
    """
    General-purpose deep learning pipeline for EEG/BCI experiments.

    This function provides a flexible interface for loading data, preprocessing,
    training deep learning models, evaluating performance, and exporting results.
    The behavior is fully customizable through callback functions.

    ---------------------------------------------------------------------------
    Data loading via “keywords” and “mode”
    ---------------------------------------------------------------------------
    The arguments ``keywords_train``, ``keywords_valid``, and ``keywords_test``
    are arbitrary user-defined objects (typically dicts) that specify how data
    should be loaded. They are passed, together with a ``mode`` string, to the
    callback functions ``func_load_epochs`` or ``func_load_ndarray``.

    - First argument:  ``keyword`` (one element of keywords_*)  
    - Second argument: ``mode`` ∈ {"train", "valid", "test"}

    This allows you to implement different behavior depending on the split,
    e.g. data augmentation only for training data.

    Example
    -------
    >>> keywords_train = [{"subject": 1, "session": 1},
    ...                   {"subject": 2, "session": 1}]

    .. code-block:: python

        def func_load_epochs(keyword, mode):
            subject = keyword["subject"]
            session = keyword["session"]
            fname = f"sub-{subject}_ses-{session}-epo.fif"
            epochs = mne.read_epochs(fname)

            if mode == "train":
                # optional: apply stronger augmentation only on training data
                epochs = epochs.crop(tmin=0.0, tmax=1.0)

            return epochs

    ---------------------------------------------------------------------------
    Grouping test data
    ---------------------------------------------------------------------------
    ``keywords_test`` controls how test data are grouped for classification.

    - ``[[a], [b]]`` → evaluate a and b **separately**  
    - ``[[a, b]]`` → **merge** the data associated with a and b and evaluate them together

    This allows flexible control over whether each test set is evaluated
    individually or jointly.

    ---------------------------------------------------------------------------

    Parameters
    ----------
    keywords_train : list
        List of keyword objects used to load training data.

    keywords_valid : list
        List of keyword objects used to load validation data.

    keywords_test : list of list
        Controls grouping of test data.
        Each inner list represents one test evaluation group.

    func_load_epochs : callable, optional
        Callback function for loading MNE Epochs. It must accept:

        .. code-block:: python

            def func_load_epochs(keyword, mode):
                ...

        where

        - ``keyword`` is one element from ``keywords_train/valid/test``
        - ``mode`` is one of ``"train"``, ``"valid"``, ``"test"``

        The function must return an ``mne.Epochs`` instance.

    func_load_ndarray : callable, optional
        Callback function that loads data as NumPy arrays instead of Epochs. It must accept:

        .. code-block:: python

            def func_load_ndarray(keyword, mode):
                ...

        and return a tuple ``(X, y)`` where ``X`` and ``y`` are NumPy arrays.

    criterion : torch.nn.Module
        Loss function instance (default: ``CrossEntropyLoss``).

    batch_size : int
        Batch size used for training.

    n_epochs : int
        Number of training epochs.

    optimizer : type
        Reference to an optimizer class (not an instance), e.g.:

        >>> optimizer = torch.optim.AdamW

    optimizer_params : dict, optional
        ``**kwargs`` passed to the optimizer constructor.

    model : torch.nn.Module, optional
        Predefined model instance. If ``None``, ``func_get_model`` must be provided.

    func_get_model : callable, optional
        Function receiving ``(X_train, y_train)`` and returning a model instance.
        Useful when model architecture depends on the input shape.

    scheduler : type, optional
        Reference to a learning-rate scheduler class.

    scheduler_params : dict, optional
        ``**kwargs`` passed to the scheduler constructor.

    device : {"cpu", "cuda"}
        Device used for training and inference.

    enable_ddp : bool
        Enable Distributed Data Parallel (DDP).
        ``device`` must be ``"cuda"`` when True.

    enable_dp : bool
        Enable DataParallel. Cannot be True at the same time as DDP.

    num_workers : int
        Number of data-loading worker processes per GPU.
        Effective only when ``enable_ddp=True``.

    func_proc_epochs : callable, optional
        Function that receives an ``mne.Epochs`` object and returns a processed one.
        Useful for channel selection, cropping, filtering, etc.

    func_proc_ndarray : callable, optional
        Preprocessing function for NumPy data.

    func_proc_mode : {"per_split", "all"}
        Defines whether preprocessing is applied independently to each split
        or jointly across all splits.

    func_convert_epochs_to_ndarray : callable
        Converter from MNE Epochs to NumPy arrays.

    enable_normalization : bool
        Whether to apply z-score normalization to X_train/X_valid/X_test.

    label_keys : dict, optional
        Mapping from label strings to integer class IDs.

    enable_wandb_logging : bool
        Enable logging to Weights & Biases.

    wandb_params : dict, optional
        Arguments passed to ``wandb.init()``.

    checkpoint_fname : path-like, optional
        File path for saving/loading PyTorch checkpoints.
        File extension should be ".pth"

    history_fname : path-like, optional
        File path for saving training history (pkl or html).
        File extension should be ".parquet"

    samples_fname : path-like, optional
        File path for saving sample-level predictions (Parquet).
        File extension should be ".parquet"

    normalization_fname : path-like, optional
        File path for saving normalization parameters (mean/std).
        File extension should be ".msgpack"

    saliency_map_fname : path-like or None
        If provided, saliency maps are computed and saved via msgpack.
        File extension should be ".msgpack"

    early_stopping : int or callable, optional
        Patience or early stopping controller.

    name_classifier : str, optional
        Name of the classifier (for logging/output).

    seed : int, optional
        Random seed for NumPy, Python, and PyTorch CPU/GPU backends.

    additional_values : dict, optional
        Extra key–value pairs appended to the output DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing classification metrics (accuracy, F1, etc.)
        and metadata such as training/validation/test keywords and classifier name.
    """
    import torch

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
        keywords_train=keywords_train,
        keywords_valid=keywords_valid,
        keywords_test=keywords_test,
        func_load_epochs=func_load_epochs,
        func_load_ndarray=func_load_ndarray,
        func_proc_epochs=func_proc_epochs,
        func_proc_ndarray=func_proc_ndarray,
        func_proc_mode=func_proc_mode,
        func_convert_epochs_to_ndarray=func_convert_epochs_to_ndarray,
    )

    if len(keywords_test) != len(X_test):
        raise RuntimeError("len(keywords_test) != len(X_test)")

    # data normalization
    if enable_normalization:
        X_train, X_valid, X_test, normalization_mean, normalization_std = (
            preprocessing.normalize(X_train, X_valid, X_test, return_params=True)
        )

    kwargs = {
        "optimizer_params": optimizer_params,
        "model": model,
        "func_get_model": func_get_model,
        "scheduler": scheduler,
        "scheduler_params": scheduler_params,
        "func_proc_epochs": func_proc_epochs,
        "enable_wandb_logging": enable_wandb_logging,
        "wandb_params": wandb_params,
        "checkpoint_fname": checkpoint_fname,
        "history_fname": history_fname,
        "early_stopping": early_stopping,
        "name_classifier": name_classifier,
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
        model = func_get_model(X_train, y_train)

    # device = "cuda" if enable_ddp else "cpu"
    model.to(device)

    # classify test data
    if checkpoint_fname is not None:
        # checkpoint = torch.load(f"{checkpoint_fname}")
        checkpoint = torch.load(checkpoint_fname, map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])

    (_, _, dataloader_test) = utils.nd_to_dataloader(
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

    if isinstance(dataloader_test, list) is False:
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
                s = attribution.saliency_map(
                    model, dataloader, device, class_index=c
                )

                s = s.tolist()

                saliency_data[idx][class_label] = s

        with open(saliency_map_fname, "wb") as f:
            msgpack.pack(saliency_data, f)

    df_list = []
    samples_list = []
    with torch.no_grad():
        for idx, (dataloader, keywords_test_single) in enumerate(
                zip(dataloader_test, keywords_test)
        ):

            preds, labels, logits, probas = utils.get_predictions(
                model,
                dataloader,
                device=device,
            )

            accuracy = sklearn.metrics.accuracy_score(labels, preds)
            f1 = sklearn.metrics.f1_score(labels, preds)
            bacc = sklearn.metrics.balanced_accuracy_score(labels, preds)

            df_results = pd.DataFrame()
            df_results["keywords_train"] = [json.dumps(keywords_train)]
            df_results["keywords_valid"] = [json.dumps(keywords_valid)]
            df_results["keywords_test"] = [json.dumps(keywords_test_single)]
            df_results["classifier"] = [name_classifier]
            df_results["accuracy"] = [accuracy]
            # df_results["labels"] = [labels]
            # df_results["preds"] = [preds]
            # df_results["probas"] = [probas]
            # df_results["logits"] = [logits]

            if normalization_fname is not None:
                normalization_dict = {"mean": normalization_mean.squeeze().tolist(),
                                      "std": normalization_std.squeeze().tolist()}

                with open(normalization_fname, "wb") as f:
                    msgpack.pack(normalization_dict, f)

            samples = pd.DataFrame()
            samples["labels"] = labels
            samples["preds"] = preds
            for idx in range(probas.shape[1]):
                samples[f"probas_{idx}"] = probas[:, idx]
            for idx in range(logits.shape[1]):
                samples[f"logits_{idx}"] = logits[:, idx]
            samples["classifier"] = [name_classifier for _ in range(len(samples))]

            if enable_wandb_logging:
                if (enable_ddp and params["rank"] == 0) or (enable_ddp is False):
                    table = wandb.Table(columns=["id", "labels", "preds"])
                    for idx, (label, pred) in enumerate(zip(labels, preds)):
                        table.add_data(idx, label, pred)

                    wandb.log(
                        {
                            "test/accuracy": accuracy,
                            "test/bacc": bacc,
                            "test/f1": f1,
                            "predictions": table,
                        }
                    )

            samples_list.append(samples)
            df_list.append(df_results)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    if additional_values is not None:
        for key, value in additional_values.items():
            df[key] = [value for m in range(df.shape[0])]

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
