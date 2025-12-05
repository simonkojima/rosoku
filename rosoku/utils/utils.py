import os
import random
import time
import numpy as np
import tag_mne as tm


##
# alies for preprocessing modules will be deprecated


# def normalize(X_train, X_valid, X_test, return_params=False):
def normalize(*args, **kwargs):
    from .. import preprocessing

    import warnings

    warnings.simplefilter("default", DeprecationWarning)

    warnings.warn(
        "rosoku.utils.normalize() is deprecated and will be removed in a future version. Use rosoku.preprocessing.normalize() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # return preprocessing.normalize(X_train, X_valid, X_test, return_params)
    return preprocessing.normalize(*args, **kwargs)


def get_ddp_params():
    """
    Retrieve Distributed Data Parallel (DDP) configuration from environment variables.

    This function collects process rank, world size, master address, and master port
    from environment variables required for PyTorch Distributed Data Parallel (DDP).
    It supports both standard variables (``RANK``, ``LOCAL_RANK``) and Slurm-based
    scheduling variables (``SLURM_PROCID``, ``SLURM_LOCALID``) as a fallback.

    Returns
    -------
    params : dict
        Dictionary containing the distributed configuration:

        - ``"world_size"`` : int
          Total number of processes participating in the job.
        - ``"master_addr"`` : str
          Address of the master process.
        - ``"master_port"`` : str
          Port used to initialize communication.
        - ``"rank"`` : int
          Global process rank.
        - ``"local_rank"`` : int
          Local rank within the current node.

    Raises
    ------
    RuntimeError
        If required environment variables cannot be parsed from either
        PyTorch-style or Slurm-style distributed launch configurations.

    Notes
    -----
    - Expected environment variables:

        **Primary (PyTorch standard)**
        ``WORLD_SIZE, MASTER_ADDR, MASTER_PORT, RANK, LOCAL_RANK``

        **Fallback (Slurm)**
        ``SLURM_PROCID, SLURM_LOCALID``

    - Intended for use inside DDP initialization routines where distributed
      parameters must be inferred from the runtime environment.

    Examples
    --------
    >>> params = get_ddp_params()
    >>> params
    {'world_size': 4, 'master_addr': '10.0.0.1', 'master_port': '12345',
     'rank': 0, 'local_rank': 0}
    """
    import os

    try:
        world_size = int(os.environ["WORLD_SIZE"])
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
    except:
        raise RuntimeError(
            "WORLD_SIZE, MASTER_ADDR, MASTER_PORT was not parsed from os.environ."
        )

    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except:
        try:
            rank = int(os.environ["SLURM_PROCID"])
            local_rank = int(os.environ["SLURM_LOCALID"])
        except:
            raise RuntimeError(
                "SLURM_PROCID, SLURM_LOCALID or RANK, LOCAL_RANK was not parsed from os.environ."
            )

    params = {
        "world_size": world_size,
        "master_addr": master_addr,
        "master_port": master_port,
        "rank": rank,
        "local_rank": local_rank,
    }

    return params


class EarlyStopping:
    """
    Simple early stopping utility for training loops.

    This class monitors validation loss and stops training when it has not
    improved for a given number of consecutive epochs (``patience``).
    A minimum required improvement between epochs can be controlled via
    ``min_delta``.

    Parameters
    ----------
    patience : int, default=3
        Number of consecutive epochs with no improvement in validation loss
        before stopping is triggered.

    min_delta : float, default=0.0
        Minimum change in validation loss required to consider it an
        improvement. Smaller or equal improvements are treated as no progress.

    Attributes
    ----------
    best_loss : float
        Best (lowest) validation loss observed so far.

    counter : int
        Number of epochs since the last improvement.

    Methods
    -------
    initialize()
        Reset internal state (`best_loss` and `counter`) before training.

    __call__(val_loss)
        Alias to :meth:`step`. Evaluates a new loss value and returns whether
        early stopping should trigger.

    step(val_loss)
        Update best loss and internal counter with a given validation loss,
        returning ``True`` when patience is exhausted and training should stop.

    Examples
    --------
    >>> early_stop = EarlyStopping(patience=5, min_delta=0.01)
    >>> for epoch in range(100):
    ...     val_loss = compute_validation_loss()
    ...     if early_stop(val_loss):
    ...         print("Early stopping triggered.")
    ...         break
    """

    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def initialize(self):
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss):
        return self.step(val_loss)

    def step(self, val_loss):
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def load_epochs(files, concat=False):
    import mne

    epochs_list = list()
    for file in files:
        epochs = mne.read_epochs(file)
        epochs_list.append(epochs)
    if concat:
        return tm.concatenate_epochs(epochs_list)
    else:
        return epochs_list


def nd_to_tensor(X_train, y_train, X_valid, y_valid, X_test, y_test, device="cpu"):
    import torch

    X_train_tensor = torch.tensor(X_train, dtype=torch.float).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64).to(device)

    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float).to(device)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.int64).to(device)

    if isinstance(X_test, list):
        X_test_tensor = [torch.tensor(X, dtype=torch.float).to(device) for X in X_test]
        y_test_tensor = [torch.tensor(y, dtype=torch.int64).to(device) for y in y_test]
    else:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float).to(device)
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
    import torch

    dataset_train = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataset_valid = torch.utils.data.TensorDataset(X_valid_tensor, y_valid_tensor)

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
        enable_DS=False,
        DS_params=None,
        generator=None,
):
    """
    Convert dataset to Dataloader

    Parameters
    ----------
    generator: None, instance of torch.Generator(), or int
        if None, no generator is set. if instance of torch.Genetator(), it will be used. If int, torch.Generator.manual_seed(generator) will be used.
    enable_DS: bool
        if True, DistributedSampler will be used. Note that you must call sampler_train.set_epoch(epoch) every epoch.

    """
    import torch

    if generator is None:
        g = None
        func_worker_init = None
    elif isinstance(generator, int):
        g = torch.Generator()
        g.manual_seed(generator)

        def func_worker_init(worker_id):
            worker_seed = generator + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

    elif isinstance(generator, torch.Generator):
        g = generator

        def func_worker_init(worker_id):
            worker_seed = g.initial_seed() + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

    else:
        raise ValueError("generator must be an int or torch.Generator")

    if enable_DS:
        world_size = DS_params["world_size"]
        num_workers = DS_params["num_workers"]
        rank = DS_params["rank"]

        persistent_workers = num_workers > 0

        sampler_train = torch.utils.data.distributed.DistributedSampler(
            dataset_train,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=generator,
        )

        sampler_valid = torch.utils.data.distributed.DistributedSampler(
            dataset_valid, num_replicas=world_size, rank=rank, shuffle=False
        )

        sampler_test = torch.utils.data.distributed.DistributedSampler(
            dataset_test, num_replicas=world_size, rank=rank, shuffle=False
        )

        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            sampler=sampler_train,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            generator=g,
            worker_init_fn=func_worker_init,
        )

        dataloader_valid = torch.utils.data.DataLoader(
            dataset_valid,
            batch_size=batch_size,
            sampler=sampler_valid,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )

        if isinstance(dataset_test, list):
            dataloader_test = [
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    sampler=sampler_test,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=persistent_workers,
                )
                for dataset in dataset_test
            ]
        else:
            dataloader_test = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=batch_size,
                sampler=sampler_test,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=persistent_workers,
            )

        return dataloader_train, dataloader_valid, dataloader_test, sampler_train

    else:

        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            worker_init_fn=func_worker_init,
        )
        dataloader_valid = torch.utils.data.DataLoader(
            dataset_valid, batch_size=batch_size, shuffle=False
        )

        if isinstance(dataset_test, list):
            dataloader_test = [
                torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=False
                )
                for dataset in dataset_test
            ]
        else:
            dataloader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=batch_size, shuffle=False
            )

        return dataloader_train, dataloader_valid, dataloader_test


def nd_to_dataloader(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        batch_size,
        device="cpu",
        enable_DS=False,
        DS_params=None,
        generator=None,
):
    (
        X_train_tensor,
        y_train_tensor,
        X_valid_tensor,
        y_valid_tensor,
        X_test_tensor,
        y_test_tensor,
    ) = nd_to_tensor(X_train, y_train, X_valid, y_valid, X_test, y_test, device=device)

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
        batch_size=batch_size,
        enable_DS=enable_DS,
        DS_params=DS_params,
        generator=generator,
    )


def get_predictions(model, dataloader, device="cpu"):
    """
    Run inference on a dataloader and return predictions, labels, logits and class probabilities.

    The model is evaluated in no-grad mode and expected to output raw logits.
    Predictions are computed via ``argmax``, while probabilities are obtained
    by applying a softmax over the class dimension.

    Parameters
    ----------
    model : torch.nn.Module
        A trained PyTorch model used for inference.
        Must return logits of shape ``(batch_size, n_classes)``.

    dataloader : torch.utils.data.DataLoader
        Dataloader yielding batches of input tensors ``X`` and labels ``y``.

    device : {"cpu", "cuda"}, default="cpu"
        Device used for running forward passes.

    Returns
    -------
    preds : np.ndarray of shape (N,)
        Predicted class indices for all samples in the dataloader.

    labels : np.ndarray of shape (N,)
        Ground-truth labels.

    logits : np.ndarray of shape (N, n_classes)
        Raw model outputs (pre-softmax).

    probas : np.ndarray of shape (N, n_classes)
        Softmax-normalized class probabilities.

    Notes
    -----
    - The model is automatically set to evaluation mode.
    - Computes outputs without gradient tracking for faster inference.
    - Outputs are concatenated across all batches and returned as NumPy arrays.
    - Order of returned values is:
      ``preds, labels, logits, probas``.

    Examples
    --------
    >>> preds, labels, logits, probas = get_predictions(model, dataloader, device="cuda")
    >>> preds[:5]
    array([1, 0, 1, 1, 2])
    """
    import torch

    model.eval()

    preds_list = list()
    labels_list = list()
    logits_list = list()
    probas_list = list()

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            probas = torch.nn.functional.softmax(logits, dim=1)

            logits_list.append(logits)
            preds_list.append(preds)
            labels_list.append(y)
            probas_list.append(probas)

    logits_list = torch.cat(logits_list).cpu().numpy()
    preds_list = torch.cat(preds_list).cpu().numpy()
    labels_list = torch.cat(labels_list).cpu().numpy()
    probas_list = torch.cat(probas_list).cpu().numpy()

    return preds_list, labels_list, logits_list, probas_list


"""
def accuracy_score_dataloader_DPP(model, dataloader, criterion=None, device="cpu"):
    import torch

    total_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_pred = model(X)
            preds = torch.argmax(y_pred, dim=1)
            if criterion is not None:
                loss = criterion(y_pred, y)
                total_loss += loss.item() * y.size(0)
            correct += (preds == y).sum().item()
            total += y.size(0)

    total_loss_tensor = torch.tensor(total_loss, device=device)
    correct_tensor = torch.tensor(correct, device=device)
    total_tensor = torch.tensor(total, device=device)

    torch.distributed.all_reduce(total_loss_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)

    loss_avg = total_loss_tensor.item() / total_tensor.item()
    acc = correct_tensor.item() / total_tensor.item()

    if criterion is not None:
        return acc, loss_avg
    else:
        return acc
"""


def evaluation_dataloader(
        model, dataloader, criterion=None, device="cpu", enable_ddp=False
):
    import torch

    total_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_pred = model(X)
            preds = torch.argmax(y_pred, dim=1)
            if criterion is not None:
                loss = criterion(y_pred, y)
                total_loss += loss.item() * y.size(0)
            correct += (preds == y).sum().item()
            total += y.size(0)

    if enable_ddp:
        total_loss_tensor = torch.tensor(total_loss, device=device)
        correct_tensor = torch.tensor(correct, device=device)
        total_tensor = torch.tensor(total, device=device)

        torch.distributed.all_reduce(
            total_loss_tensor, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)

        loss_avg = total_loss_tensor.item() / total_tensor.item()
        acc = correct_tensor.item() / total_tensor.item()

        if criterion is not None:
            return acc, loss_avg
        else:
            return acc
    else:
        acc = correct / total

        if criterion is not None:
            loss_avg = total_loss / total
            return acc, loss_avg
        else:
            return acc


def train_epoch(
        model,
        criterion,
        optimizer,
        dataloader_train,
        dataloader_valid,
        epoch,
        device="cpu",
        loss_best=None,
        history=None,
        scheduler=None,
        checkpoint_fname=None,
        enable_wandb=True,
        enable_ddp=False,
        enable_dp=False,
        rank=0,
):
    import torch

    # if enable_ddp:
    #    rank = int(os.environ["RANK"])
    # else:
    #    rank = 0

    tic = time.time()

    # train
    model.train()
    for X, y in dataloader_train:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        y_pred = model(X)
        loss = criterion(y_pred, y)
        # train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # valid
    model.eval()
    with torch.no_grad():
        train_acc, train_loss = evaluation_dataloader(
            model=model,
            dataloader=dataloader_train,
            criterion=criterion,
            device=device,
            enable_ddp=enable_ddp,
        )
        valid_acc, valid_loss = evaluation_dataloader(
            model=model,
            dataloader=dataloader_valid,
            criterion=criterion,
            device=device,
            enable_ddp=enable_ddp,
        )

    txt_print = f"epoch {epoch:03}, train_loss: {train_loss:06.4f}, train_acc: {train_acc:.2f}, valid_loss: {valid_loss:06.4f}, valid_acc: {valid_acc:.2f}"

    if scheduler is not None:
        scheduler.step()
        _lr = scheduler.get_last_lr()[0]

        txt_print += f", lr: {_lr:.4e}"

    toc = time.time()
    txt_print += f", et: {toc - tic:.4f}"

    # save history
    if history is not None and rank == 0:
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)

    # save model if loss was the lowest
    if checkpoint_fname is not None and rank == 0:
        if valid_loss < loss_best["value"]:
            checkpoint = dict()
            checkpoint["epoch"] = epoch
            checkpoint["model_state_dict"] = (
                model.module.state_dict()
                if enable_ddp or enable_dp
                else model.state_dict()
            )
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            checkpoint["valid_loss"] = valid_loss
            # torch.save(checkpoint, f"{checkpoint_fname}.pth")
            torch.save(checkpoint, checkpoint_fname)

            loss_best["value"] = valid_loss

            txt_print += ", checkpoint saved"

    # send log to wandb
    if enable_wandb and rank == 0:
        import wandb

        wandb.log(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
            }
        )

    # print log
    if rank == 0:
        print(txt_print)

    return valid_loss
