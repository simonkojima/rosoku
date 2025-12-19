import os
import random
import time
import numpy as np


def _add_values_to_df(df, values):
    for key, value in values.items():
        df[key] = [value for _ in range(len(df))]
    return df


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


def get_predictions(
        model,
        dataloader,
        device="cpu",
        callback_get_logits=None,
        callback_get_probas=None,
        callback_get_preds=None,
):
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

            if callback_get_logits is None:
                logits = model(X)
            else:
                logits = callback_get_logits(model, X)

            if callback_get_preds is None:
                preds = torch.argmax(logits, dim=1)
            else:
                preds = callback_get_preds(model, X)

            if callback_get_probas is None:
                probas = torch.nn.functional.softmax(logits, dim=1)
            else:
                probas = callback_get_probas(model, X)

            logits_list.append(logits)
            preds_list.append(preds)
            labels_list.append(y)
            probas_list.append(probas)

    logits_list = torch.cat(logits_list).cpu().numpy()
    preds_list = torch.cat(preds_list).cpu().numpy()
    labels_list = torch.cat(labels_list).cpu().numpy()
    probas_list = torch.cat(probas_list).cpu().numpy()

    return preds_list, labels_list, logits_list, probas_list


def evaluation_dataloader(model, dataloader, criterion=None, device="cpu"):
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

    acc = correct / total

    if criterion is not None:
        loss_avg = total_loss / total
        return acc, loss_avg
    else:
        return acc


def _train_epoch(
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
        enable_wandb=False,
):
    import torch

    tic = time.time()

    # train
    model.train()
    for X, y in dataloader_train:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        y_pred = model(X)
        loss = criterion(y_pred, y)

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
        )
        valid_acc, valid_loss = evaluation_dataloader(
            model=model,
            dataloader=dataloader_valid,
            criterion=criterion,
            device=device,
        )

    txt_print = f"epoch {epoch:03}, train_loss: {train_loss:06.4f}, train_acc: {train_acc:.2f}, valid_loss: {valid_loss:06.4f}, valid_acc: {valid_acc:.2f}"

    if scheduler is not None:
        scheduler.step()
        _lr = scheduler.get_last_lr()[0]

        txt_print += f", lr: {_lr:.4e}"

    toc = time.time()
    txt_print += f", et: {toc - tic:.4f}"

    # save history
    if history is not None:
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)

    # save model if loss was the lowest
    if checkpoint_fname is not None:
        if valid_loss < loss_best["value"]:
            checkpoint = dict()
            checkpoint["epoch"] = epoch
            checkpoint["model_state_dict"] = model.state_dict()
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            checkpoint["valid_loss"] = valid_loss
            torch.save(checkpoint, checkpoint_fname)

            loss_best["value"] = valid_loss

            txt_print += ", checkpoint saved"

    # send log to wandb
    if enable_wandb:
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
    print(txt_print)

    return valid_loss
