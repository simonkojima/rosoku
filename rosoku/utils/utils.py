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


def normalize_tensor(*args, **kwargs):
    from .. import preprocessing

    import warnings

    warnings.simplefilter("default", DeprecationWarning)

    warnings.warn(
        "rosoku.utils.normalize_tensor() is deprecated and will be removed in a future version. Use rosoku.preprocessing.normalize_tensor() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # return preprocessing.normalize(X_train, X_valid, X_test, return_params)
    return preprocessing.normalize_tensor(*args, **kwargs)


##


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


def get_labels_from_epochs(epochs, label_keys={"event:left": 0, "event:right": 1}):
    y = list()

    _, markers = tm.markers_from_events(epochs.events, epochs.event_id)

    for marker in markers:
        for key, val in label_keys.items():
            if key in marker:
                y.append(val)

    return y


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
):
    import torch

    if enable_DS:
        world_size = DS_params["world_size"]
        num_workers = DS_params["num_workers"]
        rank = DS_params["rank"]

        sampler_train = torch.utils.data.distributed.DistributedSampler(
            dataset_train, num_replicas=world_size, rank=rank, shuffle=True
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
        )

        dataloader_valid = torch.utils.data.DataLoader(
            dataset_valid,
            batch_size=batch_size,
            sampler=sampler_valid,
            num_workers=num_workers,
        )

        if isinstance(dataset_test, list):
            dataloader_test = [
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    sampler=sampler_test,
                    num_workers=num_workers,
                )
                for dataset in dataset_test
            ]
        else:
            dataloader_test = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=batch_size,
                sampler=sampler_test,
                num_workers=num_workers,
            )

        return dataloader_train, dataloader_valid, dataloader_test, sampler_train

    else:

        dataloader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True
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
    )


def get_predictions(model, dataloader):
    import torch

    model.eval()

    preds_list = list()
    labels_list = list()
    logits_list = list()
    probas_list = list()

    with torch.no_grad():
        for X, y in dataloader:
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


def accuracy_score_dataloader(model, dataloader, criterion=None):
    import torch

    total_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            y_pred = model(X)
            preds = torch.argmax(y_pred, dim=1)
            if criterion is not None:
                loss = criterion(y_pred, y)
                total_loss += loss.item() * y.size(0)
            correct += (preds == y).sum().item()
            total += y.size(0)
        acc = correct / total if total > 0 else 0

    if criterion is not None:
        avg_loss = total_loss / total if total > 0 else 0
        return acc, avg_loss
    else:
        return acc


class EarlyStopping:
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


def train_epoch(
    model,
    criterion,
    optimizer,
    dataloader_train,
    dataloader_valid,
    epoch,
    loss_best=None,
    history=None,
    scheduler=None,
    checkpoint_fname=None,
    enable_wandb=True,
    rank=0,
):
    import torch

    # train
    model.train()
    train_loss = 0
    for X, y in dataloader_train:
        y_pred = model(X)
        loss = criterion(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # valid
    model.eval()
    with torch.no_grad():
        train_acc, train_loss = accuracy_score_dataloader(
            model, dataloader_train, criterion
        )
        valid_acc, valid_loss = accuracy_score_dataloader(
            model, dataloader_valid, criterion
        )

    txt_print = f"epoch {epoch:03}, train_loss: {train_loss:06.4f}, train_acc: {train_acc:.2f}, valid_loss: {valid_loss:06.4f}, valid_acc: {valid_acc:.2f}"

    if scheduler is not None:
        scheduler.step()
        _lr = scheduler.get_last_lr()[0]

        txt_print += f", lr: {_lr:.4e}"

    # save history
    if history is not None and rank == 0:
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)

    # save model if loss was the lowest
    if checkpoint_fname is not None and rank == 0:
        if valid_loss < loss_best[0]:
            checkpoint = dict()
            checkpoint["epoch"] = epoch
            checkpoint["model_state_dict"] = model.state_dict()
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            checkpoint["loss"] = loss
            torch.save(checkpoint, f"{checkpoint_fname}.pth")

            loss_best[0] = valid_loss

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
