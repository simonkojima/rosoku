import numpy as np
import tag_mne as tm


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


def normalize(X_train, X_valid, X_test, return_params=False):
    """
    Standardization with training set stats

    X.shape: (n_trials, n_channels, n_times)

    X_test: list or nd.array

    """

    n_trials, n_channels, n_times = X_train.shape

    mean = np.mean(X_train.transpose((1, 0, 2)).reshape((n_channels, -1)), axis=1)
    std = np.std(X_train.transpose((1, 0, 2)).reshape((n_channels, -1)), axis=1)

    mean = np.expand_dims(mean, axis=(1, 2)).transpose((1, 0, 2))
    std = np.expand_dims(std, axis=(1, 2)).transpose((1, 0, 2))

    X_train = (X_train - mean) / std
    X_valid = (X_valid - mean) / std

    X_test_normalized = []
    if isinstance(X_test, list):
        for X in X_test:
            X = (X - mean) / std
            X_test_normalized.append(X)
        X_test = X_test_normalized
    else:
        X_test = (X_test - mean) / std

    if return_params:
        return X_train, X_valid, X_test, mean, std
    else:
        return X_train, X_valid, X_test


def normalize_tensor(X_train_tensor, X_valid_tensor, X_test_tensor):
    """
    Standardization with training set stats

    X.shape: (n_trials, n_channels, n_times)

    X_test_tensor: list

    """

    n_trials, n_channels, n_times = X_train_tensor.shape

    mean = X_train_tensor.transpose(1, 2).reshape(-1, n_channels).mean(dim=0)
    std = X_train_tensor.transpose(1, 2).reshape(-1, n_channels).std(dim=0)

    print("mean", mean.size(), mean)
    print("std", std.size(), std)

    X_train_tensor = (X_train_tensor - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(
        0
    ).unsqueeze(2)
    X_valid_tensor = (X_valid_tensor - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(
        0
    ).unsqueeze(2)

    X_test_tensor_normalized = list()
    if isinstance(X_test_tensor, list):
        for X in X_test_tensor:
            X = (X - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)
            X_test_tensor_normalized.append(X)
        X_test_tensor = X_test_tensor_normalized
    else:
        X_test_tensor = (
            X_test_tensor - mean.unsqueeze(0).unsqueeze(2)
        ) / std.unsqueeze(0).unsqueeze(2)

    return X_train_tensor, X_valid_tensor, X_test_tensor


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


def dataset_to_dataloader(dataset_train, dataset_valid, dataset_test, batch_size):
    import torch

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False
    )

    if isinstance(dataset_test, list):
        dataloader_test = [
            torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
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

    (dataloader_train, dataloader_valid, dataloader_test) = dataset_to_dataloader(
        dataset_train, dataset_valid, dataset_test, batch_size=batch_size
    )

    return dataloader_train, dataloader_valid, dataloader_test


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
    if history is not None:
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)

    # save model if loss was the lowest
    if checkpoint_fname is not None:
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
