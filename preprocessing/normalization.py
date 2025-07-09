import numpy as np


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
