import numpy as np


def normalize(X_train, X_valid, X_test, return_params=True):
    """
    Z-score normalization across channels for EEG-style tensors.

    Normalization parameters (mean, std) are computed only from ``X_train``
    and applied to validation and test data. This ensures no data leakage
    during evaluation.

    The normalization is applied independently to each channel.
    For a given data :math:`X^c` at channel :math:`c`, the normalization is defined as:

    .. math::

        \\mu =
        \\frac{1}{N}
        \\sum_{i=1}^{N} \\frac{1}{T} \\sum_{t=1}^{T} X_{i,t}^c

    .. math::

        \\sigma =
        \\sqrt{
            \\frac{1}{N}
            \\sum_{i=1}^{N} \\frac{1}{T} \\sum_{t=1}^{T}
            \\left( X_{i,t}^c - \\mu \\right)^2
        }

    .. math::

        \\hat{X}_{i,t}^c =
        \\frac{X_{i,t}^c - \\mu}{\\sigma}

    where the same normalization rule is applied independently
    to each channel :math:`c = 1, \\ldots, C`.

    Parameters
    ----------
    X_train : np.ndarray
        Training data with shape ``(n_trials, n_channels, n_times)``.
        Used to compute the normalization mean and std.

    X_valid : np.ndarray
        Validation data with shape ``(n_trials, n_channels, n_times)``.

    X_test : np.ndarray or list of np.ndarray
        Test data with shape ``(n_trials, n_channels, n_times)``
        or a list of such arrays (e.g., multiple test groups).

    return_params : bool, default=True
        If ``True``, also return the computed ``mean`` and ``std`` arrays.

    Returns
    -------
    X_train_norm : np.ndarray
        Normalized training data with same shape as ``X_train``.
    X_valid_norm : np.ndarray
        Normalized validation data.
    X_test_norm : np.ndarray or list of np.ndarray
        Normalized test data. The output type matches the input type
        (single ndarray or list of ndarrays).
    mean : np.ndarray, optional
        Mean used for normalization. Returned only when
        ``return_params=True``.
    std : np.ndarray, optional
        Standard deviation used for normalization. Returned only when
        ``return_params=True``.

    Notes
    -----
    - Input tensors must follow shape ``(n_trials, n_channels, n_times)``.
    - Mean and std are computed per-channel across all trials and time samples.
    - ``mean``/``std`` are reshaped to ``(1, n_channels, 1)``
      for broadcasting during normalization.
    - Test sets do **not** influence normalization statistics.

    """

    n_trials, n_channels, n_times = X_train.shape

    mean = np.mean(X_train.transpose((1, 0, 2)).reshape((n_channels, -1)), axis=1)
    std = np.std(X_train.transpose((1, 0, 2)).reshape((n_channels, -1)), axis=1)

    mean = np.expand_dims(mean, axis=(1, 2)).transpose((1, 0, 2))
    std = np.expand_dims(std, axis=(1, 2)).transpose((1, 0, 2))

    X_train = (X_train - mean) / std
    if X_valid is not None:
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
