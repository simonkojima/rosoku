import numpy as np
import scipy
import tqdm


def _euclidean_alignment(X):
    n_trials, n_channels, _ = X.shape

    covariances = np.zeros((n_trials, n_channels, n_channels))
    for n in range(n_trials):
        covariances[n, :, :] = X[n, :, :] @ X[n, :, :].T

    R = np.mean(covariances, axis=0)

    R_inv_sqrt = scipy.linalg.fractional_matrix_power(R, -0.5)

    X_aligned = np.zeros(X.shape)
    for n in range(n_trials):
        X_aligned[n, :, :] = R_inv_sqrt @ X[n, :, :]

    return X_aligned


def euclidean_alignment(X, online=False, enable_tqdm=True):
    """
    Perform Euclidean Alignment (EA) on EEG trials.

    EA (Euclidean Alignment) is a preprocessing method that recenters EEG signals
    in Euclidean space by aligning them to the mean covariance structure of the data.
    This reduces inter-session and inter-subject distribution shifts and is widely
    used for transfer learning in BCI.

    Parameters
    ----------
    X : np.ndarray
        EEG data with shape ``(n_trials, n_channels, n_times)``.
        EA is applied along the trial axis.

    online : bool, optional
        If ``False`` (default), EA is computed in *offline* mode:
            - EA is performed once using **all trials**.
            - All trials are aligned using the global mean covariance.

        If ``True``, EA runs in *online (causal) mode*:
            - For the n-th trial, EA uses **only the first n trials** to estimate
              the alignment matrix.
            - This simulates an online BCI scenario and enables causal
              domain adaptation, where future data are not used.

        Example:
        - ``online=False`` → standard offline transfer learning
        - ``online=True`` → online / streaming BCI, causal domain adaptation

    enable_tqdm : bool, optional
        Whether to display a progress bar during online processing.

    Returns
    -------
    X_aligned : np.ndarray
        EA-transformed data with the same shape as the input
        ``(n_trials, n_channels, n_times)``.

    References
    ----------
    H. He and D. Wu, "Transfer Learning for Brain–Computer Interfaces:
    A Euclidean Space Data Alignment Approach," IEEE Transactions on
    Biomedical Engineering, vol. 67, no. 2, pp. 399–410, 2020.
    DOI: `10.1109/TBME.2019.2913914 <https://doi.org/10.1109/TBME.2019.2913914>`_

    """

    if online is False:
        return _euclidean_alignment(X=X)
    else:
        new_X = []
        for n in tqdm.tqdm(range(X.shape[0]), disable=not enable_tqdm):
            new_X.append(_euclidean_alignment(X[0 : (n + 1), :, :])[-1, :, :])
        return np.stack(new_X, axis=0)


def _riemannian_alignment(covariances, scaling=False):
    import pyriemann

    mean_cov = pyriemann.utils.mean.mean_covariance(covariances)
    n_covs, _, _ = covariances.shape

    if scaling:
        d_tilda_sqrt = 0
        for m in range(n_covs):
            d_tilda_sqrt += pyriemann.utils.distance.distance(
                mean_cov, covariances[m, :, :], squared=True
            )
        d_tilda = np.sqrt(d_tilda_sqrt)

    mean_cov_inv_sqrt = scipy.linalg.fractional_matrix_power(mean_cov, -0.5)

    aligned_covs = np.zeros(covariances.shape)
    for m in range(n_covs):
        aligned_covs[m, :, :] = (
            mean_cov_inv_sqrt @ covariances[m, :, :] @ mean_cov_inv_sqrt
        )
        if scaling:
            aligned_covs[m, :, :] = scipy.linalg.fractional_matrix_power(
                aligned_covs[m, :, :], 1 / d_tilda
            )

    return aligned_covs


def riemannian_alignment(covariances, scaling=False, online=False, enable_tqdm=True):
    """
    Perform Riemannian Alignment (Riemannian Procrustes Analysis, RPA) on
    covariance matrices.

    This function applies the **re-centering** step of RPA, and optionally the
    **re-scaling** step, on symmetric positive definite (SPD) covariance matrices
    on the Riemannian manifold. Riemannian Alignment reduces inter-session and
    inter-subject variability and is widely used in Riemannian-based BCI decoding.

    RPA consists of three conceptual steps:
        1. **Re-centering** (unsupervised)
        2. **Re-scaling** (unsupervised)
        3. **Rotation** (supervised)

    This implementation includes:
        - **Re-centering**
        - **Optional re-scaling**

    The rotation step is supervised, because it requires class-wise means
    and alignment across classes. It cannot be applied in an unsupervised way;
    therefore it is intentionally omitted here. For full RPA, refer to the
    original publication.

    Parameters
    ----------
    covariances : np.ndarray
        Array of SPD covariance matrices with shape
        ``(n_trials, n_channels, n_channels)``.

    scaling : bool, optional
        Whether to apply the re-scaling step after re-centering.
        - ``False`` (default): perform only re-centering
        - ``True``: perform re-centering + re-scaling

    online : bool, optional
        If ``False`` (default), alignment is computed in *offline* mode:
            - RPA is computed once using **all trials**.

        If ``True``, RPA runs in *online (causal) mode*:
            - For the n-th trial, alignment uses **only the first n trials**.
            - Enables causal / streaming BCI adaptation without using future data.

        This mirrors online BCI scenarios and online domain adaptation settings.

    enable_tqdm : bool, optional
        Whether to display a progress bar during online processing.

    Returns
    -------
    cov_aligned : np.ndarray
        Aligned covariance matrices with shape
        ``(n_trials, n_channels, n_channels)``.

    References
    ----------
    P. L. C. Rodrigues, C. Jutten and M. Congedo,
    "Riemannian Procrustes Analysis: Transfer Learning for Brain–Computer Interfaces,"
    IEEE Transactions on Biomedical Engineering, vol. 66, no. 8, pp. 2390–2401, 2019.
    DOI: `10.1109/TBME.2018.2889705 <https://doi.org/10.1109/TBME.2018.2889705>`_
    """
    if online is False:
        return _riemannian_alignment(covariances=covariances, scaling=scaling)
    else:
        new_covariances = []
        for n in tqdm.tqdm(range(covariances.shape[0]), disable=not enable_tqdm):
            if n == 0:
                _scaling = False
            else:
                _scaling = scaling
            new_covariances.append(
                _riemannian_alignment(covariances[0 : (n + 1), :, :], scaling=_scaling)[
                    -1, :, :
                ]
            )
        return np.stack(new_covariances, axis=0)
