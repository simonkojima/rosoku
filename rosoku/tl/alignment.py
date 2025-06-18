import numpy as np
import scipy


def euclidean_alignment(X):
    """
    EA (Euclidean Alignment)を行う関数
    recenteringがユークリッド空間で行われる

    X: nd.array
        shape of (n_trials, n_channels, n_times)

    References:
    H. He and D. Wu, "Transfer Learning for Brain–Computer Interfaces: A Euclidean Space Data Alignment Approach," in IEEE Transactions on Biomedical Engineering, vol. 67, no. 2, pp. 399-410, Feb. 2020, doi: 10.1109/TBME.2019.2913914.
    """

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


def riemannian_alignment(covariances, scaling=False):
    """
    Riemannian Procrustes Analysisのrecentering, rescalingを行う関数
    scaling = Falseのとき，recenteringのみ

    covariances: nd.array
        shape of (n_trials, n_channels, n_channels)
    scaling: bool
        rescalingを行うかどうか

    References:
    P. L. C. Rodrigues, C. Jutten and M. Congedo, "Riemannian Procrustes Analysis: Transfer Learning for Brain–Computer Interfaces," in IEEE Transactions on Biomedical Engineering, vol. 66, no. 8, pp. 2390-2401, Aug. 2019, doi: 10.1109/TBME.2018.2889705.

    """
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
