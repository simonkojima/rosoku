import time
import json

import numpy as np
import scipy

import mne
import pyriemann
import sklearn

import tag_mne as tm
import pandas as pd

from . import utils


def func_proc_epochs(epochs, tmin=0.5, tmax=4.5):
    epochs = epochs.pick(picks="eeg").crop(tmin=tmin, tmax=tmax)
    return epochs


def load_epochs(files, concat=False):
    epochs_list = list()
    for file in files:
        epochs = mne.read_epochs(file)
        epochs_list.append(epochs)
    if concat:
        return tm.concatenate_epochs(epochs_list)
    else:
        return epochs_list


def recenter_cov(cov_session, scaling=False):
    mean_cov_session = pyriemann.utils.mean.mean_covariance(cov_session)
    n_covs, _, _ = cov_session.shape

    if scaling:
        d_tilda_sqrt = 0
        for m in range(n_covs):
            d_tilda_sqrt += pyriemann.utils.distance.distance(
                mean_cov_session, cov_session[m, :, :], squared=True
            )
        d_tilda = np.sqrt(d_tilda_sqrt)

    mean_cov_session_inv_sqrt = scipy.linalg.fractional_matrix_power(
        mean_cov_session, -0.5
    )

    recentered_covs = np.zeros(cov_session.shape)
    for m in range(n_covs):
        recentered_covs[m, :, :] = (
                mean_cov_session_inv_sqrt @ cov_session[m, :, :] @ mean_cov_session_inv_sqrt
        )
        if scaling:
            recentered_covs[m, :, :] = scipy.linalg.fractional_matrix_power(
                recentered_covs[m, :, :], 1 / d_tilda
            )

    return recentered_covs


def conventional(
        keywords_train,
        keywords_test,
        func_load_epochs=None,
        func_load_ndarray=None,
        func_proc_epochs=None,
        func_proc_ndarray=None,
        func_proc_mode="per_split",
        classifiers=[
            pyriemann.classification.TSClassifier(),
            pyriemann.classification.MDM(),
        ],
        classifier_names=["tslr", "mdm"],
        func_convert_epochs_to_ndarray=utils.convert_epochs_to_ndarray,
        samples_fname=None,
        additional_values=None,
):
    """
    General-purpose pipeline for conventional (non-deep-learning) classifiers,
    especially Riemannian-based methods.

    This function provides a flexible interface for loading data, preprocessing,
    fitting multiple classifiers, evaluating performance, and exporting results.

    ---------------------------------------------------------------------------
    Data loading via “keywords” and “mode”
    ---------------------------------------------------------------------------
    ``keywords_train`` and ``keywords_test`` are arbitrary user-defined objects
    (typically dicts) that specify how data should be loaded. They are passed,
    together with a ``mode`` string, to the callback functions
    ``func_load_epochs`` or ``func_load_ndarray``.

    - First argument:  ``keyword`` (one element of ``keywords_train``/``keywords_test``)
    - Second argument: ``mode`` ∈ {"train", "test"}

    Example
    -------
    .. code-block:: python

        def func_load_epochs(keyword, mode):
            subject = keyword["subject"]
            session = keyword["session"]
            fname = f"sub-{subject}_ses-{session}-epo.fif"
            epochs = mne.read_epochs(fname)

            if mode == "train":
                # optional: apply additional preprocessing for training data
                epochs = epochs.crop(tmin=0.0, tmax=1.0)

            return epochs

    ---------------------------------------------------------------------------
    Grouping test data
    ---------------------------------------------------------------------------
    ``keywords_test`` controls how test data are grouped for evaluation.

    - ``[[a], [b]]``  → evaluate a and b **separately**
    - ``[[a, b]]``    → **merge** the data associated with a and b
                        and evaluate them together

    This allows flexible control over whether each test set is evaluated
    individually or jointly.

    ---------------------------------------------------------------------------

    Parameters
    ----------
    keywords_train : list
        List of keyword objects used to load training data.

    keywords_test : list of list
        Controls grouping of test data.
        Each inner list corresponds to one test evaluation group.

    func_load_epochs : callable, optional
        Callback function for loading data as MNE ``Epochs`` objects. It must accept:

        .. code-block:: python

            def func_load_epochs(keyword, mode):
                ...

        where

        - ``keyword`` is one element from ``keywords_train`` or ``keywords_test``
        - ``mode`` is either ``"train"`` or ``"test"``

        and it must return an ``mne.Epochs`` instance.

    func_load_ndarray : callable, optional
        Callback function for loading data as NumPy arrays. It must accept:

        .. code-block:: python

            def func_load_ndarray(keyword, mode):
                ...

        and return a tuple ``(X, y)`` where ``X`` and ``y`` are NumPy arrays.

    func_proc_epochs : callable, optional
        Function that receives an ``mne.Epochs`` object and returns a processed one
        (e.g., channel selection, cropping, filtering).

    func_proc_ndarray : callable, optional
        Preprocessing function for NumPy data.

    func_proc_mode : {"per_split", "all"}
        Defines whether preprocessing is applied independently to each split
        or jointly across all splits.

    classifiers : list of estimator
        List of classifier instances implementing ``fit`` and ``predict``
        (optionally ``predict_proba``). By default, Riemannian classifiers
        from pyRiemann are used.

    classifier_names : list of str
        Names associated with each classifier (used in the output DataFrame).
        Must have the same length as ``classifiers``.

    func_convert_epochs_to_ndarray : callable
        Converter from MNE Epochs to NumPy arrays, used internally by
        :func:`utils.load_data`.

    samples_fname : path-like, optional
        File path for saving sample-level predictions (Parquet format).

    additional_values : dict, optional
        Extra key–value pairs appended as columns to the output DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing classification results (accuracy per classifier
        and test group), along with metadata such as train/test keywords,
        classifier name, and optional description.
    """

    # load data

    X_train, _, X_test, y_train, _, y_test = utils.load_data(
        keywords_train=keywords_train,
        keywords_valid=None,
        keywords_test=keywords_test,
        func_load_epochs=func_load_epochs,
        func_load_ndarray=func_load_ndarray,
        func_proc_epochs=func_proc_epochs,
        func_proc_ndarray=func_proc_ndarray,
        func_proc_mode=func_proc_mode,
        func_convert_epochs_to_ndarray=func_convert_epochs_to_ndarray,
    )

    # train classifiers
    for clf in classifiers:
        clf.fit(X_train, y_train)

    # classify test data and evaluate results

    if isinstance(X_test, list) is False:
        X_test = [X_test]
        y_test = [y_test]

    if len(keywords_test) != len(X_test):
        raise RuntimeError("len(keywords_test) != len(X_test)")

    df_list = []
    samples_list = []
    for X, y, keywords in zip(X_test, y_test, keywords_test):
        for clf, name in zip(classifiers, classifier_names):

            df_results = pd.DataFrame()

            preds = clf.predict(X)
            probas = clf.predict_proba(X)
            accuracy = sklearn.metrics.accuracy_score(y, preds)

            df_results["keywords_train"] = [json.dumps(keywords_train)]
            df_results["keywords_test"] = [json.dumps(keywords)]
            df_results["classifier"] = [name]
            df_results["accuracy"] = [accuracy]

            samples = pd.DataFrame()
            samples["labels"] = y
            samples["preds"] = preds
            for idx in range(probas.shape[1]):
                samples[f"probas_{idx}"] = probas[:, idx]
            samples["classifier"] = [name for _ in range(len(samples))]

            samples_list.append(samples)
            df_list.append(df_results)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    if additional_values is not None:
        for key, value in additional_values.items():
            df[key] = [value for m in range(df.shape[0])]

    if samples_fname is not None:
        samples = pd.concat(samples_list, axis=0, ignore_index=True)
        samples.to_parquet(samples_fname)

    return df
