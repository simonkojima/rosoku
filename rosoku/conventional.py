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
        # compile_test=False,
        samples_fname=None,
        desc=None,
        additional_values=None,
):
    """
    汎用的なriemannian用関数

    Parameters
    ----------
    kewords_train: list
    kewords_test: list
    func_load_epochs: callable
            第一引数がkeywords, 第二引数がmodeの，mne.Epochsオブジェクトを返す関数．
            keywordsはkeywords_trainやkeywords_testで渡されたlistオブジェクトが渡される．
            modeは，train, testのstrが渡される．

            .. code-block:: python

                def func_load_epochs(keywords, mode):
                    # keywords: keywords_train or keywords_test
                    # mode: "train" or "test"

                    # load epochs here...

                    return epochs
    func_load_ndarray: callable
            第一引数がkeywords, 第二引数がmodeの，np.ndarrayオブジェクトを返す関数．
            keywordsはkeywords_trainやkeywords_testで渡されたlistオブジェクトが渡される．
            modeは，train, testのstrが渡される．

            .. code-block:: python

                def func_load_ndarray(keywords, mode):
                    # keywords: keywords_train or keywords_test
                    # mode: "train" or "test"

                    # load data here...

                    return ndarray


    classifier: list of classifier, "tslr", "mdm", instance
    label_keys: dict
    compile_test_subjects: bool
        Trueにすると，テストsubjectのデータをまとめて，その精度とかを返す
        Falseにすると，各被験者ごとの精度をリストで返す

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
            df_results["desc"] = [desc]

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
