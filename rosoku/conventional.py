import time

import numpy as np
import scipy

import mne
import pyriemann
import sklearn

import tag_mne as tm
import pandas as pd

from . import utils


def func_proc_epochs(epochs):
    epochs = epochs.pick(picks="eeg").crop(tmin=0.25, tmax=5.0)
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


def riemannian_cross_subject(
    subjects_train,
    subjects_test,
    func_get_fnames,
    func_proc_epochs=None,
    classifiers=["tslr"],
    name_classifiers=None,
    label_keys={"event:left": 0, "event:right": 1},
    enable_cov_recentering=True,
    enable_cov_scaling=True,
    compile_test_subjects=False,
    desc=None,
):
    """
    subjects_train: list
    subjects_test: list
    files: callable
        subject名を引数とし，ファイル名を返す関数
    func_proc_epochs: function
        def func_proc_eochs(epochs)
            return epochs
    classifier: list of classifier, "tslr", "mdm", instance
    label_keys: dict
    compile_test_subjects: bool
        Trueにすると，テストsubjectのデータをまとめて，その精度とかを返す
        Falseにすると，各被験者ごとの精度をリストで返す

    """

    if not isinstance(subjects_train, list) or not isinstance(subjects_test, list):
        raise ValueError("type of subjects_train and subjects_test have to be list")

    # load data

    ## training data
    cov_train = list()
    y_train = list()
    for subject in subjects_train:
        files = func_get_fnames(subject)
        epochs = load_epochs(files, True)
        if func_proc_epochs is not None:
            epochs = func_proc_epochs(epochs)
        y_train += utils.get_labels_from_epochs(epochs, label_keys)

        cov = pyriemann.estimation.Covariances().transform(epochs.get_data())

        if enable_cov_recentering:
            cov = recenter_cov(cov, scaling=enable_cov_scaling)

        cov_train.append(cov)
    cov_train = np.concatenate(cov_train, axis=0)

    ## test data
    cov_test = list()
    y_test = list()
    for subject in subjects_test:
        files = func_get_fnames(subject)
        epochs = load_epochs(files, True)
        if func_proc_epochs is not None:
            epochs = func_proc_epochs(epochs)
        y_test.append(utils.get_labels_from_epochs(epochs, label_keys))

        cov = pyriemann.estimation.Covariances().transform(epochs.get_data())

        if enable_cov_recentering:
            cov = recenter_cov(cov, scaling=enable_cov_scaling)

        cov_test.append(cov)

    if compile_test_subjects:
        # compile y
        y_test_compiled = list()
        for y in y_test:
            y_test_compiled += y
        y_test = [y_test_compiled]

        # compile cov
        cov_test = [np.concatenate(cov_test, axis=0)]

        # compile subjects_test list
        subjects_test = [subjects_test]

    # instanciate classification model
    clf_list = list()
    clf_name = list()
    for idx_classifiers, classifier in enumerate(classifiers):
        if type(classifier) == str:
            if classifier == "tslr":
                clf_list.append(pyriemann.classification.TSClassifier())
                if name_classifiers is not None:
                    clf_name.append(name_classifiers[idx_classifiers])
                else:
                    clf_name.append("tslr")
            elif classifier == "mdm":
                clf_list.append(
                    pyriemann.classification.MDM(
                        metric=dict(mean="riemann", distance="riemann")
                    )
                )
                if name_classifiers is not None:
                    clf_name.append(name_classifiers[idx_classifiers])
                else:
                    clf_name.append("mdm")
            else:
                raise ValueError(f"classifier '{classifier}' is not known.")
        else:
            clf_list.append(classifier)
            if name_classifiers is not None:
                clf_name.append(name_classifiers[idx_classifiers])
            else:
                clf_name.append(str(classifier))

    tic = time.time()

    # train classifiers
    elapsed_times = list()
    for clf in clf_list:
        tic = time.time()
        clf.fit(cov_train, y_train)
        toc = time.time()
        elapsed_times.append(toc - tic)

    # classify test data and evaluate results

    df_list = list()
    for clf, name, elapsed_time in zip(clf_list, clf_name, elapsed_times):
        for cov, y, subject in zip(cov_test, y_test, subjects_test):
            df_results = pd.DataFrame()

            preds = clf.predict(cov)
            probas = clf.predict_proba(cov)
            accuracy = sklearn.metrics.accuracy_score(y, preds)

            df_results["subjects_train"] = [subjects_train]
            df_results["subjects_test"] = [subject]
            df_results["classifier"] = [name]
            df_results["accuracy"] = [accuracy]
            df_results["labels"] = [y]
            df_results["preds"] = [preds]
            df_results["probas"] = [probas]
            # df_results["elapsed_time"] = [elapsed_time]
            df_results["desc"] = [desc]

            df_list.append(df_results)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    return df


def conventional(
    keywords_train,
    keywords_test,
    func_load_epochs=None,
    func_load_ndarray=None,
    func_proc_epochs=None,
    func_proc_ndarray=None,
    apply_func_proc_per_obj=True,
    classifiers=[
        pyriemann.classification.TSClassifier(),
        pyriemann.classification.MDM(),
    ],
    classifier_names=["tslr", "mdm"],
    name_classifiers=None,
    func_convert_epochs_to_ndarray=utils.convert_epochs_to_ndarray,
    compile_test=False,
    desc=None,
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

    X_train, X_test, y_train, y_test = utils.load_data(
        keywords_train=keywords_train,
        keywords_test=keywords_test,
        func_load_epochs=func_load_epochs,
        func_load_ndarray=func_load_ndarray,
        func_proc_epochs=func_proc_epochs,
        func_proc_ndarray=func_proc_ndarray,
        apply_func_proc_per_obj=apply_func_proc_per_obj,
        func_convert_epochs_to_ndarray=func_convert_epochs_to_ndarray,
        compile_test=compile_test,
    )

    # train classifiers
    for clf in classifiers:
        clf.fit(X_train, y_train)

    # classify test data and evaluate results

    if isinstance(X_test, list) is False:
        X_test = [X_test]
        y_test = [y_test]

    df_list = list()
    for X, y, keywords in zip(X_test, y_test, keywords_test):
        for clf, name in zip(classifiers, classifier_names):

            df_results = pd.DataFrame()

            preds = clf.predict(X)
            probas = clf.predict_proba(X)
            accuracy = sklearn.metrics.accuracy_score(y, preds)

            df_results["keywords_train"] = [keywords_train]
            df_results["keywords_test"] = [keywords]
            df_results["classifier"] = [name]
            df_results["accuracy"] = [accuracy]
            df_results["labels"] = [y]
            df_results["preds"] = [preds]
            df_results["probas"] = [probas]
            df_results["desc"] = [desc]

            df_list.append(df_results)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    return df


if __name__ == "__main__":
    debug = True
    import sys

    sys.path.append("../")
    import load

    def _func_proc_epochs(epochs):
        epochs = epochs.pick(picks="eeg").crop(tmin=0.25, tmax=5.0)
        return epochs

    def _func_get_fnames(subject):
        base_dir = (
            load.config["dir"]["deriv"]
            / "epochs"
            / "l_freq-8.0_h_freq-30.0_resample-128"
            / subject
        )

        fnames_list = list()
        fnames_list.append(base_dir / f"sub-{subject}_acquisition-epo.fif")
        fnames_list.append(base_dir / f"sub-{subject}_online-epo.fif")

        return fnames_list

    returns = riemannian_cross_subject(
        subjects_train=["A1", "A2"],
        subjects_test=["A4", "A5"],
        func_get_fnames=_func_get_fnames,
        func_proc_epochs=_func_proc_epochs,
        classifiers=["tslr", "mdm"],
        enable_cov_recentering=False,
        compile_test_subjects=False,
    )
