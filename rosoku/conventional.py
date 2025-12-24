import json
import pyriemann
import pandas as pd
from . import utils

from .utils.core import _add_values_to_df


def conventional(
    items_train,
    items_test,
    callback_load_epochs=None,
    callback_proc_epochs=None,
    callback_convert_epochs_to_ndarray=utils.convert_epochs_to_ndarray,
    callback_load_ndarray=None,
    callback_proc_ndarray=None,
    callback_proc_mode="per_split",
    callback_fit=None,
    callback_predict=None,
    callback_predict_proba=None,
    callback_get_models=None,
    scoring="accuracy",
    scoring_name=None,
    models=[
        pyriemann.classification.TSClassifier(),
        pyriemann.classification.MDM(),
    ],
    model_names=None,
    samples_fname=None,
    additional_values=None,
):
    """
    Run a conventional (non-deep-learning) classification pipeline.

    This utility orchestrates end-to-end evaluation for classical machine-learning
    models with a scikit-learn-like interface (i.e., estimators implementing
    ``fit``/``predict`` and optionally ``predict_proba``) by:

    1) loading training and test data via user callbacks (as MNE Epochs or NumPy),
    2) optionally preprocessing Epochs/arrays,
    3) fitting one or more estimators,
    4) predicting labels (and, if available, class probabilities),
    5) computing one or more scoring functions on each test group, and
    6) returning a tidy results table (optionally exporting sample-level outputs).

    While the default behavior follows a standard scikit-learn workflow, the
    ``callback_fit``, ``callback_predict``, and ``callback_predict_proba`` hooks can be
    used to override the fitting/prediction steps, enabling more flexible behaviors
    (e.g., custom wrappers, non-standard estimators, additional post-processing, or
    alternative probability generation).

    Test data can be evaluated in user-defined groups: each element of ``items_test``
    represents one evaluation group, and can contain one or multiple items (e.g., to
    merge multiple sessions into a single test set before scoring).

    Parameters
    ----------
    items_train : list
        Item objects describing how to load the training data. The exact content is
        user-defined and interpreted by ``callback_load_epochs`` or
        ``callback_load_ndarray``.

    items_test : list of list
        Item objects describing how to load the test data, grouped for evaluation.
        Each inner list defines one evaluation group.
        For example, ``[[a], [b]]`` evaluates ``a`` and ``b`` separately, while
        ``[[a, b]]`` merges ``a`` and ``b`` into a single test set.

    callback_load_epochs : callable | None, optional
        Data loader returning an :class:`mne.Epochs` object for a given item.
        Must have signature ``callback_load_epochs(item, split)``, where ``split``
        is ``"train"`` or ``"test"``. If provided, Epochs will be converted to
        arrays via ``callback_convert_epochs_to_ndarray``.

    callback_proc_epochs : callable | None, optional
        Optional preprocessing applied to loaded Epochs (e.g., picking channels,
        cropping, filtering). Signature is expected to be
        ``callback_proc_epochs(epochs, split)`` or ``callback_proc_epochs(epochs)``
        depending on your implementation used in ``utils.load_data``.

    callback_convert_epochs_to_ndarray : callable, optional
        Converter used when loading Epochs. By default,
        ``utils.convert_epochs_to_ndarray`` is used.

    callback_load_ndarray : callable | None, optional
        Data loader returning a tuple ``(X, y)`` for a given item. Must have
        signature ``callback_load_ndarray(item, split)`` where ``split`` is
        ``"train"`` or ``"test"``. ``X`` and ``y`` must be array-like.

    callback_proc_ndarray : callable | None, optional
        Optional preprocessing applied to NumPy data (e.g., standardization, feature
        extraction), as used by ``utils.load_data``.

    callback_proc_mode : {"per_split", "joint"}, optional
        Controls how preprocessing callbacks are applied, as interpreted by
        ``apply_callback_proc``.

        - ``"per_split"``:
            The preprocessing callback is applied independently to each data
            split (train / valid / test).

            The callback function must have the signature::

                callback_proc(data, split)

            where ``split`` is one of ``{"train", "valid", "test"}``.

            The callback must return the processed version of ``data``.
            No information is shared across splits in this mode.

        - ``"joint"``:
            The preprocessing callback is applied jointly to multiple splits
            at once, allowing shared state across splits.

            If validation data are not provided (``valid is None``), the callback
            must have the signature::

                callback_proc(train, test)

            and must return::

                train, test

            If validation data are provided, the callback must have the signature::

                callback_proc(train, valid, test)

            and must return::

                train, valid, test

            This mode is intended for stateful preprocessing steps such as
            fitting a transformation on training data and applying it
            consistently to validation and test data.

    callback_fit : callable | None, optional
        Optional custom fitting hook. If provided, called as
        ``callback_fit(model, X_train, y_train)``.
        If ``None``, this function calls ``model.fit(X_train, y_train)``.
        Note: if your callback returns a new fitted estimator, ensure it is mutated
        in-place or manage estimator replacement consistently.

    callback_predict : callable | None, optional
        Optional custom prediction hook. If provided, called as
        ``callback_predict(model, X)``. If ``None``, uses ``model.predict(X)``.

    callback_predict_proba : callable | None, optional
        Optional custom probability prediction hook. If provided, called as
        ``callback_predict_proba(model, X)``. If ``None``, uses
        ``model.predict_proba(X)``. Estimators must support probability outputs.

    callback_get_models : callable | None, optional
        Factory function that returns one or more scikit-learn estimators.
        This callback is used only when ``models=None``.

        If provided, it is called as::

            callback_get_models(X_train, y_train)

        where ``X_train`` and ``y_train`` are the training arrays returned by
        ``utils.load_data``.

        The return value **must be** either:

        - a single scikit-learn–compatible estimator, or
        - a list of scikit-learn–compatible estimators.

        Each estimator must implement at least ``fit`` and ``predict``.
        If class probabilities are required (default behavior), estimators must also
        implement ``predict_proba`` unless ``callback_predict_proba`` is provided.

    scoring : str | callable | list of (str or callable), optional
        Scoring specification(s) applied to each test group.
        If a string, it is resolved with :func:`sklearn.metrics.get_scorer` and the
        underlying ``_score_func`` is used.
        If a callable, it must have signature ``scoring(y_true, y_pred)`` and return
        a scalar.

    scoring_name : str | list of str | None, optional
        Column name(s) for the returned scores. If ``None``, names are inferred:
        strings keep their name, callables become ``"callable"`` (and other types
        become ``"unknown_scoring"``). Must match ``scoring`` length.

    models : estimator | list of estimator | None, optional
        One or more estimators implementing at least ``fit`` and ``predict``.
        If probabilities are required (default behavior), estimators should also
        implement ``predict_proba`` (or you must provide ``callback_predict_proba``).

    model_names : list of str | None, optional
        Display names for ``models`` used in outputs. If ``None``, uses
        ``model.__class__.__name__`` for each estimator.

    samples_fname : path-like | None, optional
        If provided, writes sample-level outputs to this path in Parquet format.
        The file includes true labels, predicted labels, per-class probabilities,
        and the model name (plus ``additional_values`` if given).

    additional_values : dict | None, optional
        Extra metadata appended as columns to the results DataFrame (and also to the
        sample-level table if ``samples_fname`` is provided).

    Returns
    -------
    df : pandas.DataFrame
        Summary results with one row per (test group × model). Includes JSON-serialized
        ``items_train`` / ``items_test`` strings, the model name, and one column
        per requested scoring metric.

    Notes
    -----
    - ``items_test`` grouping controls evaluation granularity: each inner list is
      treated as one test set after loading/merging by ``utils.load_data``.
    - If you pass a scoring string, this function uses
      ``sklearn.metrics.get_scorer(scoring)._score_func``. This typically matches the
      metric function but may ignore scorer-specific configuration (e.g., sign flipping
      for losses) because the scorer object itself is not called.
    - Probability outputs are always attempted; ensure your estimator supports
      ``predict_proba`` or provide ``callback_predict_proba``.
    - ``callback_get_models`` is ignored when ``models`` is explicitly provided. In that
      case, the estimators passed via ``models`` are used directly.
    """

    X_train, _, X_test, y_train, _, y_test = utils.load_data(
        items_train=items_train,
        items_valid=None,
        items_test=items_test,
        callback_load_epochs=callback_load_epochs,
        callback_load_ndarray=callback_load_ndarray,
        callback_proc_epochs=callback_proc_epochs,
        callback_proc_ndarray=callback_proc_ndarray,
        callback_proc_mode=callback_proc_mode,
        callback_convert_epochs_to_ndarray=callback_convert_epochs_to_ndarray,
    )

    if models is None:
        models = callback_get_models(X_train, y_train)

    if not isinstance(models, list):
        models = [models]

    if not isinstance(scoring, list):
        scoring = [scoring]

    if scoring_name is None:
        scoring_name = []
        for idx, scoring_ in enumerate(scoring):
            if isinstance(scoring_, str):
                scoring_name.append(scoring_)
            elif callable(scoring_):
                scoring_name.append("callable")
            else:
                scoring_name.append("unknown_scoring")

    if not isinstance(scoring_name, list):
        scoring_name = [scoring_name]

    if len(scoring) != len(scoring_name):
        raise RuntimeError("len(scoring) != len(scoring_name)")

    for idx, scoring_ in enumerate(scoring):
        if isinstance(scoring_, str):
            from sklearn.metrics import get_scorer

            scoring_ = get_scorer(scoring_)._score_func
        elif callable(scoring_):
            # do nothing
            pass
        else:
            raise ValueError(f"Invalid scoring: {scoring_}")
        scoring[idx] = scoring_

    if model_names is None:
        model_names = [model.__class__.__name__ for model in models]

    # train classifiers
    for model in models:
        if callback_fit is None:
            model.fit(X_train, y_train)
        else:
            model = callback_fit(model, X_train, y_train)

    # classify test data and evaluate results

    if not isinstance(X_test, list):
        X_test = [X_test]
        y_test = [y_test]

    if len(items_test) != len(X_test):
        raise RuntimeError("len(items_test) != len(X_test)")

    df_list = []
    samples_list = []
    for X, y, item in zip(X_test, y_test, items_test):
        for model, name in zip(models, model_names):

            df_results = pd.DataFrame()

            if callback_predict is None:
                preds = model.predict(X)
            else:
                preds = callback_predict(model, X)

            if callback_predict_proba is None:
                probas = model.predict_proba(X)
            else:
                probas = callback_predict_proba(model, X)

            scores = []
            for scoring_ in scoring:
                scores.append(scoring_(y, preds))

            df_results["items_train"] = [json.dumps(items_train)]
            df_results["items_test"] = [json.dumps(item)]
            df_results["model"] = [name]

            for scoring_name_, score in zip(scoring_name, scores):
                df_results[scoring_name_] = [score]

            samples = pd.DataFrame()
            samples["labels"] = y
            samples["preds"] = preds
            for idx in range(probas.shape[1]):
                samples[f"probas_{idx}"] = probas[:, idx]
            samples["model"] = [name for _ in range(len(samples))]

            samples_list.append(samples)
            df_list.append(df_results)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    if additional_values is not None:
        df = _add_values_to_df(df, additional_values)

    if samples_fname is not None:
        samples = pd.concat(samples_list, axis=0, ignore_index=True)
        if additional_values is not None:
            samples = _add_values_to_df(samples, additional_values)
        samples.to_parquet(samples_fname)

    return df
