import numpy as np
import tag_mne as tm


def get_labels_from_epochs(epochs, label_keys={"event:left": 0, "event:right": 1}):
    y = list()

    _, markers = tm.markers_from_events(epochs.events, epochs.event_id)

    for marker in markers:
        for key, val in label_keys.items():
            if key in marker:
                y.append(val)

    return y


def apply_func_proc(func_proc, func_proc_mode, train, valid, test):

    match func_proc_mode:
        case "per_split":
            train = func_proc(train)
            if valid is not None:
                valid = func_proc(valid)
            test = [func_proc(obj) for obj in test]
        case "per_function":
            if valid is not None:
                train, test = func_proc(train, test)
            else:
                train, valid, test = func_proc(train, valid, test)

        case _:
            raise ValueError(f"func_proc_mode: {func_proc_mode} is unknown")

    return train, valid, test


def _____convert_epochs_to_ndarray(
    epochs_train,
    epochs_valid,
    epochs_test,
    label_keys={"event:left": 0, "event:right": 1},
):
    """
    will be deprecated
    """

    X_train = epochs_train.get_data()
    if epochs_valid is not None:
        X_valid = epochs_valid.get_data()
    X_test = epochs_test.get_data()

    y_train = get_labels_from_epochs(epochs_train, label_keys)

    if epochs_valid is not None:
        y_valid = get_labels_from_epochs(epochs_valid, label_keys)
    y_test = get_labels_from_epochs(epochs_test, label_keys)

    if epochs_valid is not None:
        return X_train, X_valid, X_test, y_train, y_valid, y_test
    else:
        return X_train, X_test, y_train, y_test


def load_data_2(
    keywords_train,
    keywords_valid,
    keywords_test,
    func_load_epochs=None,
    func_load_ndarray=None,
    func_proc_epochs=None,
    func_proc_ndarray=None,
    apply_func_proc_per_obj=True,
    func_proc_mode="per_split",
    func_convert_epochs_to_ndarray=convert_epochs_to_ndarray,
    compile_test=False,
):
    """
    渡されたkeywordsをもとに，データを読み出す関数

    Parameters
    ----------
    keywords_train : list
    keywords_valid : list
    keywords_test : list
    func_load_epochs : callable
    func_load_ndarray : callable
    func_proc_epochs : callable
    func_proc_ndarray : callable
    func_proc_mode : str, default = "per_split"
        "per_split" or "per_function"

        per_splitでは，func_procに対して，train, valid, testのスプリット毎に関数がコールバックされる．

        per_functionでは，func_procに対して，train, valid, testがまとめて渡される．

    func_convert_epochs_to_ndarray : callable, default=rosoku.utils.convert_epochs_to_ndarray
    compile_test : bool, default=False

    Notes
    -----
    ``func_load_epochs``か``func_load_ndarray``のどちらかが必要．

    ``func_load_ndarray=None``の場合の流れは以下のような感じ．

    .. code-block:: text

        func_load_epochs()
        ↓
        func_proc_epochs()
        ↓
        func_convert_epochs_to_ndarray()
        ↓
        func_proc_ndarray()

    ``func_load_epochs=None``の場合の流れは以下のような感じ．

    .. code-block:: text

        func_load_ndarray()
        ↓
        func_proc_ndarray()



    Returns
    -------

    """

    if func_load_epochs is None and func_load_ndarray is None:
        raise ValueError("Specify func_load_epochs or func_load_ndarray")

    if keywords_valid is None:
        if isinstance(keywords_train, list) and isinstance(keywords_test, list):
            pass
        else:
            raise ValueError(
                "keywords_train and keywords_test must be instance of list"
            )
    else:
        if (
            isinstance(keywords_train, list)
            and isinstance(keywords_valid, list)
            and isinstance(keywords_test, list)
        ):
            pass
        else:
            raise ValueError(
                "keywords_train, keywords_valid, and keywords_test must be instance of list"
            )

    if func_load_epochs is not None:
        # load epochs
        epochs_train = func_load_epochs(keywords_train, "train")
        if keywords_valid is None:
            epochs_valid = None
        else:
            epochs_valid = func_load_epochs(keywords_valid, "valid")
        epochs_test = [func_load_epochs(keyword, "test") for keyword in keywords_test]

        (epochs_train, epochs_valid, epochs_test) = apply_func_proc(
            func_proc=func_proc_epochs,
            func_proc_mode=func_proc_mode,
            train=epochs_train,
            valid=epochs_valid,
            test=epochs_test,
        )

    exit()

    if apply_func_proc_per_obj:
        _load_data_per_obj(
            keywords_train=keywords_train,
            keywords_valid=keywords_valid,
            keywords_test=keywords_test,
            func_load_epochs=func_load_epochs,
            func_load_ndarray=func_load_ndarray,
            func_proc_epochs=func_proc_epochs,
            func_proc_ndarray=func_proc_ndarray,
            func_convert_epochs_to_ndarray=func_convert_epochs_to_ndarray,
            compile_test=compile_test,
        )

    else:
        pass

    exit()
    ## training data
    if func_load_epochs is not None:
        epochs_train = func_load_epochs(keywords_train, "train")
        if keywords_valid is not None:
            epochs_valid = func_load_epochs(keywords_valid, "valid")
        epochs_test = func_load_epochs(keywords_test, "test")

        if func_proc_epochs is not None:
            if apply_func_proc_per_obj:
                epochs_train = func_proc_epochs(epochs_train, "train")
                if keywords_valid is not None:
                    epochs_valid = func_proc_epochs(epochs_valid, "valid")
                epochs_test = func_proc_epochs(epochs_test, "test")
            else:
                if keywords_valid is not None:
                    epochs_train, epochs_valid, epochs_test = func_proc_epochs(
                        epochs_train,
                        epochs_valid,
                        epochs_test,
                    )
                else:
                    epochs_train, epochs_test = func_proc_epochs(
                        epochs_train,
                        epochs_test,
                    )

        if keywords_valid is not None:
            X_train, X_valid, X_test, y_train, y_valid, y_test = (
                func_convert_epochs_to_ndarray(epochs_train, epochs_valid, epochs_test)
            )
        else:
            X_train, X_test, y_train, y_test = func_convert_epochs_to_ndarray(
                epochs_train, epochs_test
            )

    elif func_load_ndarray is not None:
        X_train, y_train = func_load_ndarray(keywords_train, "train")
        if keywords_valid is not None:
            X_valid, y_valid = func_load_ndarray(keywords_valid, "valid")
        X_test, y_test = func_load_ndarray(keywords_test, "test")
    else:
        raise ValueError(
            "either func_load_epochs or func_load_ndarray should not be None"
        )

    if compile_test:
        # compile y
        y_test = [np.concatenate(y_test, axis=0)]

        # compile X
        X_test = [np.concatenate(X_test, axis=0)]

    if func_proc_ndarray is not None:
        if apply_func_proc_per_obj:
            X_train, y_train = func_proc_ndarray(X_train, y_train, "train")
            if keywords_valid is not None:
                X_valid, y_valid = func_proc_ndarray(X_valid, y_valid, "valid")
            X_test, y_test = func_proc_ndarray(X_test, y_test, "test")
        else:
            if keywords_valid is not None:
                X_train, X_valid, X_test, y_train, y_valid, y_test = func_proc_ndarray(
                    X_train, X_valid, X_test, y_train, y_valid, y_test
                )
            else:
                X_train, X_test, y_train, y_test = func_proc_ndarray(
                    X_train, X_test, y_train, y_test
                )
    else:
        if keywords_valid is not None:
            return X_train, X_valid, X_test, y_train, y_valid, y_test
        else:
            return X_train, X_test, y_train, y_test
