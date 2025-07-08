import time

import numpy as np
import sklearn

import tag_mne as tm
import pandas as pd

from . import utils


def setup_optimizer(optimizer, optimizer_params, model):
    if optimizer_params is not None:
        optimizer = optimizer(
            params=model.parameters(),
            **optimizer_params,
        )
    else:
        optimizer = optimizer(params=model.parameters())

    return optimizer


def setup_scheduler(scheduler, scheduler_params, optimizer):
    if scheduler is not None:
        if scheduler_params is not None:
            scheduler = scheduler(
                optimizer=optimizer,
                **scheduler_params,
            )
        else:
            scheduler = scheduler(optimizer=optimizer)

    return scheduler


def deeplearning_train(
    dataloader_train,
    dataloader_valid,
    n_epochs,
    model,
    criterion,
    optimizer=None,
    scheduler=None,
    early_stopping=None,
    enable_wandb_logging=False,
    wandb_params=None,
    checkpoint_fname=None,
    history_fname=None,
):

    if enable_wandb_logging:
        import wandb

    if early_stopping is not None:
        early_stopping.initialize()

    history = {
        "epoch": list(),
        "train_loss": list(),
        "valid_loss": list(),
        "train_acc": list(),
        "valid_acc": list(),
    }

    loss_best = [float("inf")]

    # wandb.init(project = project, name = f"within-session_sub-{subject}_run-{run}")
    if enable_wandb_logging:
        wandb.init(**wandb_params)

    tic = time.time()
    for epoch in range(n_epochs):
        valid_loss = utils.train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader_train=dataloader_train,
            dataloader_valid=dataloader_valid,
            epoch=epoch,
            loss_best=loss_best,
            history=history,
            checkpoint_fname=checkpoint_fname,
            enable_wandb=enable_wandb_logging,
        )

        if early_stopping is not None:
            if early_stopping(valid_loss):
                print(f"Early stopping was triggered: epoch #{epoch+1}")
                break

    toc = time.time()
    elapsed_time = toc - tic
    print(f"Elapsed Time: {elapsed_time:.2f}s")

    if enable_wandb_logging:
        wandb.finish()

    if history_fname is not None:
        df_save = pd.DataFrame(history)
        df_save.to_pickle(f"{history_fname}.pkl")
        df_save.to_html(f"{history_fname}.html")

    return model


def deeplearning_cross_subject(
    subjects_train,
    subjects_valid,
    subjects_test,
    func_get_fnames,
    criterion,
    batch_size,
    n_epochs,
    optimizer,
    optimizer_params=None,
    model=None,
    func_get_model=None,
    scheduler=None,
    scheduler_params=None,
    device="cpu",
    func_proc_epochs=None,
    label_keys={"event:left": 0, "event:right": 1},
    compile_test_subjects=False,
    enable_wandb_logging=False,
    wandb_params=None,
    checkpoint_fname=None,
    history_fname=None,
    early_stopping=None,
    name_classifier=None,
    enable_euclidean_alignment=False,
    enable_normalization=False,
    desc=None,
):
    """
    subjects_train: list
    subjects_valid: list
    subjects_test: list
    func_get_fnames: callable
        subject名を引数とし，-epo.fifのファイル名リストを返す関数
        def fuc_get_fnames(subject):
            return [f"{subject-R1-epo.fif}", f"{subject-R2}-epo.fif"]
    criterion: instance of loss function
    batch_size: int
    n_epochs: int
    optimizer: reference to class
        e.g., torch.optim.AdamW
        インスタンスではなく，クラスへの参照を渡す
    optimizer_params: dict
        optimizerへ渡す**kwargs
    model: instance of pytorch model
        func_get_modelを使う場合はNoneでOK
    func_get_model: callable
        X_train, y_trainを引数として，modelを返す関数
        X_trainのshape依存のmodel等で便利
        modelに直接渡す場合はNoneでOK
    scheduler:
        e.g., scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
        インスタンスではなく，クラスへの参照を渡す
    scheduler_params: dict
        schedulerへ渡す**kwargs
    device: str
        "cpu", "cuda"...etc
    func_proc_epochs: callable
        mne.epochsを引数として，mne.epochsを返す
        チャネルを脳波のみにする，cropする...などで使う
    label_keys: dict
        epochsからクラス情報を抜き出すのに使う
        e.g., label_keys={"event:left": 0, "event:right": 1}
        この場合，"event:left"のエポックが0で，"event:right"が1になる
    compile_test_subjects: bool
        Trueのとき，複数のテストサブジェクトのデータを結合し，精度等を計算する
        Falseの場合は，被験者ごとの結果を返す
    enable_wandb_logging: bool
        wandbでのログ保存を有効化するか
    wandb_params: dict
        wandb.init()に渡すkwargs
        project, name, などを入れておくと良い
    checkpoint_fname: path-like
        checkpoint保存用ファイル名
        pthで保存される
    history_fname: path-like
        history保存用ファイル名
        pkl, htmlでpandas DataFrameが保存される
    early_stopping: int or callable
        early_stopping = rosoku.EarlyStopping(patience = 75)
        early_stopping = 75
    """
    import torch

    if enable_wandb_logging:
        import wandb

    if not isinstance(subjects_train, list) or not isinstance(subjects_test, list):
        raise ValueError("type of subjects_train and subjects_test have to be list")

    # load data

    ## training data
    X_train = []
    y_train = []
    for subject in subjects_train:
        files = func_get_fnames(subject)
        epochs = utils.load_epochs(files, True)
        if func_proc_epochs is not None:
            epochs = func_proc_epochs(epochs)
        y_train += utils.get_labels_from_epochs(epochs, label_keys)
        # epochs_train.append(epochs)
        X = epochs.get_data()

        if enable_euclidean_alignment:
            from . import tl

            X = tl.euclidean_alignment(X)

        X_train.append(X)

    # epochs_train = tm.concatenate_epochs(epochs_train)
    # X_train = epochs_train.get_data()
    X_train = np.concatenate(X_train, axis=0)

    ## validation data
    X_valid = list()
    y_valid = list()
    for subject in subjects_valid:
        files = func_get_fnames(subject)
        epochs = utils.load_epochs(files, True)
        if func_proc_epochs is not None:
            epochs = func_proc_epochs(epochs)
        y_valid += utils.get_labels_from_epochs(epochs, label_keys)

        X = epochs.get_data()

        if enable_euclidean_alignment:
            from . import tl

            X = tl.euclidean_alignment(X)

        X_valid.append(X)

    # epochs_valid = tm.concatenate_epochs(epochs_valid)
    # X_valid = epochs_valid.get_data()
    X_valid = np.concatenate(X_valid, axis=0)

    ## test data
    # epochs_test = list()
    X_test = list()
    y_test = list()
    for subject in subjects_test:
        files = func_get_fnames(subject)
        epochs = utils.load_epochs(files, True)
        if func_proc_epochs is not None:
            epochs = func_proc_epochs(epochs)
        y_test.append(utils.get_labels_from_epochs(epochs, label_keys))
        # epochs_test.append(epochs)
        X = epochs.get_data()

        if enable_euclidean_alignment:
            from . import tl

            X = tl.euclidean_alignment(X)

        X_test.append(X)

    if compile_test_subjects:
        # compile y
        y_test_compiled = list()
        for y in y_test:
            y_test_compiled += y
        y_test = [y_test_compiled]

        # compile X
        X_test = [np.concatenate(X_test, axis=0)]

        # compile subjects_test list
        subjects_test = [subjects_test]

    """
    # create dataloader

    (dataloader_train, dataloader_valid, dataloader_test) = utils.nd_to_dataloader(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        device=device,
        normalize=True,
        batch_size=batch_size,
    )

    # setup model

    if func_get_model is not None:
        model = func_get_model(X_train, y_train, device)

    if model is None:
        raise RuntimeError("model is None")

    if name_classifier is None:
        name_classifier = model.__class__.__name__

    # setup optimizer

    if optimizer_params is not None:
        optimizer = optimizer(
            params=model.parameters(),
            **optimizer_params,
        )
    else:
        optimizer = optimizer(params=model.parameters())

    # setup scheduler

    if scheduler is not None:
        if scheduler_params is not None:
            scheduler = scheduler(
                optimizer=optimizer,
                **scheduler_params,
            )
        else:
            scheduler = scheduler(optimizer=optimizer)

    # setup early stopping
    if early_stopping is not None:
        early_stopping = utils.EarlyStopping(patience=early_stopping)

    # misc

    history = {
        "epoch": list(),
        "train_loss": list(),
        "valid_loss": list(),
        "train_acc": list(),
        "valid_acc": list(),
    }

    loss_best = [float("inf")]

    # wandb.init(project = project, name = f"within-session_sub-{subject}_run-{run}")
    if enable_wandb_logging:
        wandb.init(**wandb_params)

    tic = time.time()
    for epoch in range(n_epochs):
        valid_loss = utils.train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader_train=dataloader_train,
            dataloader_valid=dataloader_valid,
            epoch=epoch,
            loss_best=loss_best,
            history=history,
            checkpoint_fname=checkpoint_fname,
            enable_wandb=enable_wandb_logging,
        )

        if early_stopping is not None:
            if early_stopping.step(valid_loss):
                print(f"Early stopping was triggered: epoch #{epoch+1}")
                break

    toc = time.time()
    elapsed_time = toc - tic

    if enable_wandb_logging:
        wandb.finish()

    if history_fname is not None:
        df_save = pd.DataFrame(history)
        df_save.to_pickle(f"{history_fname}.pkl")
        df_save.to_html(f"{history_fname}.html")

    print(f"Elapsed Time: {elapsed_time:.2f}s")
    """

    # data normalization
    if enable_normalization:
        X_train, X_valid, X_test, normalization_mean, normalization_std = (
            utils.normalize(X_train, X_valid, X_test, return_params=True)
        )

    # create dataloader

    (dataloader_train, dataloader_valid, dataloader_test) = utils.nd_to_dataloader(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        device=device,
        batch_size=batch_size,
    )

    # setup model

    if func_get_model is not None:
        model = func_get_model(X_train, y_train)

    if model is None:
        raise RuntimeError("model is None")

    model.to(device)

    if name_classifier is None:
        name_classifier = model.__class__.__name__

    # setup optimizer
    optimizer = setup_optimizer(optimizer, optimizer_params, model)

    # setup scheduler
    scheduler = setup_scheduler(scheduler, scheduler_params, optimizer)

    # setup early stopping
    if isinstance(early_stopping, int):
        early_stopping = utils.EarlyStopping(patience=early_stopping)

    model = deeplearning_train(
        dataloader_train=dataloader_train,
        dataloader_valid=dataloader_valid,
        n_epochs=n_epochs,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        enable_wandb_logging=enable_wandb_logging,
        wandb_params=wandb_params,
        checkpoint_fname=checkpoint_fname,
        history_fname=history_fname,
        early_stopping=early_stopping,
    )

    # classify test data
    if checkpoint_fname is not None:
        checkpoint = torch.load(f"{checkpoint_fname}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])

    df_list = list()
    model.eval()
    with torch.no_grad():
        for dataloader, subject in zip(dataloader_test, subjects_test):
            # accuracy = utils.accuracy_score_dataloader(model, dataloader_test)

            preds, labels, logits, probas = utils.get_predictions(
                model,
                dataloader,
            )

            accuracy = sklearn.metrics.accuracy_score(labels, preds)

            df_results = pd.DataFrame()
            df_results["subjects_train"] = [subjects_train]
            df_results["subjects_valid"] = [subjects_valid]
            df_results["subjects_test"] = [subject]
            df_results["classifier"] = [name_classifier]
            df_results["accuracy"] = [accuracy]
            df_results["labels"] = [labels]
            df_results["preds"] = [preds]
            df_results["probas"] = [probas]
            df_results["logits"] = [logits]
            # df_results["elapsed_time"] = [elapsed_time]
            df_results["desc"] = [desc]
            if enable_normalization:
                df_results["normalization_mean"] = normalization_mean
                df_results["normalization_std"] = normalization_std

            df_list.append(df_results)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    return df


if __name__ == "__main__":
    import torch
    import braindecode

    def _func_proc_epochs(epochs):
        epochs = epochs.pick(picks="eeg").crop(tmin=0.25, tmax=5.0)
        return epochs

    def _func_get_fnames(subject):
        from pathlib import Path

        base_dir = (
            Path("~/Documents/datasets/dreyer_2023/derivatives")
            / "epochs"
            / "l_freq-8.0_h_freq-30.0_resample-128"
            / subject
        )

        fnames_list = list()
        fnames_list.append(base_dir / f"sub-{subject}_acquisition-epo.fif")
        fnames_list.append(base_dir / f"sub-{subject}_online-epo.fif")

        return fnames_list

    def _func_get_model(X, y, device):
        print(X.shape)
        _, n_chans, n_times = X.shape
        F1 = 4
        D = 2
        F2 = F1 * D

        model = braindecode.models.EEGNetv4(
            n_chans=n_chans,
            n_outputs=2,
            n_times=n_times,
            F1=F1,
            D=D,
            F2=F2,
            drop_prob=0.5,
        )
        model.to(device)

        return model

    # setup mpdels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 0.001
    weight_decay = 0.0005
    n_epochs = 10
    batch_size = 64
    patience = 75

    early_stopping = utils.EarlyStopping(patience=patience)

    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    returns = deeplearning_cross_subject(
        subjects_train=["A1", "A2"],
        subjects_valid=["A3", "A4"],
        subjects_test=["A5", "A6"],
        func_get_fnames=_func_get_fnames,
        func_proc_epochs=_func_proc_epochs,
        func_get_model=_func_get_model,
        criterion=criterion,
        scheduler=scheduler,
        batch_size=batch_size,
        n_epochs=n_epochs,
        checkpoint_fname=None,
        early_stopping=early_stopping,
        device=device,
        history_fname=None,
        scheduler_params={"T_max": n_epochs, "eta_min": 1e-6},
        optimizer=torch.optim.AdamW,
        optimizer_params={"lr": lr, "weight_decay": weight_decay},
        compile_test_subjects=False,
    )
