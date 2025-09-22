import os
import time
import random

import numpy as np
import sklearn

import torch

import tag_mne as tm
import pandas as pd

from . import utils
from . import preprocessing
from . import attribution


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


def load_data(
    subjects, func_get_fnames, func_proc_epochs, label_keys, enable_euclidean_alignment
):

    X = []
    y = []

    for subject in subjects:

        files = func_get_fnames(subject)

        epochs = utils.load_epochs(files, True)

        if func_proc_epochs is not None:
            epochs = func_proc_epochs(epochs)

        y.append(np.array(utils.get_labels_from_epochs(epochs, label_keys)))

        X_single = epochs.get_data()

        if enable_euclidean_alignment:
            from . import tl

            X_single = tl.euclidean_alignment(X_single)

        X.append(X_single)

    return X, y


def deeplearning_train(
    dataloader_train,
    dataloader_valid,
    n_epochs,
    model,
    criterion,
    device,
    optimizer=None,
    scheduler=None,
    early_stopping=None,
    enable_wandb_logging=False,
    wandb_params=None,
    checkpoint_fname=None,
    history_fname=None,
    enable_ddp=False,
    enable_dp=False,
    sampler_train=None,
    rank=0,
):

    if enable_wandb_logging:
        if (enable_ddp and rank == 0) or (enable_ddp is False):
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

    loss_best = {"value": float("inf")}

    if enable_wandb_logging:
        if (enable_ddp and rank == 0) or (enable_ddp is False):
            wandb.init(**wandb_params)

    tic = time.time()
    for epoch in range(n_epochs):
        if enable_ddp:
            sampler_train.set_epoch(epoch)
        valid_loss = utils.train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader_train=dataloader_train,
            dataloader_valid=dataloader_valid,
            epoch=epoch,
            device=device,
            loss_best=loss_best,
            history=history,
            checkpoint_fname=checkpoint_fname,
            enable_wandb=enable_wandb_logging,
            enable_dp=enable_dp,
            enable_ddp=enable_ddp,
            rank=rank,
        )

        if early_stopping is not None:
            if enable_ddp is False and early_stopping(valid_loss):
                print(f"Early stopping was triggered: epoch #{epoch+1}")
                break
            elif enable_ddp:
                should_stop = False
                if rank == 0:
                    should_stop = early_stopping(valid_loss)
                should_stop_tensor = torch.tensor(
                    should_stop, dtype=torch.uint8, device=device
                )
                torch.distributed.broadcast(should_stop_tensor, src=0)
                should_stop = bool(should_stop_tensor.item())

                if should_stop:
                    if rank == 0:
                        print(f"Early stopping was triggered: epoch #{epoch+1}")
                    break

    toc = time.time()
    elapsed_time = toc - tic
    if rank == 0:
        print(f"Elapsed Time: {elapsed_time:.2f}s")

    if history_fname is not None and rank == 0:
        df_save = pd.DataFrame(history)
        df_save.to_pickle(f"{history_fname}.pkl")
        df_save.to_html(f"{history_fname}.html")

    return model


def load_data(
    subjects, func_get_fnames, func_proc_epochs, label_keys, enable_euclidean_alignment
):

    X = []
    y = []

    for subject in subjects:

        files = func_get_fnames(subject)

        epochs = utils.load_epochs(files, True)

        if func_proc_epochs is not None:
            epochs = func_proc_epochs(epochs)

        y.append(np.array(utils.get_labels_from_epochs(epochs, label_keys)))

        X_single = epochs.get_data()

        if enable_euclidean_alignment:
            from . import tl

            X_single = tl.euclidean_alignment(X_single)

        X.append(X_single)

    return X, y


def main(
    enable_ddp,
    enable_dp,
    num_workers,
    device,
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test,
    criterion,
    batch_size,
    n_epochs,
    optimizer,
    kwargs,
):

    import torch

    # parse kwargs

    optimizer_params = kwargs.get("optimizer_params", None)
    model = kwargs.get("model", None)
    func_get_model = kwargs.get("func_get_model", None)
    scheduler = kwargs.get("scheduler", None)
    scheduler_params = kwargs.get("scheduler_params", None)
    enable_wandb_logging = kwargs.get("enable_wandb_logging", False)
    wandb_params = kwargs.get("wandb_params", None)
    checkpoint_fname = kwargs.get("checkpoint_fname", None)
    history_fname = kwargs.get("history_fname", None)
    early_stopping = kwargs.get("early_stopping", None)
    name_classifier = kwargs.get("name_classifier", None)
    seed = kwargs.get("seed", None)

    # setup DDP
    if enable_ddp:

        params = utils.get_ddp_params()

        rank = params["rank"]
        local_rank = params["local_rank"]
        world_size = params["world_size"]
        master_addr = params["master_addr"]
        master_port = params["master_port"]

        print(f"rank: {rank}, world_size: {world_size}, local_rank: {local_rank}")
        print(f"MASTER_ADDR: {master_addr}, MASTER_PORT: {master_port}")

        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
        )

        if torch.distributed.is_initialized():
            print(f"[Rank {rank}] Distributed initialized: OK")
        else:
            print(f"[Rank {rank}] Distributed not initialized: NG")
            raise RuntimeError(f"[Rank {rank}] Distributed not initialized: NG")

        device = torch.device(f"cuda:{local_rank}")
    else:
        # non DDP
        rank = 0

    # create dataloader

    if enable_ddp:
        (dataloader_train, dataloader_valid, _, sampler_train) = utils.nd_to_dataloader(
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test,
            device="cpu",
            batch_size=batch_size,
            enable_DS=True,
            DS_params={
                "world_size": world_size,
                "num_workers": num_workers,
                "rank": rank,
            },
            generator=seed,
        )
    else:

        (dataloader_train, dataloader_valid, _) = utils.nd_to_dataloader(
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test,
            device="cpu",
            batch_size=batch_size,
            enable_DS=False,
            generator=seed,
        )
        sampler_train = None

    # setup model

    if func_get_model is not None:
        model = func_get_model(X_train, y_train)

    if model is None:
        raise RuntimeError("model is None")

    model.to(device)

    if enable_dp:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        else:
            raise RuntimeError(
                "You need to have more than one GPU when enable_dp = True."
            )

    if enable_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )

    if name_classifier is None:
        name_classifier = model.__class__.__name__

    # setup optimizer
    optimizer = setup_optimizer(optimizer, optimizer_params, model)

    # setup scheduler
    scheduler = setup_scheduler(scheduler, scheduler_params, optimizer)

    # setup early stopping
    if isinstance(early_stopping, int):
        early_stopping = utils.EarlyStopping(patience=early_stopping)

    try:
        model = deeplearning_train(
            dataloader_train=dataloader_train,
            dataloader_valid=dataloader_valid,
            n_epochs=n_epochs,
            model=model,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            enable_wandb_logging=enable_wandb_logging,
            wandb_params=wandb_params,
            checkpoint_fname=checkpoint_fname,
            history_fname=history_fname,
            early_stopping=early_stopping,
            enable_ddp=enable_ddp,
            enable_dp=enable_dp,
            sampler_train=sampler_train,
            rank=rank,
        )
    finally:
        if enable_ddp:
            torch.distributed.destroy_process_group()


def deeplearning_cross_subject(
    subjects_train,
    subjects_valid,
    subjects_test,
    func_get_fnames,
    criterion,
    batch_size,
    n_epochs,
    optimizer,
    *,
    optimizer_params=None,
    model=None,
    func_get_model=None,
    scheduler=None,
    scheduler_params=None,
    device="cpu",
    enable_ddp=False,
    enable_dp=False,
    num_workers=0,
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
    cross-subjectのDLを行う

    Parameters
    ==========
    subjects_train: list
    subjects_valid: list
    subjects_test: list
    func_get_fnames: callable
        subject名を引数とし，-epo.fifのファイル名リストを返す関数

        .. code-block:: python

            def fuc_get_fnames(subject):
                return [f"{subject}-R1-epo.fif", f"{subject}-R2-epo.fif"]

    criterion: instance of loss function
    batch_size: int
    n_epochs: int
    optimizer: reference to class
        インスタンスではなく，クラスへの参照を渡す

        e.g.,

        >>> optimizer = torch.optim.AdamW
    optimizer_params: dict
        optimizerへ渡す**kwargs
    model: instance of pytorch model
        func_get_modelを使う場合はNoneでOK
    func_get_model: callable
        - X_train, y_trainを引数として，modelを返す関数
        - X_trainのshape依存のmodel等で便利
        - modelに直接渡す場合はNoneでOK
    scheduler:
        インスタンスではなく，クラスへの参照を渡す

        e.g.,

        >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    scheduler_params: dict
        schedulerへ渡す**kwargs
    device: str
        "cpu", "cuda"...etc
    enable_ddp: bool
        DDP(Distributed Data Parallel)を使うか否か．Trueの場合はdevice = "cuda"じゃないといけないので注意
    num_workers: int
        - 各GPUごとにいくつのプロセスを使ってデータ読み込みを行うか
        - enable_ddp = Trueのときのみこの変更が有効になる
    func_proc_epochs: callable
        - mne.epochsを引数として，mne.epochsを返す
        - チャネルを脳波のみにする，cropする...などで使う
    label_keys: dict
        epochsからクラス情報を抜き出すのに使う

        e.g.,

        >>> label_keys={"event:left": 0, "event:right": 1}

        この場合，"event:left"のエポックが0で，"event:right"が1になる
    compile_test_subjects: bool
        - Trueのとき，複数のテストサブジェクトのデータを結合し，精度等を計算する
        - Falseの場合は，被験者ごとの結果を返す
    enable_wandb_logging: bool
        wandbでのログ保存を有効化するか
    wandb_params: dict
        - wandb.init()に渡すkwargs
        - project, name, などを入れておくと良い
    checkpoint_fname: path-like
        - checkpoint保存用ファイル名
        - pthで保存される
    history_fname: path-like
        - history保存用ファイル名
        - pkl, htmlでpandas DataFrameが保存される
    early_stopping: int or callable
        >>> early_stopping = rosoku.EarlyStopping(patience = 75)
        >>> early_stopping = 75

    Returns
    =======
    df: pandas DataFrame()
        結果が入っている
    """
    import torch

    if enable_ddp:
        params = utils.get_ddp_params()

    if enable_wandb_logging:
        if (enable_ddp and params["rank"] == 0) or (enable_ddp is False):
            import wandb

    if not isinstance(subjects_train, list) or not isinstance(subjects_test, list):
        raise ValueError("type of subjects_train and subjects_test have to be list")

    if enable_ddp and enable_dp:
        raise ValueError(
            "enable_ddp and enable_dp cannot be True at the same time. Choose one."
        )

    if (enable_ddp and device != "cuda") or (enable_dp and device != "cuda"):
        raise ValueError(
            "device have to be 'cuda' when enable_ddp = True or enable_dp = True."
        )

    # load data

    ## training data
    X_train, y_train = load_data(
        subjects_train,
        func_get_fnames,
        func_proc_epochs,
        label_keys,
        enable_euclidean_alignment,
    )
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    ## validation data
    X_valid, y_valid = load_data(
        subjects_valid,
        func_get_fnames,
        func_proc_epochs,
        label_keys,
        enable_euclidean_alignment,
    )
    X_valid = np.concatenate(X_valid, axis=0)
    y_valid = np.concatenate(y_valid, axis=0)

    ## test data
    X_test, y_test = load_data(
        subjects_test,
        func_get_fnames,
        func_proc_epochs,
        label_keys,
        enable_euclidean_alignment,
    )

    if compile_test_subjects:
        # compile y
        y_test = [np.concatenate(y_test, axis=0)]

        # compile X
        X_test = [np.concatenate(X_test, axis=0)]

        # compile subjects_test list
        subjects_test = [subjects_test]

    # data normalization
    if enable_normalization:
        X_train, X_valid, X_test, normalization_mean, normalization_std = (
            preprocessing.normalize(X_train, X_valid, X_test, return_params=True)
        )

    """
    X_train = np.zeros((40, 27, 128 * 4))
    y_train = 0
    X_valid = 0
    y_valid = 0
    X_test = 0
    y_test = 0
    """

    kwargs = {
        "optimizer_params": optimizer_params,
        "model": model,
        "func_get_model": func_get_model,
        "scheduler": scheduler,
        "scheduler_params": scheduler_params,
        "func_proc_epochs": func_proc_epochs,
        "label_keys": label_keys,
        "compile_test_subjects": compile_test_subjects,
        "enable_wandb_logging": enable_wandb_logging,
        "wandb_params": wandb_params,
        "checkpoint_fname": checkpoint_fname,
        "history_fname": history_fname,
        "early_stopping": early_stopping,
        "name_classifier": name_classifier,
        "enable_euclidean_alignment": enable_euclidean_alignment,
        "enable_normalization": enable_normalization,
        "desc": desc,
    }

    if enable_ddp:
        """
        args = (
            world_size,
            enable_ddp,
            num_workers,
            device,
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test,
            criterion,
            batch_size,
            n_epochs,
            optimizer,
            kwargs,
        )

        torch.multiprocessing.spawn(
            main_cross_subject, args=args, nprocs=world_size, join=True
        )
        """
        main(
            enable_ddp=enable_ddp,
            enable_dp=enable_dp,
            num_workers=num_workers,
            device=None,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            criterion=criterion,
            batch_size=batch_size,
            n_epochs=n_epochs,
            optimizer=optimizer,
            kwargs=kwargs,
        )
    else:
        main(
            # rank=None,
            # world_size=None,
            enable_ddp=enable_ddp,
            enable_dp=enable_dp,
            num_workers=num_workers,
            device=device,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            criterion=criterion,
            batch_size=batch_size,
            n_epochs=n_epochs,
            optimizer=optimizer,
            kwargs=kwargs,
        )

    if model is None:
        model = func_get_model(X_train, y_train)

    # device = "cuda" if enable_ddp else "cpu"
    model.to(device)

    # classify test data
    if checkpoint_fname is not None:
        # checkpoint = torch.load(f"{checkpoint_fname}")
        checkpoint = torch.load(checkpoint_fname, map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])

    (_, _, dataloader_test) = utils.nd_to_dataloader(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        device="cpu",
        batch_size=batch_size,
        enable_DS=False,
    )

    df_list = list()
    model.eval()
    with torch.no_grad():
        for dataloader, subject in zip(dataloader_test, subjects_test):
            # accuracy = utils.accuracy_score_dataloader(model, dataloader_test)

            preds, labels, logits, probas = utils.get_predictions(
                model,
                dataloader,
                device=device,
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
                df_results["normalization_mean"] = [normalization_mean.flatten()]
                df_results["normalization_std"] = [normalization_std.flatten()]

            df_list.append(df_results)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    return df


def deeplearning(
    keywords_train,
    keywords_valid,
    keywords_test,
    func_load_epochs=None,
    func_load_ndarray=None,
    criterion=torch.nn.CrossEntropyLoss(),
    batch_size=64,
    n_epochs=500,
    optimizer=torch.optim.AdamW,
    *,
    func_proc_mode="per_split",
    func_proc_epochs=None,
    func_proc_ndarray=None,
    func_convert_epochs_to_ndarray=utils.convert_epochs_to_ndarray,
    optimizer_params=None,
    model=None,
    func_get_model=None,
    scheduler=None,
    scheduler_params=None,
    device="cpu",
    enable_ddp=False,
    enable_dp=False,
    num_workers=0,
    enable_wandb_logging=False,
    wandb_params=None,
    checkpoint_fname=None,
    history_fname=None,
    early_stopping=None,
    name_classifier=None,
    enable_normalization=False,
    saliency_map=False,
    label_keys=None,
    seed=None,
    desc=None,
):
    """
    汎用的なdeeplearning用関数

    Parameters
    ----------
    subjects_train: list
    subjects_valid: list
    subjects_test: list
    func_get_fnames: callable
        subject名を引数とし，-epo.fifのファイル名リストを返す関数

        .. code-block:: python

            def fuc_get_fnames(subject):
                return [f"{subject-R1-epo.fif}", f"{subject-R2}-epo.fif"]

    criterion: instance of loss function
    batch_size: int
    n_epochs: int
    optimizer: reference to class
        - インスタンスではなく，クラスへの参照を渡す

        e.g.,

        >>> optimizer = torch.optim.AdamW

    optimizer_params: dict
        optimizerへ渡す**kwargs
    model: instance of pytorch model
        func_get_modelを使う場合はNoneでOK
    func_get_model: callable
        - X_train, y_trainを引数として，modelを返す関数
        - X_trainのshape依存のmodel等で便利
        - modelに直接渡す場合はNoneでOK
    scheduler:
        インスタンスではなく，クラスへの参照を渡す

        e.g.,

        >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    scheduler_params: dict
        schedulerへ渡す**kwargs
    device: str
        "cpu", "cuda"...etc
    enable_ddp: bool
        DDP(Distributed Data Parallel)を使うか否か．Trueの場合はdevice = "cuda"じゃないといけないので注意
    num_workers: int
        - 各GPUごとにいくつのプロセスを使ってデータ読み込みを行うか
        - enable_ddp = Trueのときのみこの変更が有効になる
    func_proc_epochs: callable
        - mne.epochsを引数として，mne.epochsを返す
        - チャネルを脳波のみにする，cropする...などで使う
    label_keys: dict
        epochsからクラス情報を抜き出すのに使う

        e.g.,

        >>> label_keys={"event:left": 0, "event:right": 1}

        この場合，"event:left"のエポックが0で，"event:right"が1になる
    compile_test: bool
        - Trueのとき，複数のテストサブジェクトのデータを結合し，精度等を計算する
        - Falseの場合は，被験者ごとの結果を返す
    enable_wandb_logging: bool
        wandbでのログ保存を有効化するか
    wandb_params: dict
        - wandb.init()に渡すkwargs
        - project, name, などを入れておくと良い
    checkpoint_fname: path-like
        - checkpoint保存用ファイル名
        - pthで保存される
    history_fname: path-like
        - history保存用ファイル名
        - pkl, htmlでpandas DataFrameが保存される
    early_stopping: int or callable

        >>> early_stopping = rosoku.EarlyStopping(patience = 75)
        >>> early_stopping = 75

    Returns
    -------
    df: pandas DataFrame()
        結果が入ってる
    """
    import torch

    if enable_ddp:
        params = utils.get_ddp_params()

    if enable_wandb_logging:
        if (enable_ddp and params["rank"] == 0) or (enable_ddp is False):
            import wandb

    if enable_ddp and enable_dp:
        raise ValueError(
            "enable_ddp and enable_dp cannot be True at the same time. Choose one."
        )

    if (enable_ddp and device != "cuda") or (enable_dp and device != "cuda"):
        raise ValueError(
            "device have to be 'cuda' when enable_ddp = True or enable_dp = True."
        )

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        g = torch.Generator()
        g.manual_seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # load data

    X_train, X_valid, X_test, y_train, y_valid, y_test = utils.load_data(
        keywords_train=keywords_train,
        keywords_valid=keywords_valid,
        keywords_test=keywords_test,
        func_load_epochs=func_load_epochs,
        func_load_ndarray=func_load_ndarray,
        func_proc_epochs=func_proc_epochs,
        func_proc_ndarray=func_proc_ndarray,
        func_proc_mode=func_proc_mode,
        func_convert_epochs_to_ndarray=func_convert_epochs_to_ndarray,
    )

    if len(keywords_test) != len(X_test):
        raise RuntimeError("len(keywords_test) != len(X_test)")

    # data normalization
    if enable_normalization:
        X_train, X_valid, X_test, normalization_mean, normalization_std = (
            preprocessing.normalize(X_train, X_valid, X_test, return_params=True)
        )

    kwargs = {
        "optimizer_params": optimizer_params,
        "model": model,
        "func_get_model": func_get_model,
        "scheduler": scheduler,
        "scheduler_params": scheduler_params,
        "func_proc_epochs": func_proc_epochs,
        "enable_wandb_logging": enable_wandb_logging,
        "wandb_params": wandb_params,
        "checkpoint_fname": checkpoint_fname,
        "history_fname": history_fname,
        "early_stopping": early_stopping,
        "name_classifier": name_classifier,
        "seed": seed,
        "desc": desc,
    }

    if enable_ddp:
        main(
            enable_ddp=enable_ddp,
            enable_dp=enable_dp,
            num_workers=num_workers,
            device=None,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            criterion=criterion,
            batch_size=batch_size,
            n_epochs=n_epochs,
            optimizer=optimizer,
            kwargs=kwargs,
        )
    else:
        main(
            enable_ddp=enable_ddp,
            enable_dp=enable_dp,
            num_workers=num_workers,
            device=device,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            criterion=criterion,
            batch_size=batch_size,
            n_epochs=n_epochs,
            optimizer=optimizer,
            kwargs=kwargs,
        )

    if model is None:
        model = func_get_model(X_train, y_train)

    # device = "cuda" if enable_ddp else "cpu"
    model.to(device)

    # classify test data
    if checkpoint_fname is not None:
        # checkpoint = torch.load(f"{checkpoint_fname}")
        checkpoint = torch.load(checkpoint_fname, map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])

    (_, _, dataloader_test) = utils.nd_to_dataloader(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        device="cpu",
        batch_size=batch_size,
        enable_DS=False,
        generator=seed,
    )

    if isinstance(dataloader_test, list) is False:
        dataloader_test = [dataloader_test]

    df_list = list()
    model.eval()

    if saliency_map:
        saliency_data = []
        for idx, dataloader in enumerate(dataloader_test):
            saliency_data.append({})
            if label_keys is None:

                classes = np.unique(y_test).tolist()

                label_keys = {f"{c}": c for c in classes}

            for class_label, c in label_keys.items():
                saliency_data[idx][class_label] = attribution.saliency_map(
                    model, dataloader, device, class_index=c
                )

    with torch.no_grad():
        for idx, (dataloader, keywords_test_single) in enumerate(
            zip(dataloader_test, keywords_test)
        ):

            preds, labels, logits, probas = utils.get_predictions(
                model,
                dataloader,
                device=device,
            )

            accuracy = sklearn.metrics.accuracy_score(labels, preds)
            f1 = sklearn.metrics.f1_score(labels, preds)
            bacc = sklearn.metrics.balanced_accuracy_score(labels, preds)

            df_results = pd.DataFrame()
            df_results["keywords_train"] = [keywords_train]
            df_results["keywords_valid"] = [keywords_valid]
            df_results["keywords_test"] = [keywords_test_single]
            df_results["classifier"] = [name_classifier]
            df_results["accuracy"] = [accuracy]
            df_results["labels"] = [labels]
            df_results["preds"] = [preds]
            df_results["probas"] = [probas]
            df_results["logits"] = [logits]
            df_results["desc"] = [desc]
            if enable_normalization:
                df_results["normalization_mean"] = [normalization_mean.flatten()]
                df_results["normalization_std"] = [normalization_std.flatten()]

            if saliency_map:
                df_results["saliency_map"] = [saliency_data[idx]]

            if enable_wandb_logging:
                if (enable_ddp and params["rank"] == 0) or (enable_ddp is False):
                    table = wandb.Table(columns=["id", "labels", "preds"])
                    for idx, (label, pred) in enumerate(zip(labels, preds)):
                        table.add_data(idx, label, pred)

                    wandb.log(
                        {
                            "test/accuracy": accuracy,
                            "test/bacc": bacc,
                            "test/f1": f1,
                            "predictions": table,
                        }
                    )

            df_list.append(df_results)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    if enable_wandb_logging:

        if enable_ddp:
            ddp_params = utils.get_ddp_params()
            rank = ddp_params["rank"]

        if (enable_ddp and rank == 0) or (enable_ddp is False):
            wandb.finish()

    return df
