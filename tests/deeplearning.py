import torch
import braindecode

import rosoku


def func_proc_epochs(epochs):
    epochs = epochs.pick(picks="eeg").crop(tmin=0.25, tmax=5.0)
    return epochs


def func_get_fnames(subject):
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


def func_get_model(X, y):
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

    return model


def main_cross_subject():
    # setup mpdels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 0.001
    weight_decay = 0.0005
    n_epochs = 1000
    batch_size = 64
    patience = 75

    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    early_stopping = rosoku.EarlyStopping(patience=patience)

    returns = rosoku.deeplearning_cross_subject(
        subjects_train=["A1", "A2"],
        subjects_valid=["A3", "A4"],
        subjects_test=["A5", "A6"],
        func_get_fnames=func_get_fnames,
        func_proc_epochs=func_proc_epochs,
        func_get_model=func_get_model,
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

    print(returns)


def test_early_stopping():
    early_stopping = rosoku.EarlyStopping(patience=3)

    def check(early_stopping, loss):
        is_triggered = early_stopping(loss)
        txt = f"is_triggered: {is_triggered},\tbest loss: {early_stopping.best_loss}, counter: {early_stopping.counter}"
        print(txt)

        return is_triggered

    loss_list = [
        100,
        200,
        300,
        400,
        500,
        400,
        300,
        200,
        100,
        10,
        9,
        8,
        20,
        10,
        9,
        12,
        13,
    ]

    for loss in loss_list:
        if check(early_stopping, loss):
            early_stopping.initialize()


if __name__ == "__main__":
    main_cross_subject()
    # test_early_stopping()
