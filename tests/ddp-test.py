import rosoku
import braindecode

import torch


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
    n_epochs = 10
    batch_size = 64
    patience = 75

    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    early_stopping = rosoku.utils.EarlyStopping(patience=patience)

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


if __name__ == "__main__":
    main_cross_subject()
