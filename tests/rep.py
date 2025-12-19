import numpy as np
import random
import torch
from pathlib import Path

import rosoku
import braindecode


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def callback_get_model(X, y):
    _, n_chans, n_times = X.shape
    F1 = 4
    D = 2
    F2 = F1 * D

    model = braindecode.models.EEGNet(
        n_chans=n_chans,
        n_outputs=2,
        n_times=n_times,
        F1=F1,
        D=D,
        F2=F2,
        drop_prob=0.5,
    )

    return model


fname = Path("~/rosoku-test/data.npz").expanduser()
data = np.load(fname)
seed = 42

set_seed(seed)

X_train = data["X_train"]
X_valid = data["X_valid"]
X_test = data["X_test"]

y_train = data["y_train"]
y_valid = data["y_valid"]
y_test = data["y_test"]

device = "cuda"

dataloader_train, dataloader_valid, dataloader_test = (
    rosoku.utils.ndarray_to_dataloader(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        8,
        device=device,
        generator=seed,
    )
)

model = callback_get_model(X_train, y_train)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(lr=1e-3, params=model.parameters(), weight_decay=1e-2)

model.train()
for X, y in dataloader_train:
    X = X.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"{loss.item()=}")
