"""One-stage supervised training with validation every epoch.

The caller defines data, split, model, and randomness; rosoku runs the loops.
Set validate_every=None and omit valid_loader to disable validation, or use
validate_every=2 to validate after epochs 2, 4, ... within this stage.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from rosoku import Callback, Experiment, Stage, SupervisedStep


class History(Callback):
    """Keep detached epoch losses outside the transient State dictionaries."""
    def on_experiment_start(self, state):
        self.rows = []

    def on_epoch_end(self, state):
        row = {'epoch': state.epoch + 1, **state.metrics}
        self.rows.append(row)
        print(row)


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    x = torch.randn(80, 6)
    y = (x[:, 0] + x[:, 1] > 0).long()
    train = DataLoader(TensorDataset(x[:64], y[:64]), batch_size=16, shuffle=True)
    valid = DataLoader(TensorDataset(x[64:], y[64:]), batch_size=16)
    model = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 2))
    history = History()
    experiment = Experiment(model, stages=[Stage(
        name='training', epochs=3,
        train_step=SupervisedStep(nn.CrossEntropyLoss()),
        optimizer=lambda parameters: torch.optim.AdamW(parameters, lr=0.01),
        scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=2),
        validate_every=1,
    )], callbacks=[history], device='cpu')
    state = experiment.fit(train, valid)
    assert len(history.rows) == 3
    assert state.global_step == 12
    print(f'Completed {state.global_epoch} epochs and {state.global_step} train batches')


if __name__ == '__main__':
    main()
