"""Linear probing followed by fine tuning using ordinary rosoku stages.

This small encoder stands in for a pretrained foundation model. In a real
experiment, load pretrained backbone weights before creating Experiment.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from rosoku import Callback, Experiment, Stage, SupervisedStep


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(6, 8), nn.BatchNorm1d(8), nn.ReLU())
        self.classifier = nn.Linear(8, 2)

    def forward(self, x):
        return self.classifier(self.encoder(x))


class ObserveStages(Callback):
    """Keep the probe encoder in eval mode and verify its parameters/buffers."""
    def on_stage_start(self, state):
        self.before = {name: value.clone() for name, value in state.model.encoder.state_dict().items()}
        names = [name for name, parameter in state.model.named_parameters() if parameter.requires_grad]
        print(state.stage.name, 'trainable:', names)

    def on_train_epoch_start(self, state):
        # Called after model.train(): fix BatchNorm statistics during probing.
        if state.stage.name == 'linear_probe':
            state.model.encoder.eval()

    def on_stage_end(self, state):
        after = state.model.encoder.state_dict()
        if state.stage.name == 'linear_probe':
            assert all(torch.equal(value, after[name]) for name, value in self.before.items())
            print('Probe encoder parameters and BatchNorm buffers stayed fixed')
        else:
            assert all(p.requires_grad for p in state.model.parameters())
            assert not torch.equal(self.before['0.weight'], after['0.weight'])
            print('Fine tuning unfroze and updated the encoder')


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    x = torch.randn(80, 6)
    y = (x[:, 0] > 0).long()
    train = DataLoader(TensorDataset(x[:64], y[:64]), batch_size=16, shuffle=True)
    valid = DataLoader(TensorDataset(x[64:], y[64:]), batch_size=16)
    step = SupervisedStep(nn.CrossEntropyLoss())
    stages = [
        Stage(name='linear_probe', epochs=2, train_step=step, trainable=['classifier'],
              optimizer=lambda p: torch.optim.AdamW(p, lr=0.01), validate_every=1),
        Stage(name='fine_tune', epochs=2, train_step=step, trainable='all',
              optimizer=lambda p: torch.optim.AdamW(p, lr=0.001),
              scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.5),
              validate_every=2),
    ]
    state = Experiment(Classifier(), stages, [ObserveStages()], device='cpu').fit(train, valid)
    assert state.global_epoch == 4


if __name__ == '__main__':
    main()
