"""Use EEG plus channel positions with dictionary batches and a custom Step.

This illustrative network requires no external EEG library. Replace it with
your EEGSetTransformer and its get_positions(ch_names) output for real data.
The dictionary tensors are moved recursively to Experiment's device.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from rosoku import Callback, Experiment, Stage, Step


class EEGDataset(Dataset):
    def __init__(self):
        self.eeg = torch.randn(24, 3, 32)
        self.labels = (self.eeg[:, 0].mean(-1) > 0).long()
        self.positions = torch.tensor([[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {'eeg': self.eeg[index], 'pos': self.positions, 'label': self.labels[index]}


class PositionAwareEEG(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(3, 8)
        self.classifier = nn.Linear(8, 2)

    def forward(self, eeg, pos):
        # Three features per channel: temporal mean and 2-D channel position.
        features = torch.cat((eeg.mean(-1, keepdim=True), pos), dim=-1)
        pooled = torch.relu(self.embedding(features)).mean(1)
        return self.classifier(pooled)


class EEGStep(Step):
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, state):
        return state.model(state.batch['eeg'], state.batch['pos'])

    def compute_loss(self, state):
        return self.criterion(state.output, state.batch['label'])


class SampleMetrics(Callback):
    """Show sample-weighted loss/accuracy without retaining computation graphs."""
    def reset(self):
        self.total = self.correct = self.loss_sum = 0

    def on_train_epoch_start(self, state):
        self.reset()

    def on_valid_epoch_start(self, state):
        self.reset()

    def on_batch_end(self, state):
        y = state.batch['label']
        self.total += len(y)
        self.correct += (state.output.detach().argmax(1) == y).sum().item()
        self.loss_sum += state.loss.detach().item() * len(y)

    def finish(self, state):
        state.metrics[f'{state.phase}/sample_loss'] = self.loss_sum / self.total
        state.metrics[f'{state.phase}/accuracy'] = self.correct / self.total

    def on_train_epoch_end(self, state):
        self.finish(state)

    def on_valid_epoch_end(self, state):
        self.finish(state)

    def on_epoch_end(self, state):
        print(f'Epoch {state.epoch + 1}: {state.metrics}')


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    data = EEGDataset()
    train = DataLoader(torch.utils.data.Subset(data, range(18)), batch_size=8, shuffle=True)
    valid = DataLoader(torch.utils.data.Subset(data, range(18, 24)), batch_size=4)
    experiment = Experiment(PositionAwareEEG(), stages=[Stage(
        name='eeg_training', epochs=2, train_step=EEGStep(),
        optimizer=lambda p: torch.optim.AdamW(p, lr=0.01), validate_every=1,
    )], callbacks=[SampleMetrics()], device='cpu')
    experiment.fit(train, valid)


if __name__ == '__main__':
    main()
