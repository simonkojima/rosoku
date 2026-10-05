import unittest
import torch
from torch import nn
from rosoku import Callback, Experiment, Stage, State, Step, SupervisedStep
from rosoku.parameters import configure_trainable_parameters


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.classifier = nn.Linear(4, 1)

    def forward(self, x):
        return self.classifier(torch.tanh(self.encoder(x)))


class Recorder(Callback):
    def __init__(self):
        self.valid_epochs = []
        self.starts = []
        self.ends = []
        self.backward = 0
        self.batches = []

    def on_stage_start(self, state):
        self.starts.append((state.optimizer, state.scheduler,
                            {n: p.detach().clone() for n, p in state.model.named_parameters()},
                            {n: p.requires_grad for n, p in state.model.named_parameters()}))

    def on_stage_end(self, state):
        self.ends.append({n: p.detach().clone() for n, p in state.model.named_parameters()})

    def on_valid_epoch_start(self, state):
        assert not state.model.training
        assert not torch.is_grad_enabled()
        self.valid_epochs.append(state.epoch)

    def on_after_backward(self, state):
        assert state.phase == 'train'
        self.backward += 1

    def on_batch_end(self, state):
        self.batches.append(state.phase)
        if state.phase == 'valid':
            assert not state.output.requires_grad


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = Model()
        self.loader = [(torch.randn(5, 3), torch.randn(5, 1)) for _ in range(2)]
        self.step = SupervisedStep(nn.MSELoss())

    def stage(self, **kwargs):
        args = dict(name='train', epochs=3, train_step=self.step,
                    optimizer=lambda ps: torch.optim.SGD(ps, lr=0.1), validate_every=None)
        args.update(kwargs)
        return Stage(**args)

    def test_validation_frequencies_and_single_stage(self):
        for frequency, expected in [(None, []), (1, [0, 1, 2]), (2, [1]), (5, [])]:
            with self.subTest(frequency=frequency):
                recorder = Recorder()
                before = self.model.classifier.weight.detach().clone()
                state = Experiment(self.model, [self.stage(validate_every=frequency)], [recorder]).fit(
                    self.loader, self.loader if frequency is not None else None)
                self.assertEqual(recorder.valid_epochs, expected)
                self.assertEqual(state.global_step, 6)
                self.assertEqual(state.global_epoch, 3)
                self.assertEqual(recorder.backward, 6)
                self.assertFalse(torch.equal(before, self.model.classifier.weight))
                self.assertEqual(recorder.batches.count('valid'), 2 * len(expected))

    def test_freeze_unfreeze_and_fresh_factories(self):
        recorder = Recorder()
        stages = [self.stage(name='probe', trainable=['classifier'], epochs=1,
                             scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.5)),
                  self.stage(name='fine', trainable='all', epochs=1)]
        Experiment(self.model, stages, [recorder]).fit(self.loader)
        first, second = recorder.starts
        self.assertIsNot(first[0], second[0])
        self.assertIsNone(second[1])
        self.assertAlmostEqual(first[0].param_groups[0]['lr'], 0.05)
        self.assertEqual(len(first[0].param_groups[0]['params']), 2)
        self.assertEqual(len(second[0].param_groups[0]['params']), 4)
        for name in first[2]:
            if name.startswith('encoder'):
                self.assertFalse(first[3][name])
                self.assertTrue(torch.equal(first[2][name], recorder.ends[0][name]))
                self.assertFalse(torch.equal(second[2][name], recorder.ends[1][name]))
            else:
                self.assertFalse(torch.equal(first[2][name], recorder.ends[0][name]))
            self.assertTrue(second[3][name])

    def test_callable_and_named_selectors(self):
        for selector in ['classifier', ['classifier', 'classifier'],
                         lambda m: list(m.classifier.parameters()) * 2]:
            selected = configure_trainable_parameters(self.model, selector)
            self.assertEqual(len(selected), 2)
            self.assertFalse(self.model.encoder.weight.requires_grad)
        for selector in ['missing', [], lambda m: [nn.Parameter(torch.zeros(1))]]:
            with self.assertRaises(ValueError):
                configure_trainable_parameters(self.model, selector)

    def test_invalid_config_and_missing_validation_loader(self):
        for field in ['epochs', 'validate_every']:
            for value in [0, -1, 1.5, True]:
                with self.assertRaises(ValueError):
                    self.stage(**{field: value})
        with self.assertRaises(ValueError):
            Experiment(self.model, []).fit(self.loader)
        with self.assertRaises(ValueError):
            Experiment(self.model, [self.stage(validate_every=1)]).fit(self.loader)
        with self.assertRaises(ValueError):
            Experiment(self.model, [self.stage()]).fit([])

    def test_custom_validation_step_and_plateau(self):
        class Valid(Step):
            def forward(inner, state):
                return state.model(state.batch[0])
            def compute_loss(inner, state):
                return state.output.sum() * 0 + 7
        state = Experiment(self.model, [self.stage(valid_step=Valid(), validate_every=2,
            scheduler=lambda opt: torch.optim.lr_scheduler.ReduceLROnPlateau(opt))]).fit(self.loader, self.loader)
        self.assertNotIn('valid/loss', state.metrics)  # no stale epoch-1 validation
        self.assertEqual(state.scheduler.last_epoch, 3)

    def test_stop_current_stage_and_repeat_fit(self):
        class Stop(Callback):
            def on_train_batch_end(inner, state):
                state.should_stop = True
        experiment = Experiment(self.model, [self.stage(name='a'), self.stage(name='b')], [Stop()])
        for _ in range(2):
            state = experiment.fit(self.loader)
            self.assertEqual(state.global_step, 2)
            self.assertEqual(state.global_epoch, 2)
            self.assertEqual(state.stage.name, 'b')

    def test_callback_order(self):
        events = []
        recorder = Callback()
        for event in ['experiment_start', 'stage_start', 'epoch_start', 'train_epoch_start',
                      'batch_start', 'train_batch_start', 'after_forward', 'after_loss',
                      'after_backward', 'before_optimizer_step', 'after_optimizer_step',
                      'train_batch_end', 'batch_end', 'train_epoch_end', 'epoch_end',
                      'stage_end', 'experiment_end']:
            setattr(recorder, 'on_' + event, lambda state, event=event: events.append(event))
        Experiment(self.model, [self.stage(epochs=1)], [recorder]).fit(self.loader[:1])
        self.assertEqual(events, ['experiment_start', 'stage_start', 'epoch_start', 'train_epoch_start',
            'batch_start', 'train_batch_start', 'after_forward', 'after_loss', 'after_backward',
            'before_optimizer_step', 'after_optimizer_step', 'train_batch_end', 'batch_end',
            'train_epoch_end', 'epoch_end', 'stage_end', 'experiment_end'])


if __name__ == '__main__':
    unittest.main()
