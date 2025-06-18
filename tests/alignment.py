import rosoku
from pathlib import Path
import mne

epochs_base = Path(
    "~/Documents/datasets/dreyer_2023/derivatives/epochs/l_freq-8.0_h_freq-30.0_resample-128"
)

subject = "A1"
epochs_subject_1 = mne.read_epochs(
    epochs_base / subject / f"sub-{subject}_online-epo.fif"
)

subject = "A2"
epochs_subject_2 = mne.read_epochs(
    epochs_base / subject / f"sub-{subject}_online-epo.fif"
)

subject = "A3"
epochs_subject_3 = mne.read_epochs(
    epochs_base / subject / f"sub-{subject}_online-epo.fif"
)

a = (0, 0, 1)
print(epochs_subject_1.get_data()[a])
X = rosoku.tl.euclidean_alignment(epochs_subject_1.get_data())
print(X[a])

X_train = X
import numpy as np

n_trials, n_channels, n_times = X_train.shape
mean = np.mean(X_train.transpose((1, 0, 2)).reshape((n_channels, -1)), axis=1)
std = np.std(X_train.transpose((1, 0, 2)).reshape((n_channels, -1)), axis=1)

print(mean[a[1]])
print(std[a[1]])

X, _, _ = rosoku.utils.normalize(
    epochs_subject_1.get_data(),
    epochs_subject_2.get_data(),
    epochs_subject_3.get_data(),
)

print(X[a])

# scaling
