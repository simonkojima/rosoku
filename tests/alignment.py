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

rosoku.tl.euclidean_alignment(epochs_subject_1.get_data())
