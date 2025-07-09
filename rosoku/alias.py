import warnings

# EarlyStopping = utils.EarlyStopping

from . import utils


class EarlyStopping(utils.EarlyStopping):
    def __init__(self, patience=3, min_delta=0.0):
        warnings.warn(
            "rosoku.utils.normalize() is deprecated and will be removed in a future version. Use rosoku.preprocessing.normalize() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__(patience=patience, min_delta=min_delta)
