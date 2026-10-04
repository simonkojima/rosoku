"""Stage parameter selection and freeze/unfreeze."""
from collections.abc import Sequence
from torch import nn
from .types import TrainableSelector

def configure_trainable_parameters(model: nn.Module, selector: TrainableSelector) -> list[nn.Parameter]:
    """Select all parameters, named submodules, or a callable iterable.

    Validate before modifying the model, deduplicate shared parameters, and clear
    stale gradients when switching stages. This freezes parameters, not buffers;
    use a stage/train-epoch callback to put frozen BatchNorm modules in eval mode.
    """
    all_params = list(model.parameters())
    if isinstance(selector, str) and selector == "all":
        selected = all_params
    elif callable(selector):
        selected = list(selector(model))
    elif isinstance(selector, (str, Sequence)):
        names = [selector] if isinstance(selector, str) else selector
        modules = dict(model.named_modules())
        selected = []
        for name in names:
            if name not in modules:
                raise ValueError(f"Unknown module: {name!r}")
            selected.extend(modules[name].parameters())
    else:
        raise TypeError("trainable must be 'all', module names, or a callable")
    known = {id(p) for p in all_params}
    if any(not isinstance(p, nn.Parameter) or id(p) not in known for p in selected):
        raise ValueError("Selector must return parameters belonging to the model")
    selected = list({id(p): p for p in selected}.values())
    if not selected:
        raise ValueError("The stage must select at least one parameter")
    selected_ids = {id(p) for p in selected}
    for p in all_params:
        p.requires_grad_(id(p) in selected_ids)
        p.grad = None
    return selected
