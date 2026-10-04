"""Stage parameter selection and freeze/unfreeze."""
from collections.abc import Sequence
from torch import nn
from .types import TrainableSelector

def configure_trainable_parameters(model: nn.Module, selector: TrainableSelector) -> list[nn.Parameter]:
    """Resolve a stage selector and update parameter trainability.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters are selected.
    selector : TrainableSelector
        ``"all"``, a named submodule, a sequence of named submodules, or a callable
        returning an iterable of parameters belonging to this model.

    Returns
    -------
    list of torch.nn.Parameter
        Selected parameters, deduplicated in selection order.

    Raises
    ------
    ValueError
        If a submodule is unknown, the selection is empty, or the callable returns
        a foreign parameter or an object that is not a parameter.
    TypeError
        If the selector has an unsupported type.

    Notes
    -----
    Validation occurs before model flags are changed. All model parameters have
    stale gradients cleared; only selected parameters have ``requires_grad=True``.
    Buffers and module train/eval mode are unaffected. This helper is available
    from ``rosoku.parameters`` and is not re-exported from the package root."""
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
