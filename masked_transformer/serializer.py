import functools

import torch


# We refer to https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties  # noqa: E501


def _rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(_rgetattr(obj, pre) if pre else obj, post, val)


def _rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def load_pth_renamed(model, params, correspondence):
    """Load PyTorch weight file `*.pth`.
    Since parameter names in weight file are sometimes different from model
    parameter names, correspondence argument is used for its compatibility.
    If some parameter names don't exist in model or parameter shape doesn't
    match, such parameters will be ignored.
    Args:
        model (torch.nn.Module): Model.
        params (dict of torch.Tensor): Weight parameters.
        correspondence (dict): Correspondence dict of parameters between
            weight file and model. Its key are parameter names in weight file,
            and its values are those in model.
    """
    not_parameters = set(['running_mean', 'running_var'])
    for name, param in params.items():
        if correspondence.get(name) and \
                _rgetattr(model, correspondence[name]).shape == param.shape:
            if name.split('.')[-1] not in not_parameters:
                param = torch.nn.Parameter(param)
            _rsetattr(model, correspondence[name], param)
