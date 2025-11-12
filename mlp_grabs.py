import torch
import torch.nn as nn

def get_Win(model: nn.Module, *, detach: bool = True, **kwargs):
    Win = model.input_layer.weight
    b_in = model.input_layer.bias

    if detach:
        Win = Win.detach().clone()
        b_in = b_in.detach().clone() if b_in is not None else None

    return Win, b_in


def get_Wout(model: nn.Module, *, detach: bool = True, **kwargs):
    Wout = model.output_layer.weight
    bout = model.output_layer.bias

    if detach:
        Wout = Wout.detach().clone()
        bout = bout.detach().clone() if bout is not None else None

    return Wout, bout


def get_W_gram(W: torch.Tensor, concatenate_outside: bool = True, **kwargs):
    """
    Concatenate_outside: if True, computes W^T W (so gram matrix in output space)
    """
    return W.T @ W if concatenate_outside else W @ W.T


def get_W_ii(W: torch.Tensor, i: int=None, monomial=None, **kwargs):
    """
    So Nintendo doesn't sue us.

    Assumes W is a gram matrix.
    """

    if monomial is not None and i is None:
        eyes = [int(k) for k in monomial.basis().keys()]
        return [W[i, i].item() for i in eyes]
    Wii = W[i, i] #grab i if specified
    return Wii.item()


def get_W_trace(W: torch.Tensor, **kwargs):
    """
    Check the trace of W_ii.

    Assumes W is a gram matrix.
    """
    return torch.trace(W).item()