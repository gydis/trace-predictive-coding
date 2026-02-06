"""Predictive-coding layers with explicit free-energy.

States are activations (post-nonlinearity). Each predictive layer owns a
mapping that predicts the layer below; free-energy is a sum of prediction
errors across these mappings.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

Tensor = torch.Tensor
ActivationFn = Callable[[Tensor], Tensor]


def _shape(n_units: Union[int, Sequence[int]], batch_size: int) -> Tuple[int, ...]:
    if isinstance(n_units, int):
        return (batch_size, n_units)
    if isinstance(n_units, tuple):
        return (batch_size, *n_units)
    return (batch_size, *tuple(n_units))


def _softplus_inverse(value: Tensor) -> Tensor:
    return torch.log(torch.expm1(value))


class PCLayer(nn.Module):
    """Base class for predictive-coding layers with a mutable state."""

    def __init__(
        self,
        n_units: Union[int, Sequence[int]],
        *,
        batch_size: int = 1,
        init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.init_scale = init_scale
        self.clamped = False
        self.shape = _shape(n_units, batch_size)
        self.n_units = int(torch.tensor(self.shape[1:]).prod().item())
        self.register_buffer("state", torch.zeros(self.shape))

    def state_value(self) -> Tensor:
        """Return the representation used in prediction errors.

        By default this is just ``state``. Subclasses can override this to impose
        constraints (e.g. simplex via softmax) while still inferring unconstrained
        internal variables.
        """
        return self.state

    def reset(self, batch_size: Optional[int] = None, init_scale: Optional[float] = None) -> None:
        """Reset the layer state, optionally changing batch size or init scale."""
        if batch_size is not None:
            self.batch_size = batch_size
            self.shape = (batch_size, *self.shape[1:])
        if init_scale is None:
            init_scale = self.init_scale
        if init_scale and init_scale > 0:
            state = init_scale * torch.randn(self.shape, device=self.state.device)
        else:
            state = torch.zeros(self.shape, device=self.state.device)
        self.state = state

    def clamp(self, state: Tensor) -> None:
        """Clamp the layer state to a fixed value."""
        self.state = state.detach()
        self.clamped = True

    def release_clamp(self) -> None:
        """Release the clamp so the state can be inferred."""
        self.clamped = False

    def detach_state(self) -> None:
        """Detach the state tensor to drop autograd history."""
        self.state = self.state.detach()


class InputLayer(PCLayer):
    """Input layer holding a clamped or inferred state."""

    def extra_repr(self) -> str:
        return f"shape={self.shape}"


class PredictiveLayer(PCLayer):
    """Layer with a generative mapping that predicts the layer below."""

    def predict_down(self) -> Tensor:
        """Predict the state of the layer below."""
        raise NotImplementedError

    def prediction_error(self, lower_state: Tensor) -> Tensor:
        """Compute prediction error for the layer below."""
        prediction = self.predict_down()
        error = lower_state - prediction
        self.prediction = prediction
        self.pred_error = error
        return error


class LinearLayer(PredictiveLayer):
    """Predictive layer with a linear generative mapping."""

    def __init__(
        self,
        n_in: int,
        n_units: int,
        *,
        bias: bool = True,
        batch_size: int = 1,
        nonlinearity: Optional[ActivationFn] = F.relu,
        noise_std: float = 0.0,
        init_scale: float = 0.0,
        learn_precision: bool = True,
        precision_init: float = 1.0,
        precision_eps: float = 1e-6,
        weight_norm: Optional[float] = None,
    ) -> None:
        super().__init__(n_units, batch_size=batch_size, init_scale=init_scale)
        self.n_in = n_in
        self.n_units = n_units
        self.nonlinearity = nonlinearity
        self.noise_std = noise_std
        self.learn_precision = learn_precision
        self.precision_eps = precision_eps
        self.weight_norm = weight_norm

        self.weight = nn.Parameter(torch.empty(n_units, n_in))
        nn.init.kaiming_uniform_(self.weight, mode="fan_in", nonlinearity="relu")

        if bias:
            self.bias = nn.Parameter(torch.empty(n_in))
            nn.init.constant_(self.bias, 0)
        else:
            self.bias = None

        precision_init = float(precision_init)
        if precision_init <= 0:
            raise ValueError("precision_init must be positive.")
        precision_param = _softplus_inverse(torch.tensor(precision_init))
        if learn_precision:
            self.precision_param = nn.Parameter(precision_param)
        else:
            self.register_buffer("precision_param", precision_param)

    def predict_down(self) -> Tensor:
        prediction = F.linear(self.state_value(), self.weight.T, bias=self.bias)
        if self.nonlinearity is not None:
            prediction = self.nonlinearity(prediction)
        if self.noise_std and self.noise_std > 0:
            prediction = prediction + self.noise_std * torch.randn_like(prediction)
        return prediction

    def precision(self) -> Tensor:
        return F.softplus(self.precision_param) + self.precision_eps

    def extra_repr(self) -> str:
        return f"in_shape={self.n_in}, out_shape={self.n_units}"
    
    def normalize_weights(self) -> None:
        if self.weight_norm is not None:
            with torch.no_grad():
                norms = self.weight.norm(p=2, dim=1, keepdim=True)
                self.weight.mul_(self.weight_norm / norms)



class MiddleLayer(LinearLayer):
    """Alias for a predictive linear layer."""


class OutputLayer(LinearLayer):
    """Alias for a predictive linear layer at the top of a hierarchy."""