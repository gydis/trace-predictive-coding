"""Autograd-based predictive-coding models with explicit free-energy."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from pc_layers import InputLayer, MiddleLayer, OutputLayer, PredictiveLayer

Tensor = torch.Tensor
StepSize = Union[float, Sequence[float], Dict[str, float]]


@dataclass
class PCInferenceConfig:
    n_steps: int = 5
    step_size: StepSize = 0.1
    energy_reduction: Optional[str] = None
    until_converged: bool = False
    convergence_tol: float = 1e-3
    max_steps: Optional[int] = None
    min_steps: int = 1
    state_optimizer: Optional[str] = None
    state_optimizer_kwargs: Optional[Dict[str, float]] = None


def _as_sequential(layers: Union[List[nn.Module], Dict[str, nn.Module], nn.Sequential]) -> nn.Sequential:
    if isinstance(layers, (dict, OrderedDict)):
        names = list(layers.keys())
        modules = list(layers.values())
        return nn.Sequential(OrderedDict(zip(names, modules)))
    if isinstance(layers, nn.Sequential):
        return layers
    names = ["input"] + [f"hidden{i}" for i in range(1, len(layers) - 1)] + ["output"]
    return nn.Sequential(OrderedDict(zip(names, layers)))


def _reduce_error(error: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return 0.5 * error.pow(2).mean()
    if reduction == "sum":
        return 0.5 * error.pow(2).sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


class PCModel(nn.Module):
    """Predictive-coding model with free-energy inference via autograd."""

    def __init__(
        self,
        layers: Union[List[nn.Module], Dict[str, nn.Module], nn.Sequential],
        *,
        batch_size: int = 64,
        energy_reduction: str = "mean",
        prior: Optional[Tensor] = None,
        prior_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.energy_reduction = energy_reduction
        self.prior = prior
        self.prior_weight = prior_weight

        layers = _as_sequential(layers)
        if len(layers) < 2:
            raise ValueError("The model must have at least 2 layers (input/output).")
        if not isinstance(layers[0], InputLayer):
            raise TypeError("The first layer must be an InputLayer.")
        self.layers = layers

    def clamp(self, input_data: Optional[Tensor] = None, output_data: Optional[Tensor] = None) -> None:
        """Clamp input/output states to fixed values."""
        if input_data is not None:
            self.layers[0].clamp(input_data)
        if output_data is not None:
            self.layers[-1].clamp(output_data)

    def release_clamp(self) -> None:
        """Release clamps on the input and output layers."""
        self.layers[0].release_clamp()
        self.layers[-1].release_clamp()

    def reset(self, batch_size: Optional[int] = None, init_scale: Optional[float] = None) -> None:
        """Reset the states of all layers."""
        if batch_size is not None:
            self.batch_size = batch_size
        for layer in self.layers:
            layer.reset(batch_size=batch_size, init_scale=init_scale)

    def detach_states(self) -> None:
        """Detach all layer states to drop autograd history."""
        for layer in self.layers:
            layer.detach_state()

    def _layer_names(self) -> List[str]:
        return list(self.layers._modules.keys())

    def _resolve_step_sizes(self, step_size: StepSize) -> List[float]:
        if isinstance(step_size, dict):
            default = step_size.get("default")
            sizes = []
            for name in self._layer_names():
                if name in step_size:
                    sizes.append(step_size[name])
                elif default is not None:
                    sizes.append(default)
                else:
                    raise ValueError(f"Missing step_size for layer '{name}'.")
            return sizes
        if isinstance(step_size, (list, tuple)):
            if len(step_size) != len(self.layers):
                raise ValueError("step_size must match number of layers when provided as a list.")
            return list(step_size)
        return [step_size] * len(self.layers)

    def _build_state_optimizer(
        self,
        free_layers: List[Tuple[int, nn.Module]],
        step_sizes: List[float],
        optimizer_name: str,
        optimizer_kwargs: Optional[Dict[str, float]],
    ) -> torch.optim.Optimizer:
        param_groups = []
        for idx, layer in free_layers:
            param_groups.append({"params": [layer.state], "lr": step_sizes[idx]})
        optimizer_kwargs = optimizer_kwargs or {}
        if optimizer_name.lower() == "adam":
            return torch.optim.Adam(param_groups, **optimizer_kwargs)
        if optimizer_name.lower() == "sgd":
            return torch.optim.SGD(param_groups, **optimizer_kwargs)
        raise ValueError(f"Unsupported state optimizer: {optimizer_name}")

    def free_energy(self, reduction: Optional[str] = None) -> Tuple[Tensor, List[Tensor]]:
        """Compute the predictive-coding free energy.

        Returns
        -------
        total_energy : Tensor
            Total free energy (scalar).
        per_layer : list[Tensor]
            Per-layer energy terms (scalar each).
        """
        if reduction is None:
            reduction = self.energy_reduction

        per_layer = []
        for idx in range(1, len(self.layers)):
            upper = self.layers[idx]
            lower = self.layers[idx - 1]
            if not isinstance(upper, PredictiveLayer):
                raise TypeError(
                    f"Layer {idx} ({upper.__class__.__name__}) must implement predict_down."
                )
            error = upper.prediction_error(lower.state)
            per_layer.append(_reduce_error(error, reduction))

        if self.prior is not None and self.prior_weight > 0:
            prior_error = self.layers[-1].state - self.prior
            per_layer.append(self.prior_weight * _reduce_error(prior_error, reduction))

        total = sum(per_layer) if per_layer else torch.tensor(0.0, device=self.device)
        return total, per_layer

    def infer(
        self,
        *,
        config: Optional[PCInferenceConfig] = None,
        n_steps: Optional[int] = None,
        step_size: Optional[StepSize] = None,
        energy_reduction: Optional[str] = None,
    ) -> Tensor:
        """Run gradient descent on free energy to infer states."""
        until_converged = False
        convergence_tol = None
        max_steps = None
        min_steps = 1
        state_optimizer = None
        state_optimizer_kwargs = None
        if config is not None:
            if n_steps is None:
                n_steps = config.n_steps
            if step_size is None:
                step_size = config.step_size
            if energy_reduction is None:
                energy_reduction = config.energy_reduction
            until_converged = config.until_converged
            convergence_tol = config.convergence_tol
            max_steps = config.max_steps
            min_steps = config.min_steps
            state_optimizer = config.state_optimizer
            state_optimizer_kwargs = config.state_optimizer_kwargs
        if n_steps is None:
            n_steps = 1
        if step_size is None:
            step_size = 0.1
        if min_steps < 1:
            raise ValueError("min_steps must be >= 1.")

        step_sizes = self._resolve_step_sizes(step_size)
        energy, _ = self.free_energy(reduction=energy_reduction)

        if until_converged:
            if max_steps is None:
                max_steps = n_steps
            if max_steps <= 0:
                raise ValueError("max_steps must be >= 1 when until_converged is True.")
            if convergence_tol is None:
                raise ValueError("convergence_tol must be set when until_converged is True.")
            total_steps = max_steps
        else:
            total_steps = n_steps
            if total_steps <= 0:
                return energy

        optimizer = None
        if state_optimizer is not None:
            free_layers = [
                (idx, layer) for idx, layer in enumerate(self.layers) if not layer.clamped
            ]
            if free_layers:
                for _, layer in free_layers:
                    layer.state = layer.state.detach().clone()
                optimizer = self._build_state_optimizer(
                    free_layers=free_layers,
                    step_sizes=step_sizes,
                    optimizer_name=state_optimizer,
                    optimizer_kwargs=state_optimizer_kwargs,
                )

        for step_idx in range(total_steps):
            free_layers = [
                (idx, layer) for idx, layer in enumerate(self.layers) if not layer.clamped
            ]
            if not free_layers:
                break

            free_states = []
            prev_states = []
            for _, layer in free_layers:
                layer.state.detach_()
                layer.state.requires_grad_(True)
                free_states.append(layer.state)
                if until_converged:
                    prev_states.append(layer.state.detach().clone())

            energy, _ = self.free_energy(reduction=energy_reduction)
            grads = torch.autograd.grad(energy, free_states, create_graph=False)

            with torch.no_grad():
                if optimizer is None:
                    for (idx, layer), grad in zip(free_layers, grads):
                        layer.state.sub_(step_sizes[idx] * grad)
                else:
                    optimizer.zero_grad(set_to_none=True)
                    for state, grad in zip(free_states, grads):
                        state.grad = grad
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if until_converged:
                max_delta = 0.0
                for (_, layer), prev_state in zip(free_layers, prev_states):
                    delta = (layer.state - prev_state).abs().mean().item()
                    if delta > max_delta:
                        max_delta = delta
                if step_idx + 1 >= min_steps and max_delta < convergence_tol:
                    break

            for _, layer in free_layers:
                layer.state.detach_()

        return energy

    def forward(
        self,
        input_data: Optional[Tensor] = None,
        *,
        inference_config: Optional[PCInferenceConfig] = None,
        step: Optional[StepSize] = None,
        n_steps: Optional[int] = None,
        energy_reduction: Optional[str] = None,
    ) -> Tensor:
        """Infer states and return the output layer state."""
        if input_data is not None and not self.layers[0].clamped:
            self.layers[0].state = input_data.detach()
        self.infer(
            config=inference_config,
            n_steps=n_steps,
            step_size=step,
            energy_reduction=energy_reduction,
        )
        return self.layers[-1].state

    def backward(self) -> Tuple[Optional[Tensor], Tensor, List[Tensor]]:
        """Compute reconstructions and free energy (no parameter update)."""
        energy, per_layer = self.free_energy()
        reconstruction = None
        if len(self.layers) > 1 and isinstance(self.layers[1], PredictiveLayer):
            reconstruction = self.layers[1].predict_down()
        return reconstruction, energy, per_layer

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def input(self) -> Tensor:
        return self.layers[0].state

    @property
    def output(self) -> Tensor:
        return self.layers[-1].state


def trace(
    num_words: Optional[int] = None,
    max_word_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    noise: float = 0.0,
    state_dict: Optional[Dict[str, Tensor]] = None,
    hidden_nonlinearity: Optional[Callable[[Tensor], Tensor]] = torch.relu,
    output_nonlinearity: Optional[Callable[[Tensor], Tensor]] = None,
) -> PCModel:
    """Construct a TRACE-like predictive-coding model (autograd-based)."""
    if state_dict is not None:
        output_shape = state_dict.get("layers.output.state")
        if output_shape is not None:
            if num_words is None:
                num_words = output_shape.shape[1]
            if batch_size is None:
                batch_size = output_shape.shape[0]
    if num_words is None:
        raise ValueError("num_words is required when state_dict is not provided.")
    if batch_size is None:
        batch_size = 1

    model = PCModel(
        dict(
            input=InputLayer(n_units=7, batch_size=batch_size),
            phoneme_layer=MiddleLayer(
                n_in=7,
                n_units=128,
                batch_size=batch_size,
                nonlinearity=hidden_nonlinearity,
                noise_std=noise,
            ),
            word_layer=MiddleLayer(
                n_in=128,
                n_units=num_words,
                batch_size=batch_size,
                nonlinearity=hidden_nonlinearity,
                noise_std=noise,
            ),
            output=OutputLayer(
                n_in=num_words,
                n_units=num_words,
                batch_size=batch_size,
                nonlinearity=output_nonlinearity,
                noise_std=noise,
            ),
        )
    )

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model
