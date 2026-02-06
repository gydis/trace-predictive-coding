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


def _as_sequential(
    layers: Union[List[nn.Module], Dict[str, nn.Module], nn.Sequential]
) -> nn.Sequential:
    if isinstance(layers, (dict, OrderedDict)):
        return nn.Sequential(OrderedDict(layers))
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


def _precision_energy(error: Tensor, precision: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return 0.5 * precision * error.pow(2).mean() - 0.5 * torch.log(precision)
    if reduction == "sum":
        return 0.5 * precision * error.pow(2).sum() - 0.5 * error.numel() * torch.log(precision)
    raise ValueError(f"Unsupported reduction: {reduction}")


def _resolve_inference_args(
    config: Optional[PCInferenceConfig],
    n_steps: Optional[int],
    step_size: Optional[StepSize],
    energy_reduction: Optional[str],
) -> Dict[str, object]:
    cfg = config or PCInferenceConfig()
    return {
        "n_steps": cfg.n_steps if n_steps is None else n_steps,
        "step_size": cfg.step_size if step_size is None else step_size,
        "energy_reduction": cfg.energy_reduction if energy_reduction is None else energy_reduction,
        "until_converged": cfg.until_converged,
        "convergence_tol": cfg.convergence_tol,
        "max_steps": cfg.max_steps,
        "min_steps": cfg.min_steps,
        "state_optimizer": cfg.state_optimizer,
        "state_optimizer_kwargs": cfg.state_optimizer_kwargs,
    }


class PCModel(nn.Module):
    """Predictive-coding model with free-energy inference via autograd."""

    def __init__(
        self,
        layers: Union[List[nn.Module], Dict[str, nn.Module], nn.Sequential],
        *,
        batch_size: int = 64,
        energy_reduction: str = "mean",
        init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.energy_reduction = energy_reduction

        layers = _as_sequential(layers)
        if len(layers) < 2:
            raise ValueError("The model must have at least 2 layers (input/output).")
        if not isinstance(layers[0], InputLayer):
            raise TypeError("The first layer must be an InputLayer.")
        self.layers = layers

    def weight_parameters(self) -> List[Tensor]:
        """Return a list of all weight parameters in the model."""
        params = []
        for layer in self.layers:
            if hasattr(layer, "weight") and isinstance(layer.weight, nn.Parameter):
                params.append(layer.weight)
        return params

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

    def detach_states(self, *, clone: bool = False) -> None:
        """Detach all layer states to drop autograd history."""
        for layer in self.layers:
            if clone:
                layer.state = layer.state.detach().clone()
            else:
                layer.detach_state()

    def _layer_names(self) -> List[str]:
        return list(self.layers._modules.keys())

    def _free_layers(self) -> List[Tuple[int, nn.Module]]:
        return [(idx, layer) for idx, layer in enumerate(self.layers) if not layer.clamped]

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
        param_groups = [
            {"params": [layer.state], "lr": step_sizes[idx]} for idx, layer in free_layers
        ]
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

        per_layer: List[Tensor] = []
        for idx, upper in enumerate(self.layers[1:], start=1):
            lower = self.layers[idx - 1]
            if not isinstance(upper, PredictiveLayer):
                raise TypeError(
                    f"Layer {idx} ({upper.__class__.__name__}) must implement predict_down."
                )
            lower_value = lower.state_value() if hasattr(lower, "state_value") else lower.state
            error = upper.prediction_error(lower_value)
            if hasattr(upper, "precision") and callable(getattr(upper, "precision")):
                precision = upper.precision()
                per_layer.append(_precision_energy(error, precision, reduction))
            else:
                per_layer.append(_reduce_error(error, reduction))

        total = sum(per_layer) if per_layer else self.layers[0].state.new_tensor(0.0)
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
        args = _resolve_inference_args(config, n_steps, step_size, energy_reduction)
        n_steps = args["n_steps"]
        step_size = args["step_size"]
        energy_reduction = args["energy_reduction"]
        until_converged = bool(args["until_converged"])
        convergence_tol = args["convergence_tol"]
        max_steps = args["max_steps"]
        min_steps = args["min_steps"]
        state_optimizer = args["state_optimizer"]
        state_optimizer_kwargs = args["state_optimizer_kwargs"]

        if min_steps < 1:
            raise ValueError("min_steps must be >= 1.")

        step_sizes = self._resolve_step_sizes(step_size)

        if until_converged:
            total_steps = max_steps if max_steps is not None else n_steps
            if total_steps <= 0:
                raise ValueError("max_steps must be >= 1 when until_converged is True.")
            if convergence_tol is None:
                raise ValueError("convergence_tol must be set when until_converged is True.")
        else:
            total_steps = n_steps
            if total_steps <= 0:
                energy, _ = self.free_energy(reduction=energy_reduction)
                return energy

        optimizer = None
        if state_optimizer is not None:
            free_layers = self._free_layers()
            if free_layers:
                for _, layer in free_layers:
                    layer.state = layer.state.detach().clone()
                optimizer = self._build_state_optimizer(
                    free_layers=free_layers,
                    step_sizes=step_sizes,
                    optimizer_name=state_optimizer,
                    optimizer_kwargs=state_optimizer_kwargs,
                )

        energy = None
        for step_idx in range(total_steps):
            free_layers = self._free_layers()
            if not free_layers:
                break

            free_states: List[Tensor] = []
            prev_states: List[Tensor] = []
            for _, layer in free_layers:
                layer.state = layer.state.detach()
                layer.state.requires_grad_(True)
                free_states.append(layer.state)
                if until_converged:
                    prev_states.append(layer.state.detach().clone())

            energy, _ = self.free_energy(reduction=energy_reduction)
            grads = torch.autograd.grad(energy, free_states)

            with torch.no_grad():
                if optimizer is None:
                    for (idx, layer), grad in zip(free_layers, grads):
                        layer.state = layer.state - step_sizes[idx] * grad
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
                layer.state = layer.state.detach()

        if energy is None:
            energy, _ = self.free_energy(reduction=energy_reduction)
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
    
    def normalize_weights(self) -> None:
        """Normalize weights of all predictive layers to have fixed norm."""
        for layer in self.layers:
            if hasattr(layer, "weight") and isinstance(layer.weight, nn.Parameter):
                layer.normalize_weights()

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
    mean_approx: bool = False,
    phoneme_units: int = 48,
    state_dict: Optional[Dict[str, Tensor]] = None,
    hidden_nonlinearity: Optional[Callable[[Tensor], Tensor]] = torch.relu,
    output_nonlinearity: Optional[Callable[[Tensor], Tensor]] = None,
    learn_precision: bool = True,
    precision_init: float = 1.0,
    precision_eps: float = 1e-6,
    include_output_layer: bool = True,
    weight_norm: Optional[float] = None,
    init_scale: float = 0.0,
) -> PCModel:
    """Construct a TRACE-like predictive-coding model (autograd-based)."""
    if state_dict is not None:
        # Backward compat: older checkpoints may not have an explicit output layer.
        output_state = state_dict.get("layers.output.state") or state_dict.get("layers.word_layer.state")
        if output_state is not None:
            if num_words is None:
                num_words = output_state.shape[1]
            if batch_size is None:
                batch_size = output_state.shape[0]
    if num_words is None:
        raise ValueError("num_words is required when state_dict is not provided.")
    if batch_size is None:
        batch_size = 1
    if mean_approx and max_word_length is None:
        raise ValueError("max_word_length is required when mean_approx is True.")
    if weight_norm is not None and weight_norm <= 0:
        raise ValueError("weight_norm must be positive when weight normalization is enabled.")

    input_units = 7 * int(max_word_length) if mean_approx else 7

    word_layer_kwargs = dict(
        n_in=phoneme_units,
        n_units=num_words,
        batch_size=batch_size,
        nonlinearity=output_nonlinearity,
        noise_std=noise,
        learn_precision=learn_precision,
        precision_init=precision_init,
        precision_eps=precision_eps,
        weight_norm=weight_norm
    )
    word_layer = MiddleLayer(**word_layer_kwargs)

    layers: Dict[str, nn.Module] = dict(
        input=InputLayer(n_units=input_units, batch_size=batch_size),
        phoneme_layer=MiddleLayer(
            n_in=input_units,
            n_units=phoneme_units,
            batch_size=batch_size,
            nonlinearity=hidden_nonlinearity,
            noise_std=noise,
            learn_precision=learn_precision,
            precision_init=precision_init,
            precision_eps=precision_eps,
            weight_norm=weight_norm
        ),
        word_layer=word_layer,
    )
    if include_output_layer:
        layers["output"] = OutputLayer(
            n_in=num_words,
            n_units=num_words,
            batch_size=batch_size,
            nonlinearity=output_nonlinearity,
            noise_std=noise,
            learn_precision=learn_precision,
            precision_init=precision_init,
            precision_eps=precision_eps,
        )

    model = PCModel(
        layers,
        init_scale=init_scale,
    )

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model
