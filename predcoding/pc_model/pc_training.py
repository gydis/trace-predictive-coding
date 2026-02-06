from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import TraceDataset
from pc_models import PCInferenceConfig, PCModel, trace


@dataclass
class TracePCTrainConfig:
    num_words: int = 215
    test_split: float = 0.2
    epochs: int = 50
    batch_size: Optional[int] = None
    learning_rate: float = 1e-3
    device: Optional[str] = None
    noise: float = 0.05
    steps_per_phoneme: int = 5
    inference: PCInferenceConfig = field(default_factory=PCInferenceConfig)
    clamp_output_at_last_step: bool = True
    output_inference_steps: Optional[int] = None
    hidden_nonlinearity: Optional[Callable[[torch.Tensor], torch.Tensor]] = torch.nn.ReLU()
    output_nonlinearity: Optional[Callable[[torch.Tensor], torch.Tensor]] = None 
    mean_approx: bool = False
    phoneme_units: int = 48
    include_output_layer: bool = True
    learn_precision: bool = True
    precision_init: float = 1.0
    precision_eps: float = 1e-6
    output_state_mode: str = "softmax"
    output_state_eps: float = 1e-6
    output_entropy_weight: float = 0.0
    output_balance_weight: float = 0.0
    output_prior_eps: float = 1e-8
    train_weight_every_phoneme: bool = False
    early_stop_on_train_acc: bool = False
    seed: Optional[int] = None
    num_workers: int = 0
    pin_memory: Optional[bool] = None
    drop_last: bool = True
    use_tqdm: bool = True
    weight_norm: Optional[float] = None
    init_scale: float = 0.0
    label_force_every_phoneme: bool = False
    train_acc_free_run: bool = True
    use_supervised_energy_only: bool = True


@dataclass
class TracePCTrainResult:
    model: PCModel
    history: Dict[str, List[float]]
    train_dataloader: DataLoader
    val_dataloader: DataLoader


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_trace_dataloaders(
    config: TracePCTrainConfig,
) -> Tuple[TraceDataset, DataLoader, DataLoader]:
    dataset = TraceDataset(config.num_words, mean_approx=config.mean_approx)
    train_size = int((1 - config.test_split) * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(config.seed) if config.seed is not None else None
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=generator
    )

    if test_size == 0:
        batch_size = len(dataset)
    else:
        desired = config.batch_size if config.batch_size is not None else test_size
        batch_size = max(1, min(desired, test_size))

    pin_memory = (
        _resolve_device(config.device) == "cuda"
        if config.pin_memory is None
        else config.pin_memory
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=config.drop_last,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=config.drop_last,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    return dataset, train_dataloader, test_dataloader


def _resolve_inference_steps(config: TracePCTrainConfig) -> int:
    return config.steps_per_phoneme if config.steps_per_phoneme is not None else config.inference.n_steps


def _collect_precisions(model: PCModel) -> List[float]:
    precisions: List[float] = []
    for layer in model.layers:
        if hasattr(layer, "precision"):
            precision = layer.precision()
            precisions.append(float(precision.detach().mean().item()))
    return precisions


def _infer_free_run_output(
    model: PCModel,
    *,
    features: torch.Tensor,
    inference: PCInferenceConfig,
    steps_per_phoneme: int,
    init_scale: float,
) -> torch.Tensor:
    """Run an unclamped forward dynamics pass and return output states."""
    model.reset(init_scale=init_scale)
    model.release_clamp()
    seq_len = features.shape[1]
    for step_idx in range(seq_len):
        model.layers[0].clamp(features[:, step_idx, :])
        model.layers[-1].release_clamp()
        model.infer(config=inference, n_steps=steps_per_phoneme)
    return model.output.detach()


def train_trace_model(
    config: TracePCTrainConfig,
    model: Optional[PCModel] = None,
    dataloaders: Optional[Tuple[DataLoader, DataLoader]] = None,
    dataset: Optional[TraceDataset] = None,
) -> TracePCTrainResult:
    if dataloaders is None:
        dataset, train_dataloader, val_dataloader = build_trace_dataloaders(config)
    else:
        train_dataloader, val_dataloader = dataloaders
        if dataset is None:
            raise ValueError("dataset is required when providing dataloaders.")

    device = _resolve_device(config.device)
    if model is None:
        model = trace(
            num_words=config.num_words,
            max_word_length=dataset.max_word_length,
            batch_size=train_dataloader.batch_size,
            noise=config.noise,
            mean_approx=config.mean_approx,
            phoneme_units=config.phoneme_units,
            hidden_nonlinearity=config.hidden_nonlinearity,
            output_nonlinearity=config.output_nonlinearity,
            learn_precision=config.learn_precision,
            precision_init=config.precision_init,
            precision_eps=config.precision_eps,
            output_state_mode=config.output_state_mode,
            output_state_eps=config.output_state_eps,
            output_entropy_weight=config.output_entropy_weight,
            output_balance_weight=config.output_balance_weight,
            output_prior_eps=config.output_prior_eps,
            include_output_layer=config.include_output_layer,
            weight_norm=config.weight_norm,
        )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: Dict[str, List[float]] = {
        "free_energy": [],
        "grad_norm": [],
        "acc": [],
        "val_acc": [0.0],
        "precision": [],
        "weight_norm_pre_inf": [],
        "weight_norm_post_inf": [],
    }

    total_steps = config.epochs * len(train_dataloader)
    pbar = tqdm(total=total_steps, desc="Training") if config.use_tqdm else None

    steps_per_phoneme = _resolve_inference_steps(config)
    output_inference_steps = config.output_inference_steps or steps_per_phoneme
    energy_reduction = config.inference.energy_reduction
    train_weight_every_phoneme = config.train_weight_every_phoneme

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_dataloader:
            features = batch["features"].to(device)
            labels_ind = batch["index"].to(device)
            labels = F.one_hot(labels_ind, num_classes=dataset.num_words).float()

            model.reset(init_scale=config.init_scale)
            model.release_clamp()
            optimizer.zero_grad()

            seq_len = features.shape[1]
            total_energy = torch.tensor(0.0, device=device)
            used_energy_terms = 0

            for step_idx in range(seq_len):
                if config.label_force_every_phoneme:
                    model.layers[-1].clamp(labels)

                model.layers[0].clamp(features[:, step_idx, :])
                history["weight_norm_pre_inf"].append(
                    sum(p.norm().item() for p in model.weight_parameters())
                )
                model.infer(config=config.inference, n_steps=steps_per_phoneme)
                history["weight_norm_post_inf"].append(
                    sum(p.norm().item() for p in model.weight_parameters())
                )

                if step_idx == seq_len - 1:
                    if config.clamp_output_at_last_step:
                        model.layers[-1].clamp(labels)
                        model.infer(config=config.inference, n_steps=output_inference_steps)

                supervised_step = model.layers[-1].clamped
                if train_weight_every_phoneme:
                    if config.use_supervised_energy_only and not supervised_step:
                        continue
                    optimizer.zero_grad()
                    energy, _ = model.free_energy(reduction=energy_reduction)
                    energy.backward()
                    optimizer.step()
                    used_energy_terms += 1
                    history["free_energy"].append(energy.item())
                    history["precision"].append(_collect_precisions(model))
                    grads = [
                        torch.linalg.norm(p.grad).item()
                        for p in model.parameters()
                        if p.grad is not None
                    ]
                    grad_norm = sum(grads) / len(grads) if grads else 0.0
                    history["grad_norm"].append(grad_norm)
                    model.normalize_weights()
                else:
                    energy, _ = model.free_energy(reduction=energy_reduction)
                    if config.use_supervised_energy_only and not supervised_step:
                        continue
                    total_energy = total_energy + energy
                    used_energy_terms += 1

            if not train_weight_every_phoneme:
                if used_energy_terms > 0:
                    total_energy.backward()
                    optimizer.step()
                    model.normalize_weights()

                    history["free_energy"].append(total_energy.item())
                    history["precision"].append(_collect_precisions(model))
                    grads = [
                        torch.linalg.norm(p.grad).item()
                        for p in model.parameters()
                        if p.grad is not None
                    ]
                    grad_norm = sum(grads) / len(grads) if grads else 0.0
                    history["grad_norm"].append(grad_norm)
                else:
                    history["free_energy"].append(0.0)
                    history["precision"].append(_collect_precisions(model))
                    history["grad_norm"].append(0.0)

            if config.train_acc_free_run:
                output_pred = _infer_free_run_output(
                    model,
                    features=features,
                    inference=config.inference,
                    steps_per_phoneme=steps_per_phoneme,
                    init_scale=config.init_scale,
                )
            else:
                output_pred = model.output.detach()
            correct = output_pred.argmax(dim=1).eq(labels_ind).sum().item()
            accuracy = 100.0 * correct / len(labels_ind)
            history["acc"].append(accuracy)

            if pbar is not None:
                pbar.set_description(
                    f"Epoch {epoch}/{config.epochs} "
                    f"F: {history['free_energy'][-1]:.4f} Acc: {accuracy:.2f}, "
                    f"Val {history['val_acc'][-1]:.2f}%"
                )
                pbar.update(1)

            if config.early_stop_on_train_acc and accuracy >= 95.0:
                if pbar is not None:
                    pbar.close()
                return TracePCTrainResult(
                    model=model,
                    history=history,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                )

        val_acc, _, _ = evaluate_trace_model(model, val_dataloader, config)
        history["val_acc"].append(val_acc)

    if pbar is not None:
        pbar.close()

    return TracePCTrainResult(
        model=model,
        history=history,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )


def evaluate_trace_model(
    model: PCModel, test_dataloader: DataLoader, config: TracePCTrainConfig
):
    device = _resolve_device(config.device)
    model.eval()

    accuracies: List[float] = []
    preds: List[torch.Tensor] = []
    original_words: List[str] = []
    steps_per_phoneme = _resolve_inference_steps(config)

    for batch in test_dataloader:
        features = batch["features"].to(device)
        labels_ind = batch["index"].to(device)
        original_words.extend(batch["word"])

        pred = _infer_free_run_output(
            model,
            features=features,
            inference=config.inference,
            steps_per_phoneme=steps_per_phoneme,
            init_scale=config.init_scale,
        ).argmax(dim=1)
        preds.append(pred)
        correct = preds[-1].eq(labels_ind).sum().item()
        accuracies.append(100.0 * correct / len(labels_ind))

    avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
    preds_tensor = torch.cat(preds, dim=0) if preds else torch.empty(0, dtype=torch.long)
    return avg_acc, preds_tensor, original_words
