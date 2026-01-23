from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import TraceDataset
from models import trace


@dataclass
class TraceTrainConfig:
    num_words: int = 215
    test_split: float = 0.2
    epochs: int = 50
    batch_size: Optional[int] = None
    learning_rate: float = 1e-3
    device: Optional[str] = None
    noise: float = 0.05
    steps_per_phoneme: int = 5
    step: float = 0.05
    step_optimizer_per_phoneme: bool = False
    early_stop_on_train_acc: bool = False
    seed: Optional[int] = None
    num_workers: int = 0
    pin_memory: Optional[bool] = None
    drop_last: bool = True
    use_tqdm: bool = True
    use_weight_norm: bool = True


@dataclass
class TraceTrainResult:
    model: torch.nn.Module
    history: Dict[str, List[float]]
    train_dataloader: DataLoader
    val_dataloader: DataLoader


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_trace_dataloaders(
    config: TraceTrainConfig,
) -> Tuple[TraceDataset, DataLoader, DataLoader]:
    trace_dataset = TraceDataset(config.num_words)
    train_size = int((1 - config.test_split) * len(trace_dataset))
    test_size = len(trace_dataset) - train_size
    generator = None
    if config.seed is not None:
        generator = torch.Generator().manual_seed(config.seed)
    train_dataset, test_dataset = torch.utils.data.random_split(
        trace_dataset, [train_size, test_size], generator=generator
    )
    if test_size == 0:
        batch_size = len(trace_dataset)
    else:
        desired_batch_size = (
            config.batch_size if config.batch_size is not None else test_size
        )
        batch_size = max(1, min(desired_batch_size, test_size))
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
    return trace_dataset, train_dataloader, test_dataloader


def train_trace_model(
    config: TraceTrainConfig,
    model: Optional[torch.nn.Module] = None,
    dataloaders: Optional[Tuple[DataLoader, DataLoader]] = None,
    dataset: Optional[TraceDataset] = None,
) -> TraceTrainResult:
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
            use_weight_norm=config.use_weight_norm,
        )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: Dict[str, List[float]] = {
        "loss": [],
        "loss_fw": [],
        "loss_bw": [],
        "grad_norm": [],
        "acc": [],
        "val_acc": [0.0],
    }
    total_steps = config.epochs * len(train_dataloader)
    pbar = tqdm(total=total_steps, desc="Training") if config.use_tqdm else None

    stop_training = False
    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_dataloader:
            words, features, labels_ind = batch.values()
            labels = F.one_hot(labels_ind, num_classes=dataset.num_words).float()
            features = features.to(device)
            labels_ind = labels_ind.to(device)
            labels = labels.to(device)

            model.reset()
            optimizer.zero_grad()

            seq_len = features.shape[1]
            loss_fw = torch.tensor(0.0, device=device)
            loss_bw = torch.tensor(0.0, device=device)
            grad_norm = 0.0
            for i in range(seq_len):
                for _ in range(config.steps_per_phoneme):
                    model.clamp(input_data=features[:, i, :])
                    _, loss_bw, _ = model.backward()
                    out = model.forward(features[:, i, :], step=config.step)
                    loss_fw = F.cross_entropy(out, labels)
                    loss = loss_fw + loss_bw
                    if config.step_optimizer_per_phoneme:
                        loss.backward()
                        grads_fw = [
                            torch.linalg.norm(p.grad).item()
                            for p in model.parameters()
                            if p.grad is not None
                        ]
                        grad_norm = (
                            sum(grads_fw) / len(grads_fw) if grads_fw else 0.0
                        )
                        optimizer.step()
                        optimizer.zero_grad()
                        model.detach_states()

            if config.step_optimizer_per_phoneme:
                loss = loss_fw + loss_bw
            else:
                _, loss_bw, _ = model.backward()
                out = model.forward(features[:, -1, :], step=config.step)
                loss_fw = F.cross_entropy(out, labels)
                loss = loss_fw + loss_bw
                loss.backward()
                optimizer.step()

            history["loss"].append(loss.item())
            history["loss_fw"].append(loss_fw.item())
            history["loss_bw"].append(loss_bw.item())
            if not config.step_optimizer_per_phoneme:
                grads_fw = [
                    torch.linalg.norm(p.grad).item()
                    for p in model.parameters()
                    if p.grad is not None
                ]
                grad_norm = sum(grads_fw) / len(grads_fw) if grads_fw else 0.0
            history["grad_norm"].append(grad_norm)
            correct = model.layers.output.state.argmax(dim=1).eq(labels_ind).sum().item()
            accuracy = 100.0 * correct / len(labels_ind)
            history["acc"].append(accuracy)

            if pbar is not None:
                pbar.set_description(
                    f"Epoch {epoch}/{config.epochs} "
                    f"Loss: {loss.item():.4f} Acc: {accuracy:.2f}, "
                    f"Val {history['val_acc'][-1]:.2f}%"
                )
                pbar.update(1)

            if config.early_stop_on_train_acc and accuracy >= 95.0:
                stop_training = True
                break

        if stop_training:
            break

        val_acc, _, _ = evaluate_trace_model(model, val_dataloader, config)
        history["val_acc"].append(val_acc)

    if pbar is not None:
        pbar.close()

    return TraceTrainResult(
        model=model,
        history=history,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )


def train_trace_model_legacy(
    model=None,
    test_split=0.2,
    epochs=50,
    batch_size=None,
    learning_rate=1e-3,
    device="cuda",
    noise=0.05,
    steps_per_phoneme=5,
    step=0.05,
    use_weight_norm=True,
):
    config = TraceTrainConfig(
        test_split=test_split,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        noise=noise,
        steps_per_phoneme=steps_per_phoneme,
        step=step,
        use_weight_norm=use_weight_norm,
    )
    return train_trace_model(config=config, model=model)


def evaluate_trace_model(
    model: torch.nn.Module, test_dataloader: DataLoader, config: TraceTrainConfig
):
    device = _resolve_device(config.device)
    model.eval()
    accuracies = []
    preds = []
    original_words = []
    with torch.no_grad():
        for batch in test_dataloader:
            words, features, labels_ind = batch.values()
            labels = F.one_hot(
                labels_ind, num_classes=config.num_words
            ).float().to(device)
            features = features.to(device)
            labels_ind = labels_ind.to(device)

            model.reset()
            seq_len = features.shape[1]
            for i in range(seq_len):
                model.clamp(input_data=features[:, i, :])
                for _ in range(config.steps_per_phoneme):
                    _ = model.backward()
                    _ = model.forward(features[:, i, :], step=config.step)
            preds.append(model.layers.output.state.argmax(dim=1))
            correct = preds[-1].eq(labels_ind).sum().item()
            accuracy = 100.0 * correct / len(labels_ind)
            accuracies.append(accuracy)
            original_words.extend(words)
    avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
    preds_tensor = torch.cat(preds, dim=0) if preds else torch.empty(0, dtype=torch.long)
    return avg_acc, preds_tensor, original_words


def evaluate(
    model,
    test_dataloader,
    steps_per_phoneme,
    device="cuda",
    num_classes=215,
    step=0.05,
):
    config = TraceTrainConfig(
        num_words=num_classes,
        steps_per_phoneme=steps_per_phoneme,
        device=device,
        step=step,
    )
    return evaluate_trace_model(model, test_dataloader, config)
