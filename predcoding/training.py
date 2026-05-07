from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import SpectrogramMNIST, TraceDataset, trace_collate_fn
from models import spectral_mnist, trace, trace_cnn
from settings import PHONEME_TO_INDEX

ZERO_PHONEME = "-"
ZERO_PHONEME_INDEX = PHONEME_TO_INDEX[ZERO_PHONEME]


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
    early_stop_threshold: float = 99.0
    state_dict_path: Optional[str] = None
    state_dict_strict: bool = True
    seed: Optional[int] = None
    num_workers: int = 4
    pin_memory: Optional[bool] = None
    drop_last: bool = True
    use_tqdm: bool = True
    use_weight_norm: bool = True
    use_sparse_weight_norm: bool = False
    phoneme_forcing: bool = False
    leakage: float = 0.0
    weight_decay: float = 0.0
    top_down: float = 0.0
    phoneme_memory_forcing: bool = False
    clamp_negatives: bool = False
    zero_steps: int = 0
    gradient_clipping: float = 0.0
    double_word_training: bool = False
    spectral_normalization: bool = False
    cnn: bool = False
    cnn_params: Optional[Dict] = None
    reset_model_each_batch: bool = False
    mask_padding: bool = True
    learn_padding: bool = False
    use_precision: bool = False
    nadam_optimizer: bool = False
    scheduler_min_lr: float = 1e-5
    scheduler: bool = False
    use_rnn: bool = False
    clamp_output_to_target: bool = False


@dataclass
class TraceTrainResult:
    model: torch.nn.Module
    history: Dict[str, List[float]]
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    config: TraceTrainConfig


@dataclass
class MnistTrainConfig:
    num_classes: int = 10
    test_split: float = 0.2
    epochs: int = 20
    batch_size: Optional[int] = 64
    learning_rate: float = 1e-3
    device: Optional[str] = None
    state_dict_path: Optional[str] = None
    state_dict_strict: bool = True
    seed: Optional[int] = None
    num_workers: int = 4
    pin_memory: Optional[bool] = None
    drop_last: bool = True
    use_tqdm: bool = True
    sample_rate: int = 16000
    n_fft: int = 256
    hop_length: int = 128
    time_dim: Optional[int] = 64
    inference_steps: int = 8
    step: float = 0.05
    zero_steps: int = 0
    reset_model_each_batch: bool = True
    noise: float = 0.0
    width: int = 128
    use_weight_norm: bool = True
    use_sparse_weight_norm: bool = False
    leakage: float = 0.0
    top_down: float = 0.0
    clamp_negatives: bool = False
    use_precision: bool = False
    use_rnn: bool = False
    cnn_params: Optional[Dict] = None
    weight_decay: float = 0.0
    gradient_clipping: float = 0.0
    loss_fw_weight: float = 1.0
    loss_bw_weight: float = 1.0
    nadam_optimizer: bool = False
    scheduler_min_lr: float = 1e-5
    scheduler: bool = False
    early_stop_on_train_acc: bool = False
    early_stop_threshold: float = 99.0


@dataclass
class MnistTrainResult:
    model: torch.nn.Module
    history: Dict[str, List[float]]
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    config: MnistTrainConfig


# ---------------------------------------------------------------------------
# Device / dataloader / model helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_trace_dataloaders(
    config: TraceTrainConfig,
    shuffle: bool = True,
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
    desired_batch_size = config.batch_size if config.batch_size is not None else max(1, train_size)
    if desired_batch_size is None or desired_batch_size <= 0:
        raise ValueError(
            f"TraceTrainConfig.batch_size must be a positive integer or None, got {config.batch_size}."
        )
    train_batch_size = max(1, min(desired_batch_size, max(1, train_size)))
    val_batch_size = max(1, min(desired_batch_size, max(1, test_size)))
    pin_memory = (
        _resolve_device(config.device) == "cuda"
        if config.pin_memory is None
        else config.pin_memory
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=shuffle,
        drop_last=config.drop_last,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        collate_fn=trace_collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=config.drop_last,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        collate_fn=trace_collate_fn,
    )
    return trace_dataset, train_dataloader, test_dataloader


def build_mnist_dataloaders(
    config: MnistTrainConfig,
    shuffle: bool = True,
) -> Tuple[SpectrogramMNIST, DataLoader, DataLoader]:
    train_dataset = SpectrogramMNIST(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        time_dim=config.time_dim,
    )
    test_dataset = SpectrogramMNIST(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        time_dim=config.time_dim,
        test=True,
    )
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    desired_batch_size = config.batch_size if config.batch_size is not None else 64
    max_train_batch = max(1, train_size) if train_size > 0 else 1
    batch_size = max(1, min(desired_batch_size, max_train_batch))

    pin_memory = (
        _resolve_device(config.device) == "cuda"
        if config.pin_memory is None
        else config.pin_memory
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=config.drop_last,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        collate_fn=train_dataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=config.drop_last,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        collate_fn=test_dataset.collate_fn,
    )
    return train_dataset, train_dataloader, test_dataloader


def load_trace_model(
    *,
    state_dict_path: str,
    device: Optional[str] = None,
    strict: bool = True,
    config: TraceTrainConfig = None,
    num_words: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_word_length: Optional[int] = None,
) -> torch.nn.Module:
    """Load a TRACE model from a saved state dict."""
    resolved_device = _resolve_device(device)
    state_dict = torch.load(state_dict_path, map_location=resolved_device)
    if num_words is None:
        output_state = state_dict.get("layers.output.state")
        if output_state is not None:
            num_words = output_state.shape[1]
    if num_words is None:
        raise ValueError("num_words must be provided if it cannot be inferred.")
    if batch_size is None:
        output_state = state_dict.get("layers.output.state")
        if output_state is not None:
            batch_size = output_state.shape[0]
    if batch_size is None:
        batch_size = 1
    if max_word_length is None:
        phoneme_layer_state = state_dict.get("layers.phoneme_layer.state")
        if phoneme_layer_state is not None:
            max_word_length = phoneme_layer_state.shape[1]
    model = trace(
        num_words=config.num_words,
        max_word_length=max_word_length,
        batch_size=batch_size,
        noise=config.noise,
        use_weight_norm=config.use_weight_norm,
        use_sparse_weight_norm=config.use_sparse_weight_norm,
        leakage=config.leakage,
        clamp_negatives=config.clamp_negatives,
    )
    model.load_state_dict(state_dict, strict=strict)
    return model.to(resolved_device)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _pad_features_for_sliding_window(features: torch.Tensor, conv_ph: int) -> torch.Tensor:
    pad = torch.zeros(features.shape[0], conv_ph - 1, features.shape[2], device=features.device)
    return torch.cat([pad, features], dim=1)

def _update_precisions(model) -> None:
    for layer in model.layers:
        if hasattr(layer, "update_precision"):
            layer.update_precision()


def _zero_input(features: torch.Tensor, cnn: bool, convolved_phonemes: int = 3) -> torch.Tensor:
    if cnn:
        # Shape: (B, n_features, 1, conv_ph) — matches InputLayer(n_units=(7, 1, conv_ph))
        return torch.zeros(
            features.shape[0], features.shape[2], 1, convolved_phonemes,
            device=features.device,
        )
    return torch.zeros_like(features[:, 0, :])


def _target_output_clamp(
    model: torch.nn.Module,
    labels_ind: torch.Tensor,
    enabled: bool,
) -> Optional[torch.Tensor]:
    if not enabled:
        return None
    num_classes = model.layers.output.state.shape[1]
    return F.one_hot(labels_ind, num_classes=num_classes).float()


def _run_zero_steps(
    model,
    zero_inp: torch.Tensor,
    config: TraceTrainConfig,
    output_target: Optional[torch.Tensor] = None,
) -> None:
    for _ in range(config.zero_steps):
        model.clamp(input_data=zero_inp, output_data=output_target)
        model.backward()
        model.forward(zero_inp, step=config.step)


def _phoneme_forcing_loss(
    model,
    word_padded: List[str],
    i: int,
    config: TraceTrainConfig,
    device: str,
    seq_len: int,
) -> torch.Tensor:
    has_phoneme_layer = hasattr(model.layers, "phoneme_layer")
    need_phoneme = (
        config.phoneme_forcing
        and has_phoneme_layer
        and model.layers.phoneme_layer.state.shape[1] == 15
    )
    need_memory = config.phoneme_memory_forcing
    if not (need_phoneme or need_memory):
        return torch.tensor(0.0, device=device)

    phoneme_indices = torch.tensor(
        [PHONEME_TO_INDEX[word_padded[j][i]] for j in range(len(word_padded))],
        dtype=torch.long,
        device=device,
    )
    loss = torch.tensor(0.0, device=device)
    if need_phoneme:
        loss = loss + F.cross_entropy(
            model.layers.phoneme_layer.state, phoneme_indices
        ) / (config.steps_per_phoneme * seq_len)
    if (
        need_memory
        and model.layers.memory_layer is not None
        and model.layers.memory_layer.state.shape[1] == 15
    ):
        loss = loss + F.cross_entropy(
            model.layers.memory_layer.state, phoneme_indices
        ) / (config.steps_per_phoneme * seq_len)
    return loss


def _padding_token_loss(
    model,
    zero_token_mask: torch.Tensor,
    config: TraceTrainConfig,
    device: str,
) -> torch.Tensor:
    if not config.learn_padding:
        return torch.tensor(0.0, device=device)
    if (
        zero_token_mask is None
        or zero_token_mask.numel() == 0
        or not torch.any(zero_token_mask)
    ):
        return torch.tensor(0.0, device=device)
    if not hasattr(model.layers, "phoneme_layer"):
        return torch.tensor(0.0, device=device)
    phoneme_logits = model.layers.phoneme_layer.state
    if phoneme_logits.ndim != 2 or phoneme_logits.shape[1] <= ZERO_PHONEME_INDEX:
        return torch.tensor(0.0, device=device)

    targets = torch.full(
        (int(zero_token_mask.sum().item()),),
        ZERO_PHONEME_INDEX,
        dtype=torch.long,
        device=device,
    )
    return F.cross_entropy(phoneme_logits[zero_token_mask], targets, reduction="sum")


def _permute_batch(
    features: torch.Tensor,
    labels_ind: torch.Tensor,
    lengths: torch.Tensor,
    word_padded: List[str],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    perm = torch.randperm(features.size(0))
    perm_list = perm.tolist()
    return (
        features[perm].to(device),
        labels_ind[perm].to(device),
        lengths[perm].to(device),
        [word_padded[j] for j in perm_list],
    )


def _unpack_trace_batch(
    batch,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[str], torch.Tensor]:
    if not isinstance(batch, dict):
        raise ValueError(
            "Expected TRACE batch as a dict with keys "
            "('word', 'features', 'index', 'word_padded', 'lengths')."
        )
    required = ("word", "features", "index", "word_padded")
    missing = [key for key in required if key not in batch]
    if missing:
        raise ValueError(f"TRACE batch is missing keys: {missing}")

    words = list(batch["word"])
    features = batch["features"]
    labels_ind = batch["index"]
    word_padded = list(batch["word_padded"])
    lengths = batch.get("lengths")
    if lengths is None:
        lengths = torch.tensor([len(w) for w in words], dtype=torch.long)
    return words, features, labels_ind, word_padded, lengths


def _compute_grad_norm(model) -> float:
    grads = [
        torch.linalg.norm(p.grad).item()
        for p in model.parameters()
        if p.grad is not None
    ]
    return sum(grads) / len(grads) if grads else 0.0


def _compute_per_layer_grad_norms(model) -> dict:
    return {
        f"layer_{i}": torch.linalg.norm(model.layers[i].weight.grad).item()
        for i in range(len(model.layers))
        if hasattr(model.layers[i], "weight")
        and model.layers[i].weight.grad is not None
    }


def _compute_per_layer_weight_norms(model) -> dict:
    return {
        f"layer_{i}": torch.linalg.norm(model.layers[i].weight).item()
        for i in range(len(model.layers))
        if hasattr(model.layers[i], "weight")
    }


# ---------------------------------------------------------------------------
# Default (end-of-batch) sequence runners
# ---------------------------------------------------------------------------

def _run_fc_sequence(
    model,
    features: torch.Tensor,
    labels_ind: torch.Tensor,
    lengths: torch.Tensor,
    word_padded: List[str],
    config: TraceTrainConfig,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """Process one full word sequence through the FC model.

    Runs ``steps_per_phoneme`` backward/forward inference cycles for each
    phoneme position, then one final cycle with zero input.

    Returns ``(loss_fw, loss_bw, loss_phoneme, final_out, final_losses_list)``.
    ``final_losses_list`` is the per-layer backward loss from the last cycle,
    used for spectral normalisation.  ``loss_fw`` accumulation is controlled by
    ``config.mask_padding``:

    - ``True``:  accumulate only at positions where a word ends (current default).
    - ``False``:  single cross-entropy on the zero-input final output.
    """
    seq_len = features.shape[1]
    loss_fw = torch.tensor(0.0, device=device)
    loss_bw = torch.tensor(0.0, device=device)
    loss_phoneme = torch.tensor(0.0, device=device)
    loss_padding_fw = torch.tensor(0.0, device=device)
    loss_bw_t = torch.tensor(0.0, device=device)
    final_losses: list = []
    acc_layer_losses: list = []
    output_target = _target_output_clamp(model, labels_ind, config.clamp_output_to_target)

    for i in range(seq_len):
        input_feat = features[:, i, :]
        model.clamp(input_data=input_feat, output_data=output_target)
        for _ in range(config.steps_per_phoneme):
            if config.mask_padding:
                mask = lengths == i + 1
            _, loss_bw_t, final_losses = model.backward(mask=mask if config.mask_padding else None)
            out = model.forward(input_feat, step=config.step)
            loss_phoneme = loss_phoneme + _phoneme_forcing_loss(
                model, word_padded, i, config, device, seq_len
            )
            zero_mask = torch.all(input_feat == 0, dim=1)
            loss_padding_fw = loss_padding_fw + _padding_token_loss(
                model, zero_mask, config, device
            )

        if config.mask_padding:
            mask = lengths == i + 1
            if mask.any():
                loss_fw = loss_fw + F.cross_entropy(
                    out[mask], labels_ind[mask], reduction="sum"
                )
                loss_bw = loss_bw + loss_bw_t
                if not acc_layer_losses:
                    acc_layer_losses = list(final_losses)
                else:
                    acc_layer_losses = [a + b for a, b in zip(acc_layer_losses, final_losses)]

    # Extra backward/forward with zeros — required for spectral normalisation
    # and matches the original post-loop step.
    zero_inp = _zero_input(features, cnn=False)
    _, loss_bw_t, final_losses = model.backward()
    out = model.forward(zero_inp, step=config.step)
    zero_mask = torch.ones(
        zero_inp.shape[0], dtype=torch.bool, device=zero_inp.device
    )
    loss_padding_fw = loss_padding_fw + _padding_token_loss(
        model, zero_mask, config, device
    )

    if not config.mask_padding:
        loss_fw = F.cross_entropy(out, labels_ind) + (loss_padding_fw / len(labels_ind))
        loss_bw = loss_bw_t
        acc_layer_losses = final_losses
    else:
        loss_fw = (loss_fw + loss_padding_fw) / len(labels_ind)
        loss_bw = loss_bw / len(labels_ind)

    return loss_fw, loss_bw, loss_phoneme, out, acc_layer_losses


def _run_cnn_sequence(
    model,
    features: torch.Tensor,
    labels_ind: torch.Tensor,
    lengths: torch.Tensor,
    word_padded: List[str],
    config: TraceTrainConfig,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """Process one full word sequence through the CNN model.

    Identical structure to ``_run_fc_sequence`` but iterates over phoneme
    positions in strides of ``convolved_phonemes`` and presents a windowed,
    flattened input slice to the model at each position.

    Returns ``(loss_fw, loss_bw, loss_phoneme, final_out, final_losses_list)``.
    """
    conv_ph: int = (config.cnn_params or {}).get("convolved_phonemes", 3)
    stride: int = (config.cnn_params or {}).get("stride", conv_ph)
    seq_len = features.shape[1]
    loss_fw = torch.tensor(0.0, device=device)
    loss_bw = torch.tensor(0.0, device=device)
    loss_phoneme = torch.tensor(0.0, device=device)
    loss_padding_fw = torch.tensor(0.0, device=device)
    loss_bw_t = torch.tensor(0.0, device=device)
    final_losses: list = []
    acc_layer_losses: list = []
    output_target = _target_output_clamp(model, labels_ind, config.clamp_output_to_target)

    padded = _pad_features_for_sliding_window(features, conv_ph)
    for orig_pos in range(0, seq_len, stride):
        window = padded[:, orig_pos:orig_pos + conv_ph, :]        # (B, conv_ph, 7)
        input_feat = window.permute(0, 2, 1).unsqueeze(2)         # (B, 7, 1, conv_ph)
        model.clamp(input_data=input_feat, output_data=output_target)
        for _ in range(config.steps_per_phoneme):
            if config.mask_padding:
                mask = lengths == orig_pos + 1
            _, loss_bw_t, final_losses = model.backward(mask=mask if config.mask_padding else None)
            out = model.forward(input_feat, step=config.step)
            loss_phoneme = loss_phoneme + _phoneme_forcing_loss(
                model, word_padded, orig_pos, config, device, seq_len
            )
            zero_mask = torch.all(input_feat.reshape(input_feat.shape[0], -1) == 0, dim=1)
            loss_padding_fw = loss_padding_fw + _padding_token_loss(
                model, zero_mask, config, device
            )

        if config.mask_padding:
            mask = lengths == orig_pos + 1
            if mask.any():
                loss_fw = loss_fw + F.cross_entropy(
                    out[mask], labels_ind[mask], reduction="sum"
                )
                loss_bw = loss_bw + loss_bw_t
                if not acc_layer_losses:
                    acc_layer_losses = list(final_losses)
                else:
                    acc_layer_losses = [a + b for a, b in zip(acc_layer_losses, final_losses)]

    zero_inp = _zero_input(features, cnn=True, convolved_phonemes=conv_ph)
    _, loss_bw_t, final_losses = model.backward()
    out = model.forward(zero_inp, step=config.step)
    zero_mask = torch.ones(
        zero_inp.shape[0], dtype=torch.bool, device=zero_inp.device
    )
    loss_padding_fw = loss_padding_fw + _padding_token_loss(
        model, zero_mask, config, device
    )

    # _update_precisions(model)

    if not config.mask_padding:
        loss_fw = F.cross_entropy(out, labels_ind) + (loss_padding_fw / len(labels_ind))
        loss_bw = loss_bw_t
        acc_layer_losses = final_losses
    else:
        loss_fw = (loss_fw + loss_padding_fw) / len(labels_ind)
        loss_bw = loss_bw / len(labels_ind)

    return loss_fw, loss_bw, loss_phoneme, out, acc_layer_losses


# ---------------------------------------------------------------------------
# Per-phoneme optimizer step sequence runners
# ---------------------------------------------------------------------------

def _run_fc_sequence_per_phoneme(
    model,
    features: torch.Tensor,
    labels_ind: torch.Tensor,
    lengths: torch.Tensor,
    word_padded: List[str],
    config: TraceTrainConfig,
    device: str,
    optimizer: torch.optim.Optimizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Process one full word sequence for the FC model, stepping the optimizer
    after every individual inference step.

    Returns ``(loss_fw, loss_bw, loss_phoneme, final_out, grad_norm)`` where
    the loss tensors reflect the last phoneme position and are suitable for
    logging only — weight updates have already been applied inside this function.
    """
    seq_len = features.shape[1]
    loss_fw = torch.tensor(0.0, device=device)
    loss_bw = torch.tensor(0.0, device=device)
    loss_phoneme = torch.tensor(0.0, device=device)
    loss_padding_fw = torch.tensor(0.0, device=device)
    out = None
    grad_norm = 0.0
    per_layer_grad_norms: dict = {}
    output_target = _target_output_clamp(model, labels_ind, config.clamp_output_to_target)

    for i in range(seq_len):
        input_feat = features[:, i, :]
        model.clamp(input_data=input_feat, output_data=output_target)
        for _ in range(config.steps_per_phoneme):
            _, loss_bw_t, _ = model.backward()
            out = model.forward(input_feat, step=config.step)
            ph_loss = _phoneme_forcing_loss(model, word_padded, i, config, device, seq_len)
            loss_phoneme = loss_phoneme + ph_loss
            zero_mask = torch.all(input_feat == 0, dim=1)
            padding_fw_t = _padding_token_loss(model, zero_mask, config, device)
            loss_padding_fw = loss_padding_fw + padding_fw_t

            step_loss = (
                loss_bw_t
                + ph_loss
                + F.cross_entropy(out, labels_ind)
                + (padding_fw_t / len(labels_ind))
            )
            step_loss.backward()
            if config.gradient_clipping != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
            optimizer.step()
            optimizer.zero_grad()
            model.detach_states()

        mask = lengths == i + 1
        if mask.any():
            loss_fw = loss_fw + F.cross_entropy(
                out[mask], labels_ind[mask], reduction="sum"
            )
            loss_bw = loss_bw + loss_bw_t

    loss_fw = loss_fw + loss_padding_fw
    grad_norm = _compute_grad_norm(model)
    per_layer_grad_norms = _compute_per_layer_grad_norms(model)
    return loss_fw, loss_bw, loss_phoneme, out, grad_norm, per_layer_grad_norms


def _run_cnn_sequence_per_phoneme(
    model,
    features: torch.Tensor,
    labels_ind: torch.Tensor,
    lengths: torch.Tensor,
    word_padded: List[str],
    config: TraceTrainConfig,
    device: str,
    optimizer: torch.optim.Optimizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Process one full word sequence for the CNN model, stepping the optimizer
    after every individual inference step.

    Returns ``(loss_fw, loss_bw, loss_phoneme, final_out, grad_norm)`` for
    logging only — weight updates have already been applied inside this function.
    """
    conv_ph: int = (config.cnn_params or {}).get("convolved_phonemes", 3)
    stride: int = (config.cnn_params or {}).get("stride", conv_ph)
    seq_len = features.shape[1]
    loss_fw = torch.tensor(0.0, device=device)
    loss_bw = torch.tensor(0.0, device=device)
    loss_phoneme = torch.tensor(0.0, device=device)
    loss_padding_fw = torch.tensor(0.0, device=device)
    out = None
    grad_norm = 0.0
    per_layer_grad_norms: dict = {}
    output_target = _target_output_clamp(model, labels_ind, config.clamp_output_to_target)

    padded = _pad_features_for_sliding_window(features, conv_ph)
    for orig_pos in range(0, seq_len, stride):
        window = padded[:, orig_pos:orig_pos + conv_ph, :]        # (B, conv_ph, 7)
        input_feat = window.permute(0, 2, 1).unsqueeze(2)         # (B, 7, 1, conv_ph)
        model.clamp(input_data=input_feat, output_data=output_target)
        for _ in range(config.steps_per_phoneme):
            _, loss_bw_t, _ = model.backward()
            out = model.forward(input_feat, step=config.step)
            ph_loss = _phoneme_forcing_loss(model, word_padded, orig_pos, config, device, seq_len)
            loss_phoneme = loss_phoneme + ph_loss
            zero_mask = torch.all(input_feat.reshape(input_feat.shape[0], -1) == 0, dim=1)
            padding_fw_t = _padding_token_loss(model, zero_mask, config, device)
            loss_padding_fw = loss_padding_fw + padding_fw_t

            step_loss = (
                loss_bw_t
                + ph_loss
                + F.cross_entropy(out, labels_ind)
                + (padding_fw_t / len(labels_ind))
            )
            step_loss.backward()
            if config.gradient_clipping != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
            optimizer.step()
            optimizer.zero_grad()
            model.detach_states()

        mask = lengths == orig_pos + 1
        if mask.any():
            loss_fw = loss_fw + F.cross_entropy(
                out[mask], labels_ind[mask], reduction="sum"
            )
            loss_bw = loss_bw + loss_bw_t

    loss_fw = loss_fw + loss_padding_fw
    grad_norm = _compute_grad_norm(model)
    per_layer_grad_norms = _compute_per_layer_grad_norms(model)
    return loss_fw, loss_bw, loss_phoneme, out, grad_norm, per_layer_grad_norms


# ---------------------------------------------------------------------------
# Batch training
# ---------------------------------------------------------------------------

def _train_batch(
    model,
    batch,
    config: TraceTrainConfig,
    device: str,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    **kwargs,
) -> Dict[str, float]:
    _, features, labels_ind, word_padded, lengths = _unpack_trace_batch(batch)
    features = features.to(device)
    labels_ind = labels_ind.to(device)
    lengths = lengths.to(device)

    if config.reset_model_each_batch or model.batch_size != features.shape[0]:
        model.reset(batch_size=features.shape[0])
    else:
        model.detach_states()
    optimizer.zero_grad()

    conv_ph: int = (config.cnn_params or {}).get("convolved_phonemes", 3)
    n_passes = 2 if config.double_word_training else 1

    # Accumulators across passes (end-of-batch mode only).
    acc_loss_fw = torch.tensor(0.0, device=device)
    acc_loss_bw = torch.tensor(0.0, device=device)
    acc_loss_phoneme = torch.tensor(0.0, device=device)
    final_out = None
    final_losses_list: list = []

    # Per-phoneme mode: keep only the last pass values for logging.
    last_loss_fw = torch.tensor(0.0, device=device)
    last_loss_bw = torch.tensor(0.0, device=device)
    grad_norm = 0.0
    per_layer_grad_norms: dict = {}
    accuracy = 0.0

    for w in range(n_passes):
        features, labels_ind, lengths, word_padded = _permute_batch(
            features, labels_ind, lengths, word_padded, device
        )
        zero_inp = _zero_input(features, config.cnn, conv_ph)
        output_target = _target_output_clamp(
            model, labels_ind, config.clamp_output_to_target
        )
        _run_zero_steps(model, zero_inp, config, output_target=output_target)

        if config.step_optimizer_per_phoneme:
            if config.cnn:
                loss_fw, loss_bw, loss_phoneme, out, grad_norm, per_layer_grad_norms = (
                    _run_cnn_sequence_per_phoneme(
                        model, features, labels_ind, lengths, word_padded,
                        config, device, optimizer,
                    )
                )
            else:
                loss_fw, loss_bw, loss_phoneme, out, grad_norm, per_layer_grad_norms = (
                    _run_fc_sequence_per_phoneme(
                        model, features, labels_ind, lengths, word_padded,
                        config, device, optimizer,
                    )
                )
            last_loss_fw = loss_fw
            last_loss_bw = loss_bw
        else:
            if config.cnn:
                loss_fw, loss_bw, loss_phoneme, out, final_losses_list = (
                    _run_cnn_sequence(
                        model, features, labels_ind, lengths, word_padded,
                        config, device,
                    )
                )
            else:
                loss_fw, loss_bw, loss_phoneme, out, final_losses_list = (
                    _run_fc_sequence(
                        model, features, labels_ind, lengths, word_padded,
                        config, device,
                    )
                )
            acc_loss_fw = acc_loss_fw + loss_fw
            acc_loss_bw = acc_loss_bw + loss_bw
            acc_loss_phoneme = acc_loss_phoneme + loss_phoneme
            final_out = out

        # accuracy += model.layers.output.state.argmax(dim=1).eq(labels_ind).sum().item()

    if config.step_optimizer_per_phoneme:
        loss = last_loss_fw + last_loss_bw
        layer_weight_norms = _compute_per_layer_weight_norms(model)
    else:
        if config.spectral_normalization:
            denoms = [layer.denom for layer in model.layers if hasattr(layer, "denom")]
            acc_loss_bw = torch.stack(
                [(l / d).mean() for l, d in zip(final_losses_list, denoms)]
            ).sum()
            y = F.one_hot(labels_ind, num_classes=final_out.shape[1]).float()
            delta = F.softmax(final_out, dim=-1) - y
            c_L = (model.layers.output.weight ** 2).sum(dim=1)
            acc_loss_fw = ((delta ** 2) / (c_L + 1e-6)).mean() + acc_loss_fw

        loss = acc_loss_fw + acc_loss_phoneme + acc_loss_bw
        loss.backward()
        if config.gradient_clipping != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
        grad_norm = _compute_grad_norm(model)
        per_layer_grad_norms = _compute_per_layer_grad_norms(model)
        optimizer.step()
        scheduler.step() if scheduler is not None else None
        layer_weight_norms = _compute_per_layer_weight_norms(model)

    # accuracy = (accuracy / (n_passes * labels_ind.shape[0])) * 100
    log_fw = last_loss_fw if config.step_optimizer_per_phoneme else acc_loss_fw
    log_bw = last_loss_bw if config.step_optimizer_per_phoneme else acc_loss_bw
    precisions = {f"layer_{i}": model.layers[i].pi.mean().item() for i in range(len(model.layers)) if hasattr(model.layers[i], "pi")}
    return {
        "loss": loss.item(),
        "loss_fw": log_fw.item(),
        "loss_bw": log_bw.item(),
        "grad_norm": grad_norm,
        "acc": accuracy,
        "precisions": precisions,
        "layer_backward_losses": layer_backward_losses,
        "layer_grad_norms": per_layer_grad_norms,
        "layer_weight_norms": layer_weight_norms,
    }


# ---------------------------------------------------------------------------
# Main training entrypoint
# ---------------------------------------------------------------------------

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
    state_dict = None
    if config.state_dict_path is not None:
        state_dict = torch.load(config.state_dict_path, map_location=device)

    if model is None:
        if state_dict is not None and config.num_words is None:
            output_state = state_dict.get("layers.output.state")
            if output_state is not None:
                config.num_words = output_state.shape[1]
        model_batch_size = train_dataloader.batch_size or config.batch_size or 1
        model_kwargs = dict(
            num_words=config.num_words,
            max_word_length=dataset.max_word_length,
            batch_size=model_batch_size,
            noise=config.noise,
            use_weight_norm=config.use_weight_norm,
            use_sparse_weight_norm=config.use_sparse_weight_norm,
            leakage=config.leakage,
            clamp_negatives=config.clamp_negatives,
            spectral_normalization=config.spectral_normalization,
            use_precision=config.use_precision,
            use_rnn=config.use_rnn,
        )
        model = trace_cnn(**model_kwargs, cnn_params=config.cnn_params) if config.cnn else trace(**model_kwargs)

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=config.state_dict_strict)
    model = model.to(device)

    if config.nadam_optimizer:
        optimizer = torch.optim.NAdam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        # optimizer = torch.optim.SGD(
        #     model.parameters(), lr=config.learning_rate)
    else: 
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.epochs * max(1, len(train_dataloader))),
        eta_min=config.scheduler_min_lr,
    ) if config.scheduler else None

    history: Dict[str, List] = {
        "loss": [], "loss_fw": [], "loss_bw": [], "grad_norm": [], "acc": [], "val_acc": [0.0], "precisions": [],
        "layer_backward_losses": [], "layer_grad_norms": [], "layer_weight_norms": [],
    }
    total_steps = config.epochs * len(train_dataloader)
    pbar = tqdm(total=total_steps, desc="Training") if config.use_tqdm else None

    try:
        acc = 0.0
        for epoch in range(1, config.epochs + 1):
            model.train()
            for batch in train_dataloader:
                metrics = _train_batch(model, batch, config, device, optimizer, scheduler=scheduler)
                if epoch % 5 == 0:
                    acc, _, _, _ = evaluate_trace_model(model, train_dataloader, config)
                for key in ("loss", "loss_fw", "loss_bw", "grad_norm", "acc", "precisions",
                            "layer_backward_losses", "layer_grad_norms", "layer_weight_norms"):
                    history[key].append(metrics[key])
                history["acc"][-1] = acc
                metrics["acc"] = acc
                if pbar is not None:
                    pbar.set_description(
                        f"Epoch {epoch}/{config.epochs} "
                        f"Loss: {metrics['loss']:.4f} Acc: {metrics['acc']:.2f}, "
                        f"Grad Norm: {metrics['grad_norm']:.4f}, "
                        f"FW Loss: {metrics['loss_fw']:.4f}, BW Loss: {metrics['loss_bw']:.4f}, "
                        f"Precisions: {[f'{l}: {p:.2f}' for l, p in metrics['precisions'].items()]}, "
                    )
                    pbar.update(1)

            if config.early_stop_on_train_acc and metrics["acc"] >= config.early_stop_threshold:
                break
    finally:
        if pbar is not None:
            pbar.close()

    return TraceTrainResult(
        model=model,
        history=history,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config
    )


# ---------------------------------------------------------------------------
# SpectralMNIST training
# ---------------------------------------------------------------------------

def _unpack_mnist_batch(batch):
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError(
            "Expected SpectrogramMNIST batch as (specs, labels) or (specs, labels, lengths)."
        )
    specs = batch[0]
    labels = batch[1]
    lengths = batch[2] if len(batch) > 2 else None
    return specs, labels, lengths


def _run_spectral_inference(
    model: torch.nn.Module,
    specs: torch.Tensor,
    config: MnistTrainConfig,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    bw_losses = []
    out = None
    final_losses = []
    for _ in range(config.inference_steps):
        model.clamp(input_data=specs)
        _, loss_bw_t, final_losses = model.backward()
        out = model.forward(specs, step=config.step)
        bw_losses.append(loss_bw_t if torch.is_tensor(loss_bw_t) else specs.new_tensor(loss_bw_t))
    if bw_losses:
        loss_bw = torch.stack([l if l.ndim == 0 else l.mean() for l in bw_losses]).mean()
    else:
        loss_bw = specs.new_tensor(0.0)
    return out, loss_bw, final_losses


def _train_spectral_batch(
    model: torch.nn.Module,
    batch,
    config: MnistTrainConfig,
    device: str,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Dict[str, float]:
    specs, labels, _ = _unpack_mnist_batch(batch)
    specs = specs.to(device)
    labels = labels.to(device)

    if config.reset_model_each_batch or model.batch_size != specs.shape[0]:
        model.reset(batch_size=specs.shape[0])
    else:
        model.detach_states()
    optimizer.zero_grad()

    if config.zero_steps > 0:
        zero_inp = torch.zeros_like(specs)
        for _ in range(config.zero_steps):
            model.clamp(input_data=zero_inp)
            model.backward()
            model.forward(zero_inp, step=config.step)

    out, loss_bw, final_losses_list = _run_spectral_inference(model, specs, config)
    loss_fw = F.cross_entropy(out, labels)
    loss = config.loss_fw_weight * loss_fw + config.loss_bw_weight * loss_bw

    loss.backward()
    if config.gradient_clipping != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
    grad_norm = _compute_grad_norm(model)
    per_layer_grad_norms = _compute_per_layer_grad_norms(model)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    layer_weight_norms = _compute_per_layer_weight_norms(model)

    acc = out.argmax(dim=1).eq(labels).float().mean().item() * 100.0
    layer_backward_losses = {f"layer_{i}": v for i, v in enumerate(final_losses_list)}
    precisions = {f"layer_{i}": model.layers[i].pi.mean().item() for i in range(len(model.layers)) if hasattr(model.layers[i], "pi")}
    return {
        "loss": loss.item(),
        "loss_fw": loss_fw.item(),
        "loss_bw": loss_bw.item(),
        "grad_norm": grad_norm,
        "acc": acc,
        "precisions": precisions,
        "layer_backward_losses": layer_backward_losses,
        "layer_grad_norms": per_layer_grad_norms,
        "layer_weight_norms": layer_weight_norms,
    }


def evaluate_spectral_mnist(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    config: MnistTrainConfig,
    reset: bool = True,
):
    device = _resolve_device(config.device)
    model.eval()
    accuracies = []
    preds = []
    targets = []
    with torch.no_grad():
        for batch in test_dataloader:
            specs, labels, _ = _unpack_mnist_batch(batch)
            specs = specs.to(device)
            labels = labels.to(device)

            if reset or model.batch_size != specs.shape[0]:
                model.reset(batch_size=specs.shape[0])
            else:
                model.detach_states()

            if config.zero_steps > 0:
                zero_inp = torch.zeros_like(specs)
                for _ in range(config.zero_steps):
                    model.clamp(input_data=zero_inp)
                    model.backward()
                    model.forward(zero_inp, step=config.step)

            out, _, _ = _run_spectral_inference(model, specs, config)
            pred = out.argmax(dim=1)
            preds.append(pred.detach().cpu())
            targets.append(labels.detach().cpu())
            accuracies.append(100.0 * pred.eq(labels).float().mean().item())

    avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
    preds_tensor = torch.cat(preds, dim=0) if preds else torch.empty(0, dtype=torch.long)
    targets_tensor = torch.cat(targets, dim=0) if targets else torch.empty(0, dtype=torch.long)
    return avg_acc, preds_tensor, targets_tensor


def train_spectral_mnist(
    config: MnistTrainConfig,
    model: Optional[torch.nn.Module] = None,
    dataloaders: Optional[Tuple[DataLoader, DataLoader]] = None,
    dataset: Optional[SpectrogramMNIST] = None,
) -> MnistTrainResult:
    if dataloaders is None:
        dataset, train_dataloader, val_dataloader = build_mnist_dataloaders(config)
    else:
        train_dataloader, val_dataloader = dataloaders
        if model is None and dataset is None:
            raise ValueError("dataset is required when providing dataloaders and no model.")

    device = _resolve_device(config.device)
    state_dict = None
    if config.state_dict_path is not None:
        state_dict = torch.load(config.state_dict_path, map_location=device)

    if model is None:
        if dataset is None:
            raise ValueError("dataset is required to initialize spectral_mnist model.")
        batch_size = train_dataloader.batch_size or config.batch_size or 1
        model = spectral_mnist(
            num_words=config.num_classes,
            batch_size=batch_size,
            noise=config.noise,
            use_weight_norm=config.use_weight_norm,
            use_sparse_weight_norm=config.use_sparse_weight_norm,
            leakage=config.leakage,
            top_down=config.top_down,
            clamp_negatives=config.clamp_negatives,
            cnn_params=config.cnn_params,
            use_precision=config.use_precision,
            use_rnn=config.use_rnn,
            width=config.width,
            time_dim=config.time_dim,
            freq_dim=dataset.freq_dim,
        )
    print(model)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=config.state_dict_strict)
    model = model.to(device)

    if config.nadam_optimizer:
        optimizer = torch.optim.NAdam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.epochs * max(1, len(train_dataloader))),
            eta_min=config.scheduler_min_lr,
        )
        if config.scheduler
        else None
    )

    history: Dict[str, List] = {
        "loss": [],
        "loss_fw": [],
        "loss_bw": [],
        "grad_norm": [],
        "acc": [],
        "val_acc": [0.0],
        "precisions": [],
        "layer_backward_losses": [],
        "layer_grad_norms": [],
        "layer_weight_norms": [],
    }
    total_steps = config.epochs * len(train_dataloader)
    pbar = tqdm(total=total_steps, desc="Training SpectralMNIST") if config.use_tqdm else None

    try:
        acc = 0.0
        for epoch in range(1, config.epochs + 1):
            model.train()
            for batch in train_dataloader:
                metrics = _train_spectral_batch(
                    model, batch, config, device, optimizer, scheduler=scheduler
                )
                for key in ("loss", "loss_fw", "loss_bw", "grad_norm", "acc", "precisions", "layer_backward_losses", "layer_grad_norms", "layer_weight_norms"):
                    history[key].append(metrics[key])
                if pbar is not None:
                    pbar.set_description(
                        f"Epoch {epoch}/{config.epochs} "
                        f"Loss: {metrics['loss']:.4f} TrainAcc: {history['acc'][-1]:.2f}, ValAcc: {history['val_acc'][-1]:.2f}, "
                        f"Grad Norm: {metrics['grad_norm']:.4f}, "
                        f"FW Loss: {metrics['loss_fw']:.4f}, BW Loss: {metrics['loss_bw']:.4f}, "
                        f"Precisions: {[f'{l}: {p:.2f}' for l, p in metrics['precisions'].items()]}, "
                    )
                    pbar.update(1)

            val_acc, _, _ = evaluate_spectral_mnist(
                model, val_dataloader, config, reset=True
            )
            history["val_acc"].append(val_acc)

            if config.early_stop_on_train_acc and history["acc"] is not None:
                if history["acc"][-1] >= config.early_stop_threshold:
                    break
    finally:
        if pbar is not None:
            pbar.close()

    return MnistTrainResult(
        model=model,
        history=history,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
    )


# ---------------------------------------------------------------------------
# Legacy / convenience wrappers
# ---------------------------------------------------------------------------

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
    use_sparse_weight_norm=False,
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
        use_sparse_weight_norm=use_sparse_weight_norm,
    )
    return train_trace_model(config=config, model=model)


def evaluate_trace_model(
    model: torch.nn.Module, test_dataloader: DataLoader, config: TraceTrainConfig, reset: bool = True
):
    device = _resolve_device(config.device)
    conv_ph: int = (config.cnn_params or {}).get("convolved_phonemes", 3)
    model.eval()
    accuracies = []
    preds = []
    original_words = []
    with torch.no_grad():
        for batch in test_dataloader:
            words, features, labels_ind, _, _ = _unpack_trace_batch(batch)
            features = features.to(device)
            labels_ind = labels_ind.to(device)
            seq_len = features.shape[1]

            if reset or model.batch_size != features.shape[0]:
                model.reset(batch_size=features.shape[0])
            else:
                model.detach_states()
            zero_inp = _zero_input(features, cnn=config.cnn, convolved_phonemes=conv_ph)
            for _ in range(config.zero_steps):
                model.clamp(input_data=zero_inp)
                model.backward()
                model.forward(zero_inp, step=config.step)

            if config.cnn:
                stride: int = (config.cnn_params or {}).get("stride", conv_ph)
                padded = _pad_features_for_sliding_window(features, conv_ph)
                for orig_pos in range(0, seq_len, stride):
                    inp = padded[:, orig_pos:orig_pos + conv_ph, :].permute(0, 2, 1).unsqueeze(2)
                    model.clamp(input_data=inp)
                    for _ in range(config.steps_per_phoneme):
                        model.backward()
                        model.forward(inp, step=config.step)
            else:
                for i in range(seq_len):
                    inp = features[:, i, :]
                    model.clamp(input_data=inp)
                    for _ in range(config.steps_per_phoneme):
                        model.backward()
                        model.forward(inp, step=config.step)

            preds.append(model.layers.output.state.argmax(dim=1))
            correct = preds[-1].eq(labels_ind).sum().item()
            accuracies.append(100.0 * correct / len(labels_ind))
            original_words.extend(words)
    avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
    preds_tensor = torch.cat(preds, dim=0) if preds else torch.empty(0, dtype=torch.long)
    return avg_acc, preds_tensor, original_words, preds[-1].eq(labels_ind).cpu().numpy() if preds else []


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
    avg_acc, preds_tensor, original_words, correct = evaluate_trace_model(model, test_dataloader, config)
    return avg_acc, preds_tensor, original_words, correct
