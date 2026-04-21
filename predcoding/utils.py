import seaborn as sns
import matplotlib.pyplot as plt
from training import train_trace_model
import copy
import itertools as it
from dataclasses import replace
from dataloader import TraceDataset
import torch
import random
from training import evaluate_trace_model


def plot_history(history, label="", mpl=(None, None)):

    losses = history["loss"]
    losses_fw = history["loss_fw"]
    losses_bw = history["loss_bw"]
    gradients = history["grad_norm"]
    accuracies = history["acc"]
    val_accuracies = history["val_acc"]

    precision_layers = list(history["precisions"][0].keys())
    precision_dict = {layer: [] for layer in precision_layers}
    for precision in history["precisions"]:
        for layer in precision_layers:
            precision_dict[layer].append(precision[layer])

    bw_loss_layers = list(history["layer_backward_losses"][0].keys())
    bw_loss_dict = {
        layer: [step[layer] for step in history["layer_backward_losses"]]
        for layer in bw_loss_layers
    }

    grad_norm_layers = list(history["layer_grad_norms"][0].keys())
    grad_norm_dict = {
        layer: [step[layer] for step in history["layer_grad_norms"]]
        for layer in grad_norm_layers
    }

    weight_norm_layers = list(history["layer_weight_norms"][0].keys())
    weight_norm_dict = {
        layer: [step[layer] for step in history["layer_weight_norms"]]
        for layer in weight_norm_layers
    }

    if label == "":
        fig, axes = plt.subplots(5, 2, figsize=(18, 28), sharex=False)
        axes = axes.flatten()
    else:
        fig, axes = mpl

    axes[0].plot(range(len(accuracies)), accuracies, label=f"{label}")
    axes[0].set_title("Train Accuracy")
    axes[0].legend()    

    axes[1].plot(range(len(val_accuracies)), val_accuracies, label=f"{label}")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()

    axes[2].plot(range(len(losses)), losses, label=f"{label}")
    axes[2].set_title("Total Loss")
    axes[2].legend()

    axes[3].plot(range(len(losses_fw)), losses_fw, label=f"{label}")
    axes[3].set_title("Forward Loss")
    axes[3].legend()

    axes[4].plot(range(len(losses_bw)), losses_bw, label=f"{label}")
    axes[4].set_title("Backward Loss")
    axes[4].legend()

    axes[5].plot(range(len(gradients)), gradients, label=f"{label}")
    axes[5].set_title("Gradient Norm")
    axes[5].legend()

    for layer in precision_layers:
        axes[6].plot(
            range(len(precision_dict[layer])),
            precision_dict[layer],
            label=f"{layer} {label}",
        )
    axes[6].set_title("Precision")
    axes[6].legend()
    if label == "":
        for layer in bw_loss_layers:
            axes[7].plot(bw_loss_dict[layer], label=layer+f" {label}")
        axes[7].set_title("Backward Loss per Layer")
        axes[7].set_xlabel("Training step")
        axes[7].set_ylabel("Backward loss")
        axes[7].legend()

        for layer in grad_norm_layers:
            axes[8].plot(grad_norm_dict[layer], label=layer+f" {label}")
        axes[8].set_title("Weight Gradient Norm per Layer")
        axes[8].set_xlabel("Training step")
        axes[8].set_ylabel("Gradient norm")
        axes[8].legend()

        for layer in weight_norm_layers:
            axes[9].plot(weight_norm_dict[layer], label=layer+f" {label}")
        axes[9].set_title("Weight Matrix Norm per Layer")
        axes[9].set_xlabel("Training step")
        axes[9].set_ylabel("Weight norm")
        axes[9].legend()

        for ax in axes:
            ax.grid(True, alpha=0.3)

        if label == "":
            plt.tight_layout()
            plt.show()


def plot_histories(histories: dict[str, dict]):
    fig, axes = plt.subplots(5, 2, figsize=(18, 28), sharex=False)
    axes = axes.flatten()
    for name, history in histories.items():
        print(f"Plotting history for {name}")
        plot_history(history, label=name, mpl=(fig, axes))
    plt.tight_layout()
    plt.show()

def run_experiments(params_to_vary: dict[str, list], base_params: dict):
    all_histories = {}
    for params in it.product(*params_to_vary.values()):
        param_dict = dict(zip(params_to_vary.keys(), params))
        experiment_params = copy.deepcopy(base_params)
        experiment_params = replace(experiment_params, **param_dict)
        name = ", ".join(f"{key}={value}" for key, value in param_dict.items())
        print(f"Running experiment: {name}")
        history = train_trace_model(experiment_params)
        all_histories[name] = history
    return all_histories

def pred_err_plots(model, config, dataloader, i=0):
    device = model.parameters().__next__().device
    conv_ph: int = (config.cnn_params or {}).get("convolved_phonemes", 3)
    stride: int = (config.cnn_params or {}).get("stride", conv_ph)
    words, features, _, words_padded = next(iter(dataloader)).values()
    features = features.to(device)
    model.to(device)
    model.release_clamp()
    word = words[i]

    model.eval()
    model.reset(batch_size=1)
    layer_inds_to_plot = [
        idx for idx, layer in enumerate(model.layers) if hasattr(layer, "state")
    ]
    pred_errs = {layer_ind: [] for layer_ind in layer_inds_to_plot}
    preds = []

    if config.cnn:
        zero_inp = torch.zeros(1, features.shape[2], 1, conv_ph, device=device)
    else:
        zero_inp = torch.zeros(1, features.shape[2], device=device)

    with torch.no_grad():
        for _ in range(config.zero_steps):
            model.clamp(input_data=zero_inp)
            model.backward()
            out = model.forward(zero_inp, step=config.step)
        pred_word = TraceDataset().words[torch.argmax(out, dim=1).item()]
        preds.append(pred_word)

        seq_len = len(words_padded[0])
        if config.cnn:
            phoneme_positions = list(range(0, seq_len, stride))
            pad = torch.zeros(1, conv_ph - 1, features.shape[2], device=device)
            padded_f = torch.cat([pad, features[i:i+1]], dim=1)
        else:
            phoneme_positions = list(range(seq_len))

        for orig_pos in phoneme_positions:
            if config.cnn:
                inp = padded_f[:, orig_pos:orig_pos + conv_ph, :].permute(0, 2, 1).unsqueeze(2)
            else:
                inp = features[i:i+1, orig_pos, :]
            model.clamp(input_data=inp)
            for _ in range(config.steps_per_phoneme):
                model.backward()
                out = model.forward(inp, step=config.step)
                for layer_ind in layer_inds_to_plot:
                    pred_errs[layer_ind].append(
                        torch.linalg.vector_norm(model.layers[layer_ind].pred_err.reshape(-1)).item()
                    )
            pred_word = TraceDataset().words[torch.argmax(out, dim=1).item()]
            preds.append(pred_word)

    # Build phoneme tick labels
    if config.cnn:
        padded_word = ('-' * (conv_ph - 1)) + words_padded[i]
        ph_labels = [padded_word[orig_pos:orig_pos + conv_ph] for orig_pos in phoneme_positions]
    else:
        ph_labels = [words_padded[i][ph] for ph in phoneme_positions]

    num_layers = len(layer_inds_to_plot)
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 3 * num_layers))
    if num_layers == 1:
        axes = [axes]

    layer_names = list(model.layers._modules.keys())
    for ax_idx, layer_ind in enumerate(layer_inds_to_plot):
        axes[ax_idx].plot(pred_errs[layer_ind])
        axes[ax_idx].set_title(f'Prediction Error for Layer {layer_names[layer_ind]}')
        axes[ax_idx].set_ylabel('Prediction Error Norm')
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].set_xlabel('Time Step')
        axes[ax_idx].set_xticks(np.arange(-1, len(pred_errs[layer_ind]), config.steps_per_phoneme))
        tick_labels = [f"0\n{preds[0]}"] + [f"{ph}\n{p}" for ph, p in zip(ph_labels, preds[1:])]
        axes[ax_idx].set_xticklabels(tick_labels)

    fig.suptitle(f'Prediction Error over Time for Word: {word}. Predicted: {pred_word}', fontsize=14)
    plt.tight_layout()
    plt.show()
    return pred_errs


def pred_err_stimuli(model, config, dataloader, stimuli=[10, "w", 10, "w"]):
    device = model.parameters().__next__().device
    conv_ph: int = (config.cnn_params or {}).get("convolved_phonemes", 3)
    stride: int = (config.cnn_params or {}).get("stride", conv_ph)
    words, features, _, words_padded = next(iter(dataloader)).values()
    features = features.to(device)
    model.to(device)
    model.release_clamp()

    if config.cnn:
        zero_inp_template = torch.zeros(1, features.shape[2], 1, conv_ph, device=device)
    else:
        zero_inp_template = torch.zeros(1, features.shape[2], device=device)

    model.eval()
    model.reset(batch_size=1)
    layer_inds_to_plot = [
        idx for idx, layer in enumerate(model.layers) if hasattr(layer, "state")
    ]
    pred_errs = {layer_ind: [] for layer_ind in layer_inds_to_plot}
    preds = [0]
    words_used = []
    word_inds = []
    phonemes = []

    seq_len = len(words_padded[0])
    if config.cnn:
        phoneme_positions = list(range(0, seq_len, stride))
    else:
        phoneme_positions = list(range(seq_len))

    with torch.no_grad():
        for s in stimuli:
            if isinstance(s, int):
                for step in range(s):
                    model.clamp(input_data=zero_inp_template)
                    model.backward()
                    out = model.forward(zero_inp_template, step=config.step)
                    for layer_ind in layer_inds_to_plot:
                        pred_errs[layer_ind].append(
                            torch.linalg.vector_norm(model.layers[layer_ind].pred_err.reshape(-1)).item()
                        )
                    if step % config.steps_per_phoneme == 0:
                        pred_word = TraceDataset().words[torch.argmax(out, dim=1).item()]
                        preds.append(pred_word)
                        word_inds.append(-1)
                        phonemes.append(0)
            elif s == "w" or (isinstance(s, str) and len(s) > 1 and s[0] == "w"):
                if s == "w":
                    idx = random.randint(0, features.shape[0] - 1)
                else:
                    idx = int(s[1:])
                word = words[idx]
                words_used.append(word)

                if config.cnn:
                    pad = torch.zeros(1, conv_ph - 1, features.shape[2], device=device)
                    padded_f = torch.cat([pad, features[idx:idx+1]], dim=1)
                    padded_word = ('-' * (conv_ph - 1)) + words_padded[idx]

                for orig_pos in phoneme_positions:
                    if config.cnn:
                        inp = padded_f[:, orig_pos:orig_pos + conv_ph, :].permute(0, 2, 1).unsqueeze(2)
                        ph_label = padded_word[orig_pos:orig_pos + conv_ph]
                    else:
                        inp = features[idx:idx+1, orig_pos, :]
                        ph_label = words_padded[idx][orig_pos]
                    model.clamp(input_data=inp)
                    for _ in range(config.steps_per_phoneme):
                        model.backward()
                        out = model.forward(inp, step=config.step)
                        for layer_ind in layer_inds_to_plot:
                            pred_errs[layer_ind].append(
                                torch.linalg.vector_norm(model.layers[layer_ind].pred_err.reshape(-1)).item()
                            )
                    pred_word = TraceDataset().words[torch.argmax(out, dim=1).item()]
                    preds.append(pred_word)
                    word_inds.append(idx)
                    phonemes.append(ph_label)
            else:
                raise ValueError(f"Invalid stimulus: {s}")

    num_layers = len(layer_inds_to_plot)
    fig, axes = plt.subplots(num_layers, 1, figsize=(20, 3 * num_layers))
    if num_layers == 1:
        axes = [axes]

    print(phonemes)

    layer_names = list(model.layers._modules.keys())
    for ax_idx, layer_ind in enumerate(layer_inds_to_plot):
        axes[ax_idx].plot(pred_errs[layer_ind])
        axes[ax_idx].set_title(f'Prediction Error for Layer {layer_names[layer_ind]}')
        axes[ax_idx].set_ylabel('Prediction Error Norm')
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].set_xlabel('Time Step')
        axes[ax_idx].set_xticks(np.arange(-1, len(pred_errs[layer_ind]), config.steps_per_phoneme))
        tick_labels = [f"{ph}\n{w}" for ph, w in zip(phonemes + ["end"], preds)]
        axes[ax_idx].set_xticklabels(tick_labels)

    fig.suptitle(
        f'Prediction error for stimulus {stimuli}, words used: {", ".join(words_used)}, Predicted: {pred_word}',
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()
    return pred_errs

def k_repeat(model, config, dataloader, zero_steps=None, k=3):
    from training import _run_zero_steps, _zero_input
    _, features, _, _ = next(iter(dataloader)).values()
    zero_inp = _zero_input(features, cnn=config.cnn, convolved_phonemes=(config.cnn_params or {}).get("convolved_phonemes", 3)).to(model.device)
    correct_acc = 1
    config.zero_steps = zero_steps or config.zero_steps
    model.reset(batch_size=features.shape[0])
    _run_zero_steps(model, zero_inp, config)
    for _ in range(k):
        avg_acc, preds_tensor, original_words, correct = evaluate_trace_model(model, dataloader, config, reset=False)
        _run_zero_steps(model, zero_inp, config)
        _run_zero_steps(model, zero_inp, config)
        correct_acc = correct_acc * correct
    accuracy = np.mean(correct_acc).item()*100
    print(f"Accuracy after {k} repeats: {accuracy:.2f}%")