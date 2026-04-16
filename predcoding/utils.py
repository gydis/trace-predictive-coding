import seaborn as sns
import matplotlib.pyplot as plt
from training import train_trace_model
import copy
import itertools as it
from dataclasses import replace


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

    # for layer in bw_loss_layers:
    #     axes[7].plot(bw_loss_dict[layer], label=layer+f" {label}")
    # axes[7].set_title("Backward Loss per Layer")
    # axes[7].set_xlabel("Training step")
    # axes[7].set_ylabel("Backward loss")
    # axes[7].legend()

    # for layer in grad_norm_layers:
    #     axes[8].plot(grad_norm_dict[layer], label=layer+f" {label}")
    # axes[8].set_title("Weight Gradient Norm per Layer")
    # axes[8].set_xlabel("Training step")
    # axes[8].set_ylabel("Gradient norm")
    # axes[8].legend()

    # for layer in weight_norm_layers:
    #     axes[9].plot(weight_norm_dict[layer], label=layer+f" {label}")
    # axes[9].set_title("Weight Matrix Norm per Layer")
    # axes[9].set_xlabel("Training step")
    # axes[9].set_ylabel("Weight norm")
    # axes[9].legend()

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