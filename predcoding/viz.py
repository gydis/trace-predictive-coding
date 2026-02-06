import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import animation
from tqdm import tqdm

from layers import ConvLayer, InputLayer, MiddleLayer, OutputLayer, FcLayer


# def viz(model, input, n_steps=200, step_size=0.05, scales=None):
#     """Produce an animation of the predictive coding model.

#     Parameters
#     ----------
#     model : PCModel
#         The model to visualize.
#     input : torch.Tensor
#         The input to give to the model.
#     step_size: float
#         The step size.
#     n_steps : int
#         The number of steps to run the simulation for.
#     scales : list | None
#         For each layer, the scale (vmin/vmax) to use for the colormap.

#     Returns
#     -------
#     anim : matplotlib.animation.FuncAnimation
#         The matplotlib animation.
#     """
#     model.eval()

#     layer_inds_to_plot = [
#         i for i, layer in enumerate(model.layers) if hasattr(layer, "state")
#     ]
#     n_layers = len(layer_inds_to_plot)

#     if scales is None:
#         scales = [1] * n_layers
#     else:
#         assert len(scales) == n_layers

#     init_state = model.state_dict()

#     fig, axes = plt.subplots(ncols=n_layers, nrows=3, figsize=(3 * n_layers, 9))

#     axes[0][0].text(
#         x=-0.1,
#         y=0.5,
#         s="state",
#         transform=axes[0][0].transAxes,
#         va="center",
#         ha="right",
#         size=12,
#     )
#     axes[1][0].text(
#         x=-0.1,
#         y=0.5,
#         s="prediction",
#         transform=axes[1][0].transAxes,
#         va="center",
#         ha="right",
#         size=12,
#     )
#     axes[2][0].text(
#         x=-0.1,
#         y=0.5,
#         s="error",
#         transform=axes[2][0].transAxes,
#         va="center",
#         ha="right",
#         size=12,
#     )

#     state_images = []
#     for layer_ind, scale, ax in zip(layer_inds_to_plot, scales, axes[0]):
#         state_images.append(
#             ax.imshow(np.zeros((1, 1)), vmin=-scale, vmax=scale, cmap="RdBu_r")
#         )
#         ax.set_axis_off()
#         if isinstance(model.layers[layer_ind], InputLayer):
#             ax.set_title("input")
#         elif isinstance(model.layers[layer_ind], FcLayer):
#             ax.set_title("phoneme")
#         elif isinstance(model.layers[layer_ind], FcLayer):
#             ax.set_title("word")
#         elif isinstance(model.layers[layer_ind], OutputLayer):
#             ax.set_title("output")
#     state_images = tuple(state_images)

#     recon_images = []
#     for layer_ind, scale, ax in zip(layer_inds_to_plot[:-1], scales[:-1], axes[1]):
#         recon_images.append(
#             ax.imshow(np.zeros((1, 1)), vmin=-scale, vmax=scale, cmap="RdBu_r")
#         )
#         ax.set_axis_off()
#     axes[1][-1].set_axis_off()
#     axes[2][0].set_axis_off()
#     recon_images = tuple(recon_images)

#     err_images = []
#     for layer_ind, scale, ax in zip(layer_inds_to_plot, scales, axes[2]):
#         err_images.append(
#             ax.imshow(np.zeros((1, 1)), vmin=-scale, vmax=scale, cmap="RdBu_r")
#         )
#         ax.set_axis_off()
#     err_images = tuple(err_images)

#     progress = tqdm(total=n_steps + 1)

#     def update_images(layer_images, layer_data):
#         for layer_ind, layer_image, data in zip(
#             layer_inds_to_plot, layer_images, layer_data
#         ):
#             if isinstance(model.layers[layer_ind], ConvLayer):
#                 filter_grid_width = int(np.ceil(np.sqrt(data.shape[0])))
#                 image_height = filter_grid_width * data.shape[1]
#                 image_width = filter_grid_width * data.shape[2]
#                 image = np.zeros((image_height, image_width))
#                 y = 0
#                 x = 0
#                 for conv_filter in data:
#                     filt_height, filt_width = conv_filter.shape
#                     image[y : y + filt_height, x : x + filt_width] = conv_filter
#                     x += filt_width
#                     if x >= image_width:
#                         x = 0
#                         y += filt_height
#             elif data.shape[-1] < 20:
#                 image_height = data.shape[-1]
#                 image_width = int(np.prod(data.shape[:-1]))
#                 image = np.zeros((image_height, image_width))
#                 image.ravel()[: data.nelement()] = data.ravel()
#             elif isinstance(model.layers[layer_ind], InputLayer):
#                 image_height = model.layers[layer_ind].shape[2]
#                 image_width = model.layers[layer_ind].shape[3]
#                 image = np.zeros((image_height, image_width))
#                 image.ravel()[: data.nelement()] = data.ravel()
#             else:
#                 image_width = int(np.ceil(np.sqrt(data.nelement())))
#                 image_height = image_width
#                 image = np.zeros((image_height, image_width))
#                 image.ravel()[: data.nelement()] = data.ravel()
#                 image.ravel()[data.nelement() :] = np.nan
#             layer_image.set_data(image)

#     def animate(i):
#         if i == 0:
#             model.load_state_dict(init_state)
#             # progress.reset()
#         else:
#             model(input, step=step_size)
#             model.backward()
#         state = [
#             torch.clamp(model.layers[layer_ind].state[0].detach().cpu(), 0)
#             for layer_ind in layer_inds_to_plot
#         ]
#         recon = [
#             torch.clamp(model.layers[layer_ind].reconstruction[0].detach().cpu(), 0)
#             for layer_ind in layer_inds_to_plot[:-1]
#         ]
#         err = [
#             model.layers[layer_ind].pred_err[0].detach().cpu()
#             for layer_ind in layer_inds_to_plot
#         ]
#         update_images(state_images, state)
#         update_images(recon_images, recon)
#         update_images(err_images, err)
#         progress.update(1)
#         return state_images + recon_images + err_images

#     anim = animation.FuncAnimation(
#         fig, animate, frames=n_steps, interval=100, blit=False, repeat=False
#     )
#     return anim


def viz_trace(
    model,
    stimulus,
    *,
    steps_per_phoneme=5,
    step_size=0.05,
    scales=None,
    lexicon=None,
    interval=100,
    repeat=False,
    use_tqdm=True,
):
    """Produce an animation of a TRACE model while it processes an input word.

    This mirrors the style of :func:`viz`, but runs the TRACE model over a sequence of
    phoneme feature vectors (one per phoneme position), performing multiple inference
    steps per phoneme position.

    Parameters
    ----------
    model : PCModel
        A TRACE-like model produced by ``models.trace``.
    stimulus : dict | torch.Tensor
        Either a dataset item (dict containing a ``'features'`` tensor and optionally
        a ``'word'`` string), or the features tensor directly. Feature shapes accepted:
        ``(seq_len, n_features)`` or ``(batch, seq_len, n_features)``.
    steps_per_phoneme : int
        How many inference iterations to perform for each phoneme position.
    step_size : float
        The step size used in ``model.forward(..., step=step_size)``.
    scales : list | dict | None
        Colormap scale (vmin/vmax) per plotted layer. If a dict is provided, keys are
        layer names from ``model.layers``. If left as ``None``, a heuristic scale is
        chosen (input is scaled to the max absolute feature value; others default to 1).
    lexicon : list[str] | None
        Optional mapping from output index to a word string for display.
    interval : int
        Delay between frames in milliseconds.
    repeat : bool
        Whether the animation should repeat after the last frame.
    use_tqdm : bool
        Whether to show a tqdm progress bar while frames are generated.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The matplotlib animation.
    """
    model.eval()
    model.reset()

    if isinstance(stimulus, dict):
        features = stimulus["features"]
        word = stimulus.get("word")
    else:
        features = stimulus
        word = None

    if not isinstance(features, torch.Tensor):
        raise TypeError("stimulus must be a dict with a 'features' tensor or a tensor.")
    if features.ndim == 2:
        features = features.unsqueeze(0)
    if features.ndim != 3:
        raise ValueError(
            "features must have shape (seq_len, n_features) or (batch, seq_len, n_features)."
        )
    if features.shape[0] != 1:
        features = features[:1]
        word = word[0] if word is not None else None

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = next(model.buffers()).device
    features = features.to(model_device)

    batch_size, seq_len, _ = features.shape
    if steps_per_phoneme < 1:
        raise ValueError("steps_per_phoneme must be >= 1.")

    # Layers to plot (i.e., those with state).
    layer_names = list(model.layers._modules.keys())
    layer_inds_to_plot = [
        i for i, layer in enumerate(model.layers) if hasattr(layer, "state")
    ]
    layer_names_to_plot = [layer_names[i] for i in layer_inds_to_plot]
    n_layers = len(layer_inds_to_plot)

    if scales is None:
        input_scale = float(features.abs().max().item()) if features.numel() else 1.0
        scales = [max(1.0, input_scale)] + [1.0] * (n_layers - 1)
    elif isinstance(scales, dict):
        scales = [float(scales.get(name, 1.0)) for name in layer_names_to_plot]
    else:
        assert len(scales) == n_layers

    # Initialize model state for a clean run.
    model.reset(batch_size=batch_size)
    model.release_clamp()
    init_state = {k: v.clone() for k, v in model.state_dict().items()}

    n_frames = 1 + seq_len * steps_per_phoneme

    fig, axes = plt.subplots(ncols=n_layers, nrows=3, figsize=(3 * n_layers, 9))
    title_text = fig.suptitle("", y=0.98)

    axes[0][0].text(
        x=-0.1,
        y=0.5,
        s="state",
        transform=axes[0][0].transAxes,
        va="center",
        ha="right",
        size=12,
    )
    axes[1][0].text(
        x=-0.1,
        y=0.5,
        s="prediction",
        transform=axes[1][0].transAxes,
        va="center",
        ha="right",
        size=12,
    )
    axes[2][0].text(
        x=-0.1,
        y=0.5,
        s="error",
        transform=axes[2][0].transAxes,
        va="center",
        ha="right",
        size=12,
    )

    state_images = []
    for layer_ind, layer_name, scale, ax in zip(
        layer_inds_to_plot, layer_names_to_plot, scales, axes[0]
    ):
        state_images.append(
            ax.imshow(np.zeros((1, 1)), vmin=-scale, vmax=scale, cmap="RdBu_r")
        )
        ax.set_axis_off()
        ax.set_title(layer_name)
    state_images = tuple(state_images)

    recon_images = []
    for layer_ind, scale, ax in zip(layer_inds_to_plot[:-1], scales[:-1], axes[1]):
        recon_images.append(
            ax.imshow(np.zeros((1, 1)), vmin=-scale, vmax=scale, cmap="RdBu_r")
        )
        ax.set_axis_off()
    axes[1][-1].set_axis_off()
    recon_images = tuple(recon_images)

    err_images = []
    for layer_ind, scale, ax in zip(layer_inds_to_plot, scales, axes[2]):
        err_images.append(
            ax.imshow(np.zeros((1, 1)), vmin=-scale, vmax=scale, cmap="RdBu_r")
        )
        ax.set_axis_off()
    err_images = tuple(err_images)

    def update_images(layer_images, layer_data):
        for layer_ind, layer_image, data in zip(
            layer_inds_to_plot, layer_images, layer_data
        ):
            # if np.prod(data.shape) < 20:
            #     image_height = data.shape[-1]
            #     image_width = 1
            #     image = np.zeros((image_height, image_width))
            #     image.ravel()[: data.nelement()] = data.ravel()
            # elif data.shape[-1] < 20:
            #     image_height = int(np.prod(data.shape[:-1]))
            #     image_width = data.shape[-1]
            #     image = np.zeros((image_height, image_width))
            #     image.ravel()[: data.nelement()] = data.ravel()
            # elif isinstance(model.layers[layer_ind], InputLayer):
            #     image_height = model.layers[layer_ind].shape[2]
            #     image_width = model.layers[layer_ind].shape[3]
            #     image = np.zeros((image_height, image_width))
            #     image.ravel()[: data.nelement()] = data.ravel()
            # else:
            #     image_width = int(np.ceil(np.sqrt(data.nelement())))
            #     image_height = image_width
            #     image = np.zeros((image_height, image_width))
            #     image.ravel()[: data.nelement()] = data.ravel()
            #     image.ravel()[data.nelement() :] = np.nan
            if np.prod(data.shape) < 50:
                image_height = data.shape[-1]
                image_width = 1
                image = np.zeros((image_height, image_width))
                image.ravel()[: data.nelement()] = data.ravel()
            else:
                image_width = int(np.ceil(np.sqrt(data.nelement())))
                image_height = image_width
                image = np.zeros((image_height, image_width))
                image.ravel()[: data.nelement()] = data.ravel()
                image.ravel()[data.nelement() :] = np.nan

            # print(f"Layer ind: {layer_ind}, data shape: {data.shape}")
            layer_image.set_data(image)

    progress = tqdm(total=n_frames) if use_tqdm else None

    def _set_title(phoneme_idx, iter_in_phoneme):
        parts = []
        if word is not None:
            parts.append(f"word='{word}'")
        parts.append(f"phoneme={word[phoneme_idx] if phoneme_idx < len(word) else ''}({phoneme_idx + 1}/{seq_len})")
        parts.append(f"iter={iter_in_phoneme + 1}/{steps_per_phoneme}")
        try:
            out_state = model.layers[-1].state[0].detach()
            pred_ind = int(out_state.argmax().item())
            if lexicon is not None and 0 <= pred_ind < len(lexicon):
                parts.append(f"pred='{lexicon[pred_ind]}'")
            else:
                parts.append(f"pred_idx={pred_ind}")
        except Exception:
            pass
        title_text.set_text("  ".join(parts))

    def animate(i):
        with torch.no_grad():
            if i == 0:
                model.load_state_dict(init_state, strict=True)
                model.release_clamp()
                model.clamp(input_data=features[:, 0, :].clone())
                _ = model.backward()
                _ = model.forward(None, step=0.0)
                _ = model.backward()
                phoneme_idx, iter_in_phoneme = 0, 0
            else:
                phoneme_idx = (i - 1) // steps_per_phoneme
                iter_in_phoneme = (i - 1) % steps_per_phoneme
                phoneme_idx = min(phoneme_idx, seq_len - 1)
                model.clamp(input_data=features[:, phoneme_idx, :])
                _ = model.backward()
                _ = model.forward(None, step=step_size)
                _ = model.backward()

            state = [
                model.layers[layer_ind].state[0].detach().cpu()
                for layer_ind in layer_inds_to_plot
            ]
            recon = [
                model.layers[layer_ind].reconstruction[0].detach().cpu()
                for layer_ind in layer_inds_to_plot[:-1]
            ]
            err = [
                model.layers[layer_ind].pred_err[0].detach().cpu()
                for layer_ind in layer_inds_to_plot
            ]

        update_images(state_images, state)
        update_images(recon_images, recon)
        update_images(err_images, err)
        _set_title(phoneme_idx, iter_in_phoneme)

        # print(f"Word: {word}, Frame{i}, Phoneme index: {phoneme_idx}, Iter in phoneme: {iter_in_phoneme}")
        # print(f"input state: {model.layers[0].state[0].detach().cpu()}")
        # print(f"phoneme feature: {features[:, phoneme_idx, :].detach().cpu()}")

        if progress is not None:
            progress.update(1)
            if i == n_frames - 1:
                progress.close()

        return state_images + recon_images + err_images

    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames, interval=interval, blit=False, repeat=repeat
    )
    return anim


def plot_prederr(errors, mark=0, labels=None):
    """Plot prediction error over time.

    Parameters
    ----------
    errors : torch.Tensor | list of torch.Tensor, shape (n_stimuli, n_layers, n_steps)
        For each layer in the model, the total prediction error over time. You can pass
        a list of tensors to overlay multiple conditions.
    mark : int
        Timestep to mark with a vertical line (usually indicates the start of the target
        stimulus).
    labels : list of str | None
        The name of each condition. Only used when overlaying multiple conditions.

    Returns
    -------
    fig : matplotlib.Figure
        The produced figure.
    """
    if not isinstance(errors, list):
        errors = [errors]
    n_layers = errors[0].shape[1]

    fig, axes = plt.subplots(ncols=n_layers, sharex=True, figsize=(3 * n_layers, 3))
    for cond_ind, cond_err in enumerate(errors):
        cond_err = cond_err.mean(0)
        for layer_ind, layer_err in enumerate(cond_err):
            axes[layer_ind].plot(
                layer_err,
                label=labels[cond_ind] if labels is not None else None,
                color=f"C{cond_ind}",
            )
            axes[layer_ind].axvline(mark, color="k", linestyle="--")
            axes[layer_ind].set_xlabel("iterations")
    if labels is not None:
        axes[-1].legend(bbox_to_anchor=(1.05, 1))
    axes[0].set_ylabel("prediction error")
    fig.set_tight_layout(True)
    return fig


# def plot_prederr_init(
#     model, dataset, n_zeros=200, n_pre_iter=200, n_iter=200, step=0.05, baseline=0
# ):
#     """Plot prediction error with a model initialized to a previous batch."""
#     data_iter = iter(dataset)
#     model.reset()
#     model.release_clamp()
#     data1, target1 = next(data_iter)
#     data2, target2 = next(data_iter)
#     if isinstance(target1, list):
#         target1, vectors1 = target1
#     if isinstance(target2, list):
#         target2, vectors2 = target2
#     data1, target1 = data1.to(model.device), target1.to(model.device)
#     data2, target2 = data2.to(model.device), target2.to(model.device)
#     zeros = torch.zeros_like(data1)
#
#     errs, *acc = run_model(
#         model=model,
#         data=[zeros, data1, zeros, data2],
#         target=target2,
#         vectors=dataset.dataset.vectors,
#         n_iter=[n_zeros, n_pre_iter, n_zeros, n_iter],
#         step=step,
#     )
#     print("Accuracy:", acc)
#
#     fig, axes = plt.subplots(ncols=len(errs), sharex=True, figsize=(15, 3))
#     for ax, e in zip(axes, errs):
#         ax.plot(e[baseline:])
#         ax.axvline(2 * n_zeros + n_pre_iter - baseline, color="k", linestyle="--")
#
#
# def plot_repetition_priming(
#     model,
#     dataloader,
#     n_iter=200,
#     step=0.05,
#     n_pre_iter=100,
#     n_pre_zeros=100,
#     n_zeros=100,
#     baseline=0,
# ):
#     """Perform repetition priming experiment."""
#     stimuli = pd.read_csv("/m/nbe/archive/viswordrec/stimuli.csv", index_col=0)
#     stimuli = stimuli[["prime", "target", "trigger_code_target"]]
#     stimuli.columns = ["prime", "target", "condition"]
#     stimuli.replace(dict(condition=11), "repetition", inplace=True)
#     stimuli.replace(dict(condition=12), "unrelated", inplace=True)
#     stimuli.replace(dict(condition=13), "related-orthographic", inplace=True)
#     stimuli.replace(dict(condition=14), "related-uppercase", inplace=True)
#     stimuli.replace(dict(condition=15), "related-position", inplace=True)
#     stimuli.loc[stimuli.condition == "related-uppercase", "target"] = stimuli.loc[
#         stimuli.condition == "related-uppercase", "target"
#     ].str.upper()
#     stimuli_storysem = pd.read_csv(
#         "/m/nbe/project/storysem/stimuli/stimuli.csv", index_col=0, quotechar="'"
#     )
#     stimuli_storysem = stimuli_storysem[["primer2", "target", "dist_group"]]
#     stimuli_storysem.columns = ["prime", "target", "condition"]
#     stimuli_storysem.replace(dict(condition=0), "related-semantic", inplace=True)
#     stimuli_storysem.replace(dict(condition=2), "semantic-weak", inplace=True)
#     stimuli_storysem.replace(dict(condition=4), "semantic-unrelated", inplace=True)
#     stimuli = pd.concat([stimuli, stimuli_storysem], ignore_index=True)
#
#     transform = dataloader.dataset.transform
#     data1 = render_words(
#         stimuli.prime, transform, font="training_datasets/fonts/arial.ttf"
#     )
#     data2 = render_words(
#         stimuli.target, transform, font="training_datasets/fonts/arial.ttf"
#     )
#     target = torch.tensor(
#         [dataloader.dataset.classes.tolist().index(x.lower()) for x in stimuli.target]
#     )
#
#     """Plot prediction error with a model initialized to a previous batch."""
#     data1, data2, target = (
#         data1.to(model.device),
#         data2.to(model.device),
#         target.to(model.device),
#     )
#     zeros = torch.zeros_like(data1)
#
#     fig, axes = plt.subplots(ncols=len(model.layers) - 1, sharex=True, figsize=(15, 3))
#     conditions = stimuli.condition.unique()
#     for i, cond in enumerate(conditions):
#         ind = stimuli.query(f"condition == '{cond}'").index
#         errs, *acc = run_model(
#             model=model,
#             data=[zeros[ind], data1[ind], zeros[ind], data2[ind]],
#             target=target[ind],
#             vectors=dataloader.dataset.vectors,
#             n_iter=[n_pre_zeros, n_pre_iter, n_zeros, n_iter],
#             step=step,
#         )
#         for ax, e in zip(axes, errs):
#             ax.plot(e[baseline:], color=f"C{i}", label=cond)
#             ax.axvline(
#                 n_pre_zeros + n_zeros + n_pre_iter - baseline, color="k", linestyle="--"
#             )
#         print(f"Accuracy ({cond}):", acc)
#     ax.legend()
#     return fig
