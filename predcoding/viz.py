import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import animation
from matplotlib.patches import Rectangle
from tqdm import tqdm
from settings import PHONEME_TO_INDEX

from layers import ConvLayer, InputLayer, MiddleLayer, OutputLayer, FcLayer


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
    chosen_word_ind=None,
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
        target_word_idx = (
            stimulus.get("index")
            if "index" in stimulus
            else stimulus.get("target_index")
        )
        if target_word_idx is None:
            target_word_idx = stimulus.get("target")
        if target_word_idx is None:
            target_word_idx = stimulus.get("label")
    else:
        features = stimulus
        word = None
        target_word_idx = None

    if not isinstance(features, torch.Tensor):
        raise TypeError("stimulus must be a dict with a 'features' tensor or a tensor.")
    if features.ndim == 2:
        features = features.unsqueeze(0)
    if features.ndim != 3:
        raise ValueError(
            "features must have shape (seq_len, n_features) or (batch, seq_len, n_features)."
        )
    
    if chosen_word_ind is not None and word is not None:
        word = word[chosen_word_ind] if chosen_word_ind < len(word) else None
        features = features[chosen_word_ind:chosen_word_ind+1, :, :]
        target_word_idx = chosen_word_ind if target_word_idx is not None else None
    elif features.shape[0] != 1:
        features = features[:1]
        word = word[0] if word is not None else None
        if target_word_idx is not None:
            if isinstance(target_word_idx, torch.Tensor):
                target_word_idx = (
                    target_word_idx.reshape(-1)[0] if target_word_idx.numel() else None
                )
            elif isinstance(target_word_idx, (list, tuple, np.ndarray)):
                target_word_idx = target_word_idx[0] if len(target_word_idx) else None

    if target_word_idx is not None:
        if isinstance(target_word_idx, torch.Tensor):
            target_word_idx = (
                int(target_word_idx.reshape(-1)[0].item())
                if target_word_idx.numel()
                else None
            )
        elif isinstance(target_word_idx, np.ndarray):
            target_word_idx = (
                int(target_word_idx.reshape(-1)[0]) if target_word_idx.size else None
            )
        elif isinstance(target_word_idx, (list, tuple)):
            target_word_idx = int(target_word_idx[0]) if target_word_idx else None
        else:
            try:
                target_word_idx = int(target_word_idx)
            except (TypeError, ValueError):
                target_word_idx = None

    print(f"Word: {word}, Target word index: {target_word_idx}")

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
    word_layer_inds_to_plot = {
        layer_ind
        for layer_ind, layer_name in zip(layer_inds_to_plot, layer_names_to_plot)
        if (
            "word" in layer_name.lower()
            or "lexicon" in layer_name.lower()
            or layer_name.lower() == "output"
        )
    }
    phoneme_layer_inds_to_plot = {
        layer_ind
        for layer_ind, layer_name in zip(layer_inds_to_plot, layer_names_to_plot)
        if "phoneme" in layer_name.lower()
    }
    n_layers = len(layer_inds_to_plot)

    if target_word_idx is None and word is not None and lexicon is not None:
        if word in lexicon:
            target_word_idx = lexicon.index(word)
        elif isinstance(word, str) and word.lower() in lexicon:
            target_word_idx = lexicon.index(word.lower())

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

    # Detect CNN model by inspecting input layer state shape.
    input_state = model.layers[0].state
    is_cnn = input_state.ndim > 2
    conv_ph = input_state.shape[-1] if is_cnn else 1
    n_windows = seq_len // conv_ph  # number of inference steps in the animation

    n_frames = 1 + n_windows * steps_per_phoneme

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

    def _make_markers(layer_inds, axes_row):
        markers = []
        for layer_ind, ax in zip(layer_inds, axes_row):
            if (target_word_idx is not None and layer_ind in word_layer_inds_to_plot) or (layer_ind in phoneme_layer_inds_to_plot):
                # Double-outline box keeps marker visible on both dark and light cells.
                marker_outer = Rectangle(
                    (0, 0),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.4,
                    visible=False,
                    zorder=20,
                )
                marker_inner = Rectangle(
                    (0, 0),
                    1,
                    1,
                    fill=False,
                    edgecolor="yellow",
                    linewidth=1.4,
                    visible=False,
                    zorder=21,
                )
                ax.add_patch(marker_outer)
                ax.add_patch(marker_inner)
                markers.append((marker_outer, marker_inner))
            else:
                markers.append(None)
        return tuple(markers)

    state_markers = _make_markers(layer_inds_to_plot, axes[0])
    recon_markers = _make_markers(layer_inds_to_plot[:-1], axes[1])
    err_markers = _make_markers(layer_inds_to_plot, axes[2])
    marker_artists = tuple(
        artist
        for marker in (state_markers + recon_markers + err_markers)
        if marker is not None
        for artist in marker
    )

    def update_images(layer_inds, layer_images, layer_data, layer_markers=None, phoneme_idx=None):
        if layer_markers is None:
            layer_markers = [None] * len(layer_images)
        for layer_ind, layer_image, data, marker in zip(
            layer_inds, layer_images, layer_data, layer_markers
        ):
            # Squeeze singleton spatial dims so CNN tensors (e.g. (7,1,3)) become
            # 2D (7,3) or 1D (15,) before the display branching below.
            data = data.squeeze()
            if data.ndim == 0:
                data = data.unsqueeze(0)

            if data.ndim == 2:
                # Render 2D tensors directly: e.g. (7, conv_ph) CNN input shows
                # features on the Y axis and time windows on the X axis.
                image = data.numpy()
            elif data.nelement() < 50:
                image = data.reshape(-1, 1).numpy()
            else:
                side = int(np.ceil(np.sqrt(data.nelement())))
                image = np.full((side, side), np.nan)
                image.ravel()[:data.nelement()] = data.reshape(-1).numpy()

            # print(f"Layer ind: {layer_ind}, data shape: {data.shape}")
            layer_image.set_data(image)
            # Keep axis limits in sync with current image shape so overlays align.
            layer_image.set_extent(
                (-0.5, image.shape[1] - 0.5, image.shape[0] - 0.5, -0.5)
            )
            layer_image.axes.set_xlim(-0.5, image.shape[1] - 0.5)
            layer_image.axes.set_ylim(image.shape[0] - 0.5, -0.5)
            if marker is not None:
                marker_outer, marker_inner = marker
                if target_word_idx is not None and 0 <= target_word_idx < data.nelement():
                    row = target_word_idx // image.shape[1]
                    col = target_word_idx % image.shape[1]
                    marker_outer.set_xy((col - 0.5, row - 0.5))
                    marker_inner.set_xy((col - 0.5, row - 0.5))
                    marker_outer.set_visible(True)
                    marker_inner.set_visible(True)
                elif layer_ind in phoneme_layer_inds_to_plot and phoneme_idx is not None:
                    if 0 <= phoneme_idx < data.nelement():
                        row = phoneme_idx // image.shape[1]
                        col = phoneme_idx % image.shape[1]
                        marker_outer.set_xy((col - 0.5, row - 0.5))
                        marker_inner.set_xy((col - 0.5, row - 0.5))
                        marker_outer.set_visible(True)
                        marker_inner.set_visible(True)
                else:
                    marker_outer.set_visible(False)
                    marker_inner.set_visible(False)

    progress = tqdm(total=n_frames) if use_tqdm else None

    def _set_title(window_idx, ph_str, iter_in_phoneme):
        parts = []
        if word is not None:
            parts.append(f"word='{word}'")
        parts.append(
            f"target_idx={target_word_idx if target_word_idx is not None else 'NA'}"
        )
        parts.append(f"phoneme={ph_str}({window_idx + 1}/{n_windows})")
        parts.append(f"iter={iter_in_phoneme + 1}/{steps_per_phoneme}")
        try:
            out_state = model.layers[-1].state.detach()
            pred_ind = int(out_state.argmax(dim=1).item())
            if lexicon is not None and 0 <= pred_ind < len(lexicon):
                parts.append(f"pred='{lexicon[pred_ind]}'")
            else:
                parts.append(f"pred_idx={pred_ind}")
        except Exception as e:
            print(e)
            pass
        title_text.set_text("  ".join(parts))

    def animate(i):
        with torch.no_grad():
            if i == 0:
                model.load_state_dict(init_state, strict=True)
                model.release_clamp()
                if is_cnn:
                    zero_inp = torch.zeros(batch_size, features.shape[2], 1, conv_ph, device=model_device)
                    model.clamp(input_data=zero_inp)
                else:
                    model.clamp(input_data=features[:, 0, :].clone())
                window_idx, iter_in_phoneme = 0, 0
            else:
                window_idx = (i - 1) // steps_per_phoneme
                iter_in_phoneme = (i - 1) % steps_per_phoneme
                window_idx = min(window_idx, n_windows - 1)
                if is_cnn:
                    ph_start = window_idx * conv_ph
                    inp = features[:, ph_start:ph_start + conv_ph, :].permute(0, 2, 1).unsqueeze(2)
                else:
                    inp = features[:, window_idx, :]
                model.clamp(input_data=inp)
                _ = model.backward()
                _ = model.forward(inp, step=step_size)

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

            # Build phoneme string for title and marker.
            if word is not None:
                if is_cnn:
                    ph_start = window_idx * conv_ph
                    ph_str = word[ph_start:ph_start + conv_ph]
                    # Use the last phoneme in the window for the phoneme-layer marker.
                    ph_char = word[min(ph_start + conv_ph - 1, len(word) - 1)]
                else:
                    ph_str = word[window_idx] if window_idx < len(word) else ""
                    ph_char = word[window_idx] if window_idx < len(word) else "-"
                phoneme_idx_for_markers = PHONEME_TO_INDEX.get(ph_char, None)
            else:
                ph_str = ""
                phoneme_idx_for_markers = None

        update_images(layer_inds_to_plot, state_images, state, state_markers, phoneme_idx_for_markers)
        update_images(layer_inds_to_plot[:-1], recon_images, recon, recon_markers, phoneme_idx_for_markers)
        update_images(layer_inds_to_plot, err_images, err, err_markers, phoneme_idx_for_markers)
        _set_title(window_idx, ph_str, iter_in_phoneme)

        if progress is not None:
            progress.update(1)
            if i == n_frames - 1:
                progress.close()

        return state_images + recon_images + err_images + marker_artists

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
