"""Full predictive coding models."""

from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm

from layers import (
    ConvLayer,
    FcLayer,
    FlattenLayer,
    AvgPoolLayer,
    InputLayer,
    MiddleLayer,
    OutputLayer,
    cosine_accuracy,
)


class PCModel(torch.nn.Module):
    """A full predictive coding model.

    This class wraps around a list of ``torch.nn.Module`` objects that represent the
    different layers of the model. During the ``forward()`` and ``backward()`` passed,
    all modules will be called in sequence, propagating prediction error and
    reconstructions.

    There should be at least two layers: the input and output layers (of types
    ``InputLayer`` and ``OutputLayer`` respectively. Layers are accessible though
    ``model.layers.{name}`` attributes. By default, the first layer is called ``input``,
    the final layer is called ``output`` and all other layers are called ``hidden{i}``,
    but these names can be overridden by either providing a dictionary or a
    ``torch.nn.Sequential`` object.

    Parameters
    ----------
    layers : list of torch.nn.Module | dict of str:torch.nn.Module | torch.nn.Sequential
        The layers of the model. These should be predictive coding layers as defined in
        ``torch_predcoding.py``. When given a dictionary, the keys will be used as the
        names for the layers. When given a plain list, automatic names will be assigned.
    batch_size : int
        The batch size used during operation of the model.
    immediate : bool
        Whether to use "immediate" mode, which means that prediction error is propagated
        through all the layers in a single step. By default, prediction error is only
        propagated to the next layer.
    top_down : float | None
        Override the layers' setting of the amount of top-down prediction error to take
        into account when updating its state. By default, the layers' own value is used.
    leakage : float | None
        Override the layers' setting of the amount of "leakage" (pull towards zero) of
        the state. By default, the layers' own value is used.
    """

    def __init__(
        self, layers, batch_size=64, immediate=False, top_down=None, leakage=None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.immediate = immediate
        self.top_down = top_down
        self.leakage = leakage

        assert len(layers) >= 2, "The model must have at least 2 layers (input/output)"

        # Give layers names if needed.
        if isinstance(layers, (dict, OrderedDict)):
            names = list(layers.keys())
            layers = layers.values()
            layers = torch.nn.Sequential(OrderedDict(zip(names, layers)))
        elif not isinstance(layers, torch.nn.Sequential):
            assert all([isinstance(layer, torch.nn.Module) for layer in layers])
            names = (
                ["input"]
                + [f"hidden{i}" for i in range(1, len(layers) - 1)]
                + ["output"]
            )
            layers = torch.nn.Sequential(OrderedDict(zip(names, layers)))
        else:
            assert isinstance(layers, torch.nn.Sequential)

        # make sure there are input/output layers
        assert isinstance(layers[0], InputLayer)
        assert isinstance(layers[-1], OutputLayer)

        self.layers = layers
        self.eval()  # we're not learning weights

    def clamp(self, input_data=None, output_data=None):
        """Clamp input/output units unto a given state.

        Parameters
        ----------
        input_data: tensor (batch_size, n_in) | None
            The data to be clamped to the input layer. If left at ``None``, do not clamp
            the input layer to anything.
        output_data: tensor (batch_size, n_out) | None
            The data to be clamped to the output layer. If left at ``None``, do not
            clamp the output layer to anything.
        """
        if input_data is not None:
            self.layers[0].clamp(input_data)
        if output_data is not None:
            self.layers[-1].clamp(output_data)

    def release_clamp(self):
        """Release any clamps on the input and output units."""
        self.layers[0].release_clamp()
        self.layers[-1].release_clamp()

    def forward(
        self, input_data=None, step=0.1, top_down=None, leakage=None, immediate=None
    ):
        """Perform a forward pass throught the model.

        Parameters
        ----------
        input_data: tensor (batch_size, n_in) | None
            The data to be clamped to the input layer during the forward pass. If left
            at ``None``, do not clamp the input layer to anything.
        step : float
            Step size.
        top_down : float | None
            Amount of top-down error to take into account.
        leakage : float | None
            Amount of activity lost every step.
        immediate : bool
            Whether to do the forward pass in "immediate" mode.

        Returns
        -------
        output_data: tensor (batch_size, n_out)
            The state of the output units in the model.
        """
        if immediate is None:
            immediate = self.immediate
        if top_down is None:
            top_down = self.top_down
        if leakage is None:
            leakage = self.leakage
        if not isinstance(leakage, list):
            leakage = [leakage] * len(self.layers)
        if not isinstance(step, list):
            step = [step] * len(self.layers)
        if not isinstance(top_down, list):
            top_down = [top_down] * len(self.layers)
        bu_err = self.layers[0](
            input_data, step=step[0], top_down=top_down[0], immediate=immediate
        )
        for layer, this_step, this_leakage, this_top_down in zip(
            self.layers[1:-1], step[1:-1], leakage[1:-1], top_down[1:-1]
        ):
            bu_err = layer(
                bu_err,
                step=this_step,
                top_down=this_top_down,
                leakage=this_leakage,
                immediate=immediate,
            )
        output_data = self.layers[-1](bu_err, step=step[-1], immediate=immediate)
        return output_data

    def backward(self):
        """Perform a backward pass through the model.

        Returns
        -------
        reconstruction: tensor (batch_size, n_in)
            The reconstruction of the input units made by the upper layers.
        """
        losses = []
        reconstruction, loss = self.layers[-1].backward()
        losses.append(loss)
        for layer in self.layers[-2::-1]:
            reconstruction, layer_loss = layer.backward(reconstruction)
            if hasattr(layer_loss, "item"):
                losses.append(layer_loss.detach().item())
            else:
                losses.append(layer_loss)
            loss = loss + layer_loss
        return loss, losses[::-1]

    def reset(self, batch_size=None):
        """Reset the state of all units to small randomized values.

        Parameters
        ----------
        batch_size : int | None
            Optionally change the batch size used by the model.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        for layer in self.layers:
            layer.reset(batch_size)

    def detach(self):
        """Detach the hidden state of the model."""
        for layer in self.layers:
            layer.detach()

    

    @property
    def device(self):
        """Current device the model is loaded on."""
        return next(self.parameters()).device

    @property
    def input(self):
        """The state of the input layer (=first layer)."""
        return self.layers[0].state

    @property
    def output(self):
        """The state of the output layer (=final layer)."""
        return self.layers[-1].state


def run_model(model, *, data, n_iter, targets, vectors=None, step=0.05):
    """Run a model of a list of batches of data.

    The batches are fed into the model sequentially and the model is not reset between
    batches. This way, you can present a sequence of stimuli to the model.

    Parameters
    ----------
    model : PCModel
        The model to run.
    data : list of torch.Tensor, shape (n_stimuli, 1, 64, 224)
        The batches of data to use.
    n_iter : list of int
        For how many steps to run each batch.
    targets : torch.Tensor, shape (n_stimuli,)
        The identity (as an integer) of each stimulus in the final batch.
    vectors : list of torch.Tensor, shape (n_stimuli, n_features) | None
        For each batch, the semantic vector for each stimulus. Not required if the model
        does not contain a semantic layer.
    step : float
        The step size.

    Returns
    -------
    errors : torch.Tensor, shape (n_stimuli, n_layers, n_steps)
        For each layer in the model, the total prediction error over time.
    lex_acc : float
        The accuracy of the lexical representation at the final timestep.
    vec_acc : float
        The accuracy of the semantic representation at the final timestep.
        This is only returned if `vectors` was specified.

    Examples
    --------
    This performs a priming experiment with the model, where first a blank screen is
    presented for 50 iterations, followed by a prime word for 100 iterations, then a
    blank screen again for 50 iterations and finally the target word for 100 iterations.
    >>> run_model(
    ...     model,
    ...     data=[zeros, primes, zeros, targets],
    ...     n_iter=[50, 100, 50, 100],
    ...     target=target_indices,
    ...     vectors=target_vectors,
    ...     step=0.05,
    ... )
    """
    model.eval()
    if not isinstance(data, list):
        data = [data]
    if not isinstance(n_iter, list):
        n_iter = [n_iter]

    old_batch_size = model.batch_size
    batch_size = len(data[0])
    model.reset(batch_size)
    model.release_clamp()

    pbar = tqdm(total=np.sum(n_iter))
    errors = list()
    for batch_data, n_batch_iter in zip(data, n_iter):
        for _ in range(n_batch_iter):
            with torch.no_grad():
                model.backward()
                output = model(batch_data, step=step)
            e = []
            for layer in model.layers:
                if hasattr(layer, "pred_err") and not isinstance(layer, InputLayer):
                    e.append(
                        torch.sqrt(torch.square(layer.pred_err))
                        .mean(dim=tuple(range(1, layer.pred_err.ndim)))
                        .detach()
                        .cpu()
                    )
            errors.append(e)

            lex_output = model.layers.lexicon.state if "lexicon" in model.layers._modules else model.layers.output.state
            pbar.update(1)
    errors = np.array(errors).T

    model.reset(old_batch_size)
    pbar.close()

    lex_acc = torch.mean((targets == lex_output.argmax(dim=1)).float()).item()

    if vectors is not None:
        vec_acc = cosine_accuracy(
            output,
            targets,
            vectors,
        )[1]
        return errors, lex_acc, vec_acc
    else:
        return errors, lex_acc


def mnist(batch_size=None, state_dict=None):
    """Construct a model for classifying the MNIST dataset.

    This is the original LeNet-5 architecture as described in:

    Lecun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning
    applied to document recognition. Proceedings of the IEEE, 86(11), 2278–2324.
    https://doi.org/10.1109/5.726791

    Parameters
    ----------
    batch_size : int | None
        The number of stimuli per batch. Not needed when initializing from a state_dict.
    state_dict : dict | None
        State dict to initialize the model from. If not specified, you need to specify
        batch_size.
    """
    model = PCModel(
        dict(
            input=InputLayer(n_units=(1, 28, 28), batch_size=batch_size),
            conv1=ConvLayer(
                in_channels=1,
                out_channels=6,
                in_height=28,
                kernel_size=5,
                padding=2,
                stride=1,
                output_padding=0,
                batch_size=batch_size,
            ),
            pool1=AvgPoolLayer(kernel_size=2, batch_size=batch_size),
            conv2=ConvLayer(
                in_channels=6,
                out_channels=16,
                in_height=14,
                kernel_size=5,
                padding=0,
                stride=1,
                output_padding=0,
                batch_size=batch_size,
            ),
            pool2=AvgPoolLayer(kernel_size=2, batch_size=batch_size),
            flatten=FlattenLayer(input_shape=(16, 5, 5), batch_size=batch_size),
            fc1=MiddleLayer(
                16 * 5 * 5,
                120,
                batch_size=batch_size,
            ),
            fc2=MiddleLayer(120, 84, batch_size=batch_size),
            output=OutputLayer(84, 10, batch_size=batch_size),
        ),
        leakage=0.2,
    )

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model

def trace(num_words=None, max_word_length=None, batch_size=None, state_dict=None):
    """Construct a predictive coding TRACE-like model for phoneme recognition.
    
    Parameters
    ----------
    batch_size : int | None
        The number of stimuli per batch.
    num_words : int | None
        The number of words (output classes).
    """
    model = PCModel(
        dict(
            input = InputLayer(n_units=7, batch_size=batch_size),
            phoneme_layer = FcLayer(n_in=7, n_units=128, batch_size=batch_size),
            word_layer = FcLayer(n_in=128, n_units=num_words, batch_size=batch_size),
            output = OutputLayer(n_in=num_words, n_units=num_words, batch_size=batch_size),
        ),
        leakage=0,
        top_down=None,
    )

    if state_dict is not None:
        model.load_state_dict(state_dict)


    return model


def viswordrec_lex(n_classes=None, batch_size=None, state_dict=None):
    """Construct a model of visual word recoginition ending in a lexicon.

    First working version of the viswordrec model! Ends in a lexicon, rather than a
    semantic representation.

    Parameters
    ----------
    n_classes : int | None
        Number of classes, i.e. number of output units. Not needed when initializing
        from a state_dict.
    batch_size : int | None
        The number of stimuli per batch. Not needed when initializing from a state_dict.
    state_dict : dict | None
        State dict to initialize the model from. If not specified, you need to specify
        n_classes and batch_size.
    """
    if state_dict is not None:
        batch_size, n_classes = state_dict["layers.lexicon.state"].shape
    model = PCModel(
        dict(
            input=InputLayer(n_units=(1, 64, 224), batch_size=batch_size),
            conv1=ConvLayer(
                in_channels=1,
                out_channels=32,
                in_height=64,
                in_width=224,
                kernel_size=5,
                padding=2,
                stride=2,
                output_padding=1,
                batch_size=batch_size,
            ),
            conv2=ConvLayer(
                in_channels=32,
                out_channels=64,
                in_height=32,
                in_width=112,
                kernel_size=5,
                padding=2,
                stride=2,
                output_padding=1,
                batch_size=batch_size,
            ),
            conv3=ConvLayer(
                in_channels=64,
                out_channels=128,
                in_height=16,
                in_width=56,
                kernel_size=5,
                padding=2,
                stride=2,
                output_padding=1,
                batch_size=batch_size,
            ),
            conv4=ConvLayer(
                in_channels=128,
                out_channels=64,
                in_height=8,
                in_width=28,
                kernel_size=5,
                padding=2,
                stride=2,
                output_padding=1,
                batch_size=batch_size,
            ),
            flatten=FlattenLayer(input_shape=(64, 4, 14), batch_size=batch_size),
            fc=MiddleLayer(64 * 4 * 14, 1024, batch_size=batch_size),
            lexicon=OutputLayer(1024, n_classes, batch_size=batch_size),
        ),
        leakage=0.35,
    )

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model


def viswordrec_sem(vectors, batch_size=None, state_dict=None):
    """Construct a model of visual word recognition that ends with semanitcs.

    First somewhat working version of the viswordrec model that includes a semantic
    representation.

    Parameters
    ----------
    vectors : ndarray, shape (n_classes, embedding_size)
        The semantic embedding vectors.
    batch_size : int | None
        The number of stimuli per batch. Not needed when initializing from a state_dict.
    state_dict : dict | None
        State dict to initialize the model from. If not specified, you need to specify
        n_classes and batch_size.
    """
    if state_dict is not None:
        batch_size = state_dict["layers.semantics.state"].shape[0]
    model = PCModel(
        dict(
            input=InputLayer(n_units=(1, 64, 224), batch_size=batch_size),
            conv1=ConvLayer(
                in_channels=1,
                out_channels=32,
                in_height=64,
                in_width=224,
                kernel_size=5,
                padding=2,
                stride=2,
                output_padding=1,
                batch_size=batch_size,
            ),
            conv2=ConvLayer(
                in_channels=32,
                out_channels=64,
                in_height=32,
                in_width=112,
                kernel_size=5,
                padding=2,
                stride=2,
                output_padding=1,
                batch_size=batch_size,
            ),
            conv3=ConvLayer(
                in_channels=64,
                out_channels=128,
                in_height=16,
                in_width=56,
                kernel_size=5,
                padding=2,
                stride=2,
                output_padding=1,
                batch_size=batch_size,
            ),
            conv4=ConvLayer(
                in_channels=128,
                out_channels=64,
                in_height=8,
                in_width=28,
                kernel_size=5,
                padding=2,
                stride=2,
                output_padding=1,
                batch_size=batch_size,
            ),
            flatten=FlattenLayer(input_shape=(64, 4, 14), batch_size=batch_size),
            fc=MiddleLayer(64 * 4 * 14, 1024, batch_size=batch_size),
            lexicon=MiddleLayer(1024, vectors.shape[0], batch_size=batch_size),
            semantics=OutputLayer(
                vectors.shape[0],
                vectors.shape[1],
                batch_size=batch_size,
                use_weight_norm=True,
            ),
        ),
        leakage=0.35,
    )

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model
