"""Torch modules to perform predictive coding.

A model can be assembled by stacking an ``InputLayer``, as many ``MiddleLayer``s as
needed (can be zero) and finally an ``OutputLayer``.

These module define both a ``forward`` and ``backward`` method. First, the ``forward``
methods should be called in sequence, followed by calling all the ``backward`` methods
in reverse sequence.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal


class PCLayer(nn.Module):
    """A prototype for a predictive-coding layer.

    This is the class all other predictive-coding layers inherit from. The default
    implementation just passes bottom-up errors forwards and reconstructions backwards.
    """

    def __init__(self, batch_size=1, top_down=None, leakage=None):
        super().__init__()
        self.clamped = False
        self.batch_size = batch_size
        self.top_down = top_down
        self.leakage = leakage

    def reset(self, batch_size=None):
        """Set the values of the units to their initial state.

        Parameters
        ----------
        batch_size : int | None
            Optionally you can change the batch size to use from now on.
        """
        if batch_size is not None:
            self.batch_size = batch_size

    def forward(self, bu_err, step=0.1, top_down=None, leakage=None, immediate=False):
        """Propagate prediction error forward.

        Parameters
        ----------
        bu_err : tensor (batch_size, ...)
            The bottom-up error computed in the previous layer.
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
        bu_err : tensor (batch_size, n_units)
            The bottom-up error that needs to propagate to the next layer.
        """
        return bu_err

    def backward(self, reconstruction) -> tuple[torch.Tensor, float]:
        """Back-propagate the reconstruction.

        Parameters
        ----------
        reconstruction : tensor (batch_size, ...)
            The reconstruction of the state of the units in this layer that was computed
            and then back-propagated from the next layer.

        Returns
        -------
        reconstruction : tensor (batch_size, ...)
            The reconstruction of the state of the units in the previous layer
            that needs to be back-propagated.
        """
        return reconstruction, 0

    def clamp(self, state):
        """Clamp the units to a predefined state.

        Parameters
        ----------
        state : tensor (batch_size, ...)
            The clamped state of the units.
        """
        self.clamped = True

    def release_clamp(self):
        """Release any clamped state from the units."""
        self.clamped = False

    def parameters_forward(self):
        """Return the parameters in the forward direction."""
        return []

    def parameters_backward(self):
        """Return the parameters in the backward direction."""
        return []


class ConvLayer(PCLayer):
    """A predictive-coding layer that performs 2d convolution.

    This layer propagates errors onward, and back-propagates reconstructions.

    Parameters
    ----------
    n_in_channels : int
        How many channels the data coming into the layer has.
    n_out_channels : int
        How many channels the data coming out of this layer has.
    kernel_size : int
        The size of the convolution kernel.
    in_width : int
        The width of the data coming into the layer. Height is assumed to be the same.
    batch_size : int
        The number of inputs we compute per batch.
    top_down : float
        Amount of top-down error to take into account.
    leakage : float
        Amount of activity lost every step.
    use_weight_norm : bool
        Whether to apply weight normalization to the forward and backward weights.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        in_height,
        in_width=None,
        padding=0,
        stride=1,
        dilation=1,
        output_padding=0,
        groups=1,
        bias=True,
        batch_size=1,
        top_down=1.0,
        leakage=0,
        noise=0,
        use_weight_norm=True,
    ):
        super().__init__(batch_size, top_down, leakage)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_height = in_height
        if in_width is None:
            in_width = in_height
        self.in_width = in_width
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.output_padding = output_padding
        self.out_height = (
            math.floor((self.in_height + (2 * padding) - kernel_size) / stride) + 1
        )
        self.out_width = (
            math.floor((self.in_width + (2 * padding) - kernel_size) / stride) + 1
        )
        self.shape = (batch_size, out_channels, self.out_height, self.out_width)
        self.n_units = torch.prod(torch.tensor(self.shape[1:])).item()
        self.noise = noise
        self.use_weight_norm = use_weight_norm

        self.register_buffer("state", torch.zeros(self.shape))
        self.register_buffer(
            "reconstruction",
            torch.zeros(self.shape),
        )
        self.register_buffer("td_err", torch.zeros(self.shape))
        self.register_buffer("bu_err", torch.zeros(self.shape))

        self.weight = nn.Parameter(
            torch.empty((out_channels, in_channels // groups, kernel_size, kernel_size))
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.bias = None
        self.groups = groups

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if bias:
            nn.init.constant_(self.bias, 0)
        self.weight_norm = nn.Parameter(
            torch.norm(self.weight.view(self.out_channels, -1), dim=1, keepdim=True)
        )

    def reset(self, batch_size=None):
        """Set the values of the units to their initial state.

        Parameters
        ----------
        batch_size : int | None
            Optionally you can change the batch size to use from now on.
        """
        if batch_size is not None:
            self.batch_size = batch_size
            self.shape = (
                batch_size,
                self.out_channels,
                self.out_height,
                self.out_width,
            )
        device = self.state.device
        with torch.no_grad():
            self.state = torch.zeros(self.shape, device=device)
            self.reconstruction = torch.zeros(self.shape, device=device)
            self.td_err = torch.zeros(self.shape, device=device)
            self.bu_err = torch.zeros(self.shape, device=device)

    def forward(self, bu_err, step=0.1, top_down=None, leakage=None, immediate=False):
        """Update state, propagate prediction error forward.

        Parameters
        ----------
        bu_err : tensor (batch_size, in_channels, in_width, in_width)
            The bottom-up error computed in the previous layer.
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
        bu_err : tensor (batch_size, out_channels, out_width, out_width)
            The bottom-up error that needs to propagate to the next layer.
        """
        self.bu_err = F.relu(self.state) - self.reconstruction
        self.td_err = -self.bu_err  # self.reconstruction - self.state
        if not self.clamped:
            if self.use_weight_norm:
                weight = weight_norm(self.weight, self.weight_norm)
            else:
                weight = self.weight
            bu_err = F.conv2d(
                bu_err,
                weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            self.pred_err = bu_err
            if top_down is None:
                top_down = self.top_down
            if top_down:
                pred_err = self.pred_err + top_down * self.td_err
            else:
                pred_err = self.pred_err
            # pred_err = pred_err / self.word_norm_fw.T
            if leakage is None:
                leakage = self.leakage
            if leakage:
                self.state = self.state + step * pred_err - leakage * step * self.state
            else:
                self.state = self.state + step * pred_err
            self.state = torch.clamp(self.state, min=-0.1, max=None)
        if immediate:
            return self.pred_err
        else:
            return self.bu_err

    def backward(self, reconstruction):
        """Back-propagate the reconstruction.

        Parameters
        ----------
        reconstruction : tensor (batch_size, out_channels, out_width, out_width)
            The reconstruction of the state of the units in this layer that was computed
            and then back-propagated from the next layer.

        Returns
        -------
        reconstruction : tensor (batch_size, in_channels, in_width, in_width)
            The reconstruction of the state of the units in the previous layer
            that needs to be back-propagated.
        """
        self.reconstruction = reconstruction
        if self.use_weight_norm:
            weight = weight_norm(self.weight, self.weight_norm)
        else:
            weight = self.weight
        reconstruction = F.conv_transpose2d(
            F.relu(self.state),
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            output_padding=self.output_padding,
        )
        if self.noise > 0:
            noise = Normal(torch.zeros_like(reconstruction), scale=1).rsample()
            reconstruction = reconstruction + self.noise * noise
        return F.relu(reconstruction), backward_loss(
            self.reconstruction, F.relu(self.state)
        )

    def clamp(self, state):
        """Clamp the units to a predefined state.

        Parameters
        ----------
        state : tensor (batch_size, out_channels, out_width, out_width)
            The clamped state of the units.
        """
        self.state = state
        self.clamped = True

    def parameters_forward(self):
        """Return the parameters in the forward direction."""
        return self.conv_forward.parameters()

    def parameters_backward(self):
        """Return the parameters in the backward direction."""
        return self.conv_backward.parameters()

    def extra_repr(self):
        """Get some additional information about this module.

        Returns
        -------
        out : str
            Some extra information about this module.
        """
        return f"shape={self.shape}"


class MiddleLayer(PCLayer):
    """A predictive-coding layer that is sandwiched between two other layers.

    This layer propagates errors onward, and back-propagates reconstructions.

    Parameters
    ----------
    n_in : int
        How many units in the previous layer, i.e. the number of incoming connections.
    n_units : int
        How many units in this layer.
    batch_size : int
        The number of inputs we compute per batch.
    top_down : float
        Amount of top-down error to take into account.
    leakage : float
        Amount of activity lost every step.
    use_weight_norm : bool
        Whether to apply weight normalization to the forward and backward weights.
    """

    def __init__(
        self,
        n_in,
        n_units,
        bias=True,
        batch_size=1,
        top_down=1.0,
        leakage=0,
        noise=0,
        use_weight_norm=True,
    ):
        super().__init__(batch_size)
        self.n_units = n_units
        self.n_in = n_in
        self.top_down = top_down
        self.leakage = leakage
        self.noise = noise
        self.use_weight_norm = use_weight_norm

        self.register_buffer("state", torch.zeros((self.batch_size, self.n_units)))
        self.register_buffer(
            "reconstruction",
            torch.zeros((self.batch_size, self.n_units)),
        )
        self.register_buffer("td_err", torch.zeros((self.batch_size, self.n_units)))
        self.register_buffer("bu_err", torch.zeros((self.batch_size, self.n_units)))

        self.weight = nn.Parameter(torch.empty(n_units, n_in))
        nn.init.kaiming_uniform_(self.weight, mode="fan_in", nonlinearity="relu")

        if bias:
            self.bias = nn.Parameter(torch.empty(n_in))
            nn.init.constant_(self.bias, 0)
        else:
            self.bias = None
        self.weight_norm = nn.Parameter(torch.norm(self.weight, dim=1, keepdim=True))
        self.sparse_norm = nn.Parameter(-4.0 * torch.ones([1, 1]))

    def reset(self, batch_size=None):
        """Set the values of the units to their initial state.

        Parameters
        ----------
        batch_size : int | None
            Optionally you can change the batch size to use from now on.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        device = self.state.device
        with torch.no_grad():
            self.state = torch.zeros((self.batch_size, self.n_units), device=device)
            self.reconstruction = torch.zeros(
                (self.batch_size, self.n_units), device=device
            )
            self.td_err = torch.zeros((self.batch_size, self.n_units), device=device)
            self.bu_err = torch.zeros((self.batch_size, self.n_units), device=device)

    def forward(self, bu_err, step=0.1, top_down=None, leakage=None, immediate=False):
        """Update state, propagate prediction error forward.

        Parameters
        ----------
        bu_err : tensor (batch_size, n_in)
            The bottom-up error computed in the previous layer.
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
        bu_err : tensor (batch_size, n_units)
            The bottom-up error that needs to propagate to the next layer.
        """
        self.bu_err = F.relu(self.state) - self.reconstruction
        self.td_err = -self.bu_err  # self.reconstruction - self.state

        if not self.clamped:
            if self.use_weight_norm:
                weight = weight_norm(self.weight, self.weight_norm)
            else:
                weight = self.weight
            bu_err = F.linear(bu_err, weight)
            self.pred_err = bu_err
            if top_down is None:
                top_down = self.top_down
            if top_down:
                pred_err = self.pred_err + top_down * self.td_err
            else:
                pred_err = self.pred_err
            # pred_err = pred_err / self.word_norm_fw
            if leakage is None:
                leakage = self.leakage
            if leakage:
                self.state = self.state + step * pred_err - leakage * step * self.state
            else:
                self.state = self.state + step * pred_err
            self.state = torch.clamp(self.state, min=-0.1, max=None)
        if immediate:
            return self.pred_err
        else:
            return self.bu_err

    def backward(self, reconstruction):
        """Back-propagate the reconstruction.

        Parameters
        ----------
        reconstruction : tensor (bathc_size, n_units)
            The reconstruction of the state of the units in this layer that was computed
            and then back-propagated from the next layer.

        Returns
        -------
        reconstruction : tensor (batch_size, n_in)
            The reconstruction of the state of the units in the previous layer
            that needs to be back-propagated.
        """
        self.reconstruction = reconstruction
        weight = self.weight
        if self.use_weight_norm:
            weight = weight_norm(weight, self.weight_norm)
        weight = sparse_norm(weight, self.sparse_norm)
        reconstruction = F.linear(F.relu(self.state), weight.T, bias=self.bias)
        if self.noise > 0:
            noise = Normal(torch.zeros_like(reconstruction), scale=1).rsample()
            reconstruction = reconstruction + self.noise * noise
        return F.relu(reconstruction), backward_loss(
            self.reconstruction, F.relu(self.state)
        )

    def clamp(self, state):
        """Clamp the units to a predefined state.

        Parameters
        ----------
        state : tensor (batch_size, n_units)
            The clamped state of the units.
        """
        self.state = state
        self.clamped = True

    def release_clamp(self):
        """Release any clamped state from the units."""
        self.clamped = False

    def parameters_forward(self):
        """Return the parameters in the forward direction."""
        return self.linear_forward.parameters()

    def parameters_backward(self):
        """Return the parameters in the backward direction."""
        return self.linear_backward.parameters()

    def extra_repr(self):
        """Get some additional information about this module.

        Returns
        -------
        out : str
            Some extra information about this module.
        """
        return f"in_shape={self.n_in}, out_shape={self.n_units}"

class FcLayer(PCLayer):
    """A predictive-coding layer that is sandwiched between two other layers.

    This layer propagates errors onward, and back-propagates reconstructions.

    Parameters
    ----------
    n_in : int
        How many units in the previous layer, i.e. the number of incoming connections.
    n_units : int
        How many units in this layer.
    batch_size : int
        The number of inputs we compute per batch.
    top_down : float
        Amount of top-down error to take into account.
    leakage : float
        Amount of activity lost every step.
    use_weight_norm : bool
        Whether to apply weight normalization to the forward and backward weights.
    """

    def __init__(
        self,
        n_in,
        n_units,
        bias=True,
        batch_size=1,
        top_down=None,
        leakage=0,
        noise=0,
        use_weight_norm=False,
    ):
        super().__init__(batch_size)
        self.n_units = n_units
        self.n_in = n_in
        self.top_down = top_down
        self.leakage = leakage
        self.noise = noise
        self.use_weight_norm = use_weight_norm

        self.register_buffer("state", torch.zeros((self.batch_size, self.n_units)))
        self.register_buffer(
            "reconstruction",
            torch.zeros((self.batch_size, self.n_units)),
        )
        self.register_buffer("td_err", torch.zeros((self.batch_size, self.n_units)))
        self.register_buffer("bu_err", torch.zeros((self.batch_size, self.n_units)))

        self.weight = nn.Parameter(torch.empty(n_units, n_in))
        nn.init.kaiming_uniform_(self.weight, mode="fan_in", nonlinearity="relu")

        if bias:
            self.bias = nn.Parameter(torch.empty(n_in))
            nn.init.constant_(self.bias, 0)
        else:
            self.bias = None
        if self.use_weight_norm:
            self.weight_norm = nn.Parameter(
                torch.norm(self.weight, dim=1, keepdim=True)
            )
        self.sparse_norm = nn.Parameter(-4.0 * torch.ones([1, 1]))

    def reset(self, batch_size=None):
        """Set the values of the units to their initial state.

        Parameters
        ----------
        batch_size : int | None
            Optionally you can change the batch size to use from now on.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        device = self.state.device
        with torch.no_grad():
            self.state = torch.zeros((self.batch_size, self.n_units), device=device)
            self.reconstruction = torch.zeros(
                (self.batch_size, self.n_units), device=device
            )
            self.td_err = torch.zeros((self.batch_size, self.n_units), device=device)
            self.bu_err = torch.zeros((self.batch_size, self.n_units), device=device)

    def forward(self, bu_err, step=0.1, top_down=None, leakage=None, immediate=False):
        """Update state, propagate prediction error forward.

        Parameters
        ----------
        bu_err : tensor (batch_size, n_in)
            The bottom-up error computed in the previous layer.
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
        bu_err : tensor (batch_size, n_units)
            The bottom-up error that needs to propagate to the next layer.
        """
        self.bu_err = self.state - self.reconstruction
        self.td_err = -self.bu_err  # self.reconstruction - self.state

        if not self.clamped:
            weight = self.weight
            if self.use_weight_norm:
                weight = weight_norm(weight, self.weight_norm)
            bu_err = F.linear(bu_err, weight)
            self.pred_err = bu_err
            if top_down is None:
                top_down = self.top_down
            if top_down:
                pred_err = self.pred_err + top_down * self.td_err
            else:
                pred_err = self.pred_err
            # pred_err = pred_err / self.word_norm_fw
            if leakage is None:
                leakage = self.leakage
            if leakage:
                self.state = self.state + step * pred_err - leakage * step * self.state
            else:
                self.state = self.state + step * pred_err
            #self.state = torch.clamp(self.state, min=-0.1, max=None)
        if immediate:
            return self.pred_err
        else:
            return self.bu_err

    def backward(self, reconstruction):
        """Back-propagate the reconstruction.

        Parameters
        ----------
        reconstruction : tensor (bathc_size, n_units)
            The reconstruction of the state of the units in this layer that was computed
            and then back-propagated from the next layer.

        Returns
        -------
        reconstruction : tensor (batch_size, n_in)
            The reconstruction of the state of the units in the previous layer
            that needs to be back-propagated.
        loss : float
            The backward loss for this layer. The loss is an MSE between the reconstruction
            from the layer above and the state of the layer.
        """
        self.reconstruction = reconstruction
        weight = self.weight
        if self.use_weight_norm:
            weight = weight_norm(weight, self.weight_norm)
        reconstruction = F.relu(F.linear(self.state, weight.T, bias=self.bias))
        if self.noise > 0:
            noise = Normal(torch.zeros_like(reconstruction), scale=1).rsample()
            reconstruction = reconstruction + self.noise * noise
        return reconstruction, backward_loss(
            self.reconstruction, self.state
        )

    def clamp(self, state):
        """Clamp the units to a predefined state.

        Parameters
        ----------
        state : tensor (batch_size, n_units)
            The clamped state of the units.
        """
        self.state = state
        self.clamped = True

    def release_clamp(self):
        """Release any clamped state from the units."""
        self.clamped = False

    # def parameters_forward(self):
    #     """Return the parameters in the forward direction."""
    #     return self.linear_forward.parameters()

    # def parameters_backward(self):
    #     """Return the parameters in the backward direction."""
    #     return self.linear_backward.parameters()

    def extra_repr(self):
        """Get some additional information about this module.

        Returns
        -------
        out : str
            Some extra information about this module.
        """
        return f"in_shape={self.n_in}, out_shape={self.n_units}"


class InputLayer(PCLayer):
    """A predictive-coding layer that is at the bottom of the stack.

    This layer propagates errors onward, but does not compute reconstructions.

    Parameters
    ----------
    n_units : int | tuple of int
        How many units in this layer. This can be a tuple to specify the amount of units
        in more than one dimension.
    top_down : float | None
        Amount of top-down error to take into account.
    batch_size : int
        The number of inputs we compute per batch.
    """

    def __init__(self, n_units, top_down=1.0, batch_size=1):
        super().__init__(batch_size)
        if isinstance(n_units, int):
            self.shape = (batch_size, n_units)
        elif isinstance(n_units, tuple):
            self.shape = (batch_size,) + n_units
        else:
            self.shape = (batch_size,) + tuple(n_units)
        self.n_units = torch.prod(torch.tensor(self.shape[1:])).item()
        self.top_down = top_down

        self.register_buffer("state", torch.zeros(self.shape))
        self.register_buffer("reconstruction", torch.zeros(self.shape))
        self.register_buffer("td_err", torch.zeros(self.shape))

    def reset(self, batch_size=None):
        """Set the values of the units to their initial state.

        Parameters
        ----------
        batch_size : int | None
            Optionally you can change the batch size to use from now on.
        """
        if batch_size is not None:
            self.batch_size = batch_size
            self.shape = (batch_size,) + self.shape[1:]
        device = self.state.device
        with torch.no_grad():
            self.state = torch.zeros(self.shape, device=device)
            self.reconstruction = torch.zeros(self.shape, device=device)
            self.td_err = torch.zeros(self.shape, device=device)

    def forward(self, x=None, step=0.1, top_down=None, leakage=None, immediate=False):
        """Update state, propagate prediction error forward.

        Parameters
        ----------
        x : tensor (batch_size, n_units) | None
            The input given to the model. This will be the new state of the
            units in this layer. Set this to ``None`` to indicate there is no input and,
            unless the units are clamped, the state of the units should be affected only
            by top-down error.
        step : float
            Step size.
        leakage : float | None
            Amount of activity lost every step.
        top_down : float | None
            Amount of top-down error to take into account.
        immediate : bool
            Whether to do the forward pass in "immediate" mode.

        Returns
        -------
        bu_err : tensor (batch_size, n_units)
            The bottom-up error that needs to propagate to the next layer.
        """
        if not self.clamped:
            if x is not None:
                self.state = x
            else:
                if top_down is None:
                    top_down = self.top_down
                if top_down:
                    self.state = self.state + step * top_down * self.td_err
        self.bu_err = self.state - self.reconstruction
        self.td_err = -self.bu_err
        self.pred_err = self.bu_err
        return self.bu_err

    def backward(self, reconstruction):
        """Take in a reconstruction for use in the next iteration.

        Parameters
        ----------
        reconstruction : tensor (batch_size, n_units)
            The reconstruction of the state of the units in this layer that was computed
            and then back-propagated from the next layer.
        
        Returns
        -------
        reconstruction : tensor (batch_size, n_units)
            The reconstruction of the state of the units in the layer above
        loss : float
            The backward loss for this layer. The loss is an MSE between the reconstruction
            from the layer above and the state of the layer.
        """
        self.reconstruction = reconstruction
        return reconstruction, backward_loss(self.reconstruction, self.state)

    def clamp(self, state):
        """Clamp the units to a predefined state.

        Parameters
        ----------
        state : tensor (batch_size, n_units)
            The clamped state of the units.
        """
        self.state = state
        self.clamped = True

    def extra_repr(self):
        """Get some additional information about this module.

        Returns
        -------
        out : str
            Some extra information about this module.
        """
        return f"shape={self.shape}"


class OutputLayer(PCLayer):
    """A predictive-coding layer that is at the end of the stack.

    This layer back-propagates reconstructions, but does not propagate errors forward.

    Parameters
    ----------
    n_in : int
        How many units in the previous layer, i.e. the number of incoming connections.
    n_units : int
        How many units in this layer.
    batch_size : int
        The number of inputs we compute per batch.
    leakage : float
        Amount of activity lost every step.
    noise : float
        Amount of noise to add to the layer's activation
    weight_norm : bool
        Whether to use weight normalization.
    """

    def __init__(
        self,
        n_in,
        n_units,
        bias=True,
        batch_size=1,
        leakage=0,
        noise=0,
        use_weight_norm=True,
    ):
        super().__init__(batch_size)
        self.n_in = n_in
        self.n_units = n_units
        self.leakage = leakage
        self.noise = noise
        self.use_weight_norm = use_weight_norm

        self.register_buffer("state", torch.zeros((self.batch_size, self.n_units)))

        self.weight = nn.Parameter(torch.empty(n_units, n_in))
        nn.init.kaiming_uniform_(self.weight, mode="fan_in", nonlinearity="relu")

        if bias:
            self.bias = nn.Parameter(torch.empty(n_in))
            nn.init.constant_(self.bias, 0)
        else:
            self.bias = None

        if self.use_weight_norm:
            self.weight_norm = nn.Parameter(
                torch.norm(self.weight, dim=1, keepdim=True)
            )

    def reset(self, batch_size=None):
        """Set the values of the units to their initial state.

        Parameters
        ----------
        batch_size : int | None
            Optionally you can change the batch size to use from now on.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        device = self.state.device
        with torch.no_grad():
            self.state = torch.zeros((self.batch_size, self.n_units), device=device)

    def forward(self, bu_err, step=0.1, top_down=None, leakage=None, immediate=False):
        """Update state based on the bottom-up error propagated from the previous layer.

        Parameters
        ----------
        bu_err : tensor (batch_size, n_in)
            The bottom-up error computed in the previous layer.
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
        state : tensor (batch_size, n_units)
            The new state of the units in this layer. This is the output of the model.
        """
        if not self.clamped:
            if self.use_weight_norm:
                weight = weight_norm(self.weight, self.weight_norm)
            else:
                weight = self.weight
            bu_err = F.linear(bu_err, weight)
            self.pred_err = bu_err
            pred_err = bu_err
            if leakage is None:
                leakage = self.leakage
            if leakage:
                self.state = self.state + step * pred_err - leakage * step * self.state
            else:
                self.state = self.state + step * pred_err
            self.state = torch.clamp(self.state, min=-0.1, max=None)
        return self.state

    def backward(self):
        """Back-propagate the reconstruction.

        Returns
        -------
        reconstruction : tensor (n_in, batch_size)
            The reconstruction of the state of the units in the previous layer
            that needs to be back-propagated.
        """
        if self.use_weight_norm:
            weight = weight_norm(self.weight, self.weight_norm)
        else:
            weight = self.weight
        reconstruction = F.linear(F.relu(self.state), weight.T, bias=self.bias)
        if self.noise > 0:
            noise = Normal(torch.zeros_like(reconstruction), scale=1).rsample()
            reconstruction = reconstruction + self.noise * noise
        return F.relu(reconstruction), 0

    def clamp(self, state):
        """Clamp the units to a predefined state.

        Parameters
        ----------
        state : tensor (batch_size, n_units)
            The clamped state of the units.
        """
        self.state = state
        self.clamped = True

    def parameters_forward(self):
        """Return the parameters in the forward direction."""
        return self.linear_forward.parameters()

    def parameters_backward(self):
        """Return the parameters in the backward direction."""
        return self.linear_backward.parameters()

    def extra_repr(self):
        """Get some additional information about this module.

        Returns
        -------
        out : str
            Some extra information about this module.
        """
        return f"in_shape={self.n_in}, out_shape={self.n_units}"


class FlattenLayer(PCLayer):
    """A predictive-coding layer that performs a flattening operation.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input.
    batch_size : int
        The number of inputs we compute per batch.
    """

    def __init__(self, input_shape, batch_size=1):
        super().__init__(batch_size)
        self.input_shape = input_shape
        self.shape = (self.batch_size,) + input_shape

    def forward(self, bu_err, step=0.1, top_down=None, leakage=None, immediate=False):
        """Flatten and propagate prediction error forward.

        Parameters
        ----------
        bu_err : tensor (batch_size, input_shape)
            The bottom-up error computed in the previous layer.
        step : float
            The step size.
        top_down : float | None
            Amount of top-down error to take into account.
        leakage : float | None
            Amount of activity lost every step.
        immediate : bool
            Whether to do the forward pass in "immediate" mode.

        Returns
        -------
        bu_err : tensor (batch_size, -1)
            The bottom-up error that needs to propagate to the next layer.
        """
        return bu_err.view(self.batch_size, -1)

    def backward(self, reconstruction):
        """Un-flatten and back-propagate the reconstruction.

        Parameters
        ----------
        reconstruction : tensor (batch_size, -1)
            The reconstruction back-propagated from the next layer.

        Returns
        -------
        reconstruction : tensor (batch_size, input_shape)
            The reconstruction of the state of the units in the previous layer
            that needs to be back-propagated.
        """
        return reconstruction.view(self.shape), 0

    def reset(self, batch_size=None):
        """Set the values of the units to their initial state.

        Parameters
        ----------
        batch_size : int | None
            Optionally you can change the batch size to use from now on.

        """
        if batch_size is not None:
            self.batch_size = batch_size
            self.shape = (batch_size,) + self.input_shape

    def extra_repr(self):
        """Get some additional information about this module.

        Returns
        -------
        out : str
            Some extra information about this module.
        """
        return f"input_shape={self.input_shape}"


class AvgPoolLayer(PCLayer):
    """A predictive-coding layer that performs an avg-pool operation.

    Parameters
    ----------
    kernel_size : int
        How large the patch is that should be max-pooled over.
    batch_size : int
        The number of inputs we compute per batch.
    """

    def __init__(self, kernel_size, batch_size=1):
        super().__init__(batch_size)
        self.kernel_size = kernel_size

    def forward(self, bu_err, step=0.1, top_down=None, leakage=None, immediate=False):
        """Flatten and propagate prediction error forward.

        Parameters
        ----------
        bu_err : tensor (batch_size, channels, original_width, original_height)
            The bottom-up error computed in the previous layer.
        step : float
            The step size.
        top_down : float | None
            Amount of top-down error to take into account.
        leakage : float | None
            Amount of activity lost every step.
        immediate : bool
            Whether to do the forward pass in "immediate" mode.

        Returns
        -------
        bu_err : tensor (batch_size, channels, reduced_width, reduced_height)
            The bottom-up error that needs to propagate to the next layer.
        """
        bu_err = F.avg_pool2d(bu_err, self.kernel_size)
        return bu_err

    def backward(self, reconstruction):
        """Un-flatten and back-propagate the reconstruction.

        Parameters
        ----------
        reconstruction : tensor (batch_size, channels, reduced_width, reduced_height)
            The reconstruction back-propagated from the next layer.

        Returns
        -------
        reconstruction : tensor (batch_size, channels, original_width, original_height)
            The reconstruction of the state of the units in the previous layer
            that needs to be back-propagated.
        """
        return F.interpolate(reconstruction, scale_factor=self.kernel_size), 0

    def extra_repr(self):
        """Get some additional information about this module.

        Returns
        -------
        out : str
            Some extra information about this module.
        """
        return f"kernel_size={self.kernel_size}"


class AdaptivePoolLayer(PCLayer):
    """A predictive-coding layer that performs an avg-pool operation.

    Parameters
    ----------
    kernel_size : int
        How large the patch is that should be max-pooled over.
    batch_size : int
        The number of inputs we compute per batch.
    """

    def __init__(self, in_channels, kernel_size, batch_size=1):
        super().__init__(batch_size)
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.ones((in_channels, 1, kernel_size, kernel_size)) / in_channels
        )

    def forward(self, bu_err, step=0.1, top_down=None, leakage=None, immediate=False):
        """Flatten and propagate prediction error forward.

        Parameters
        ----------
        bu_err : tensor (batch_size, channels, original_width, original_height)
            The bottom-up error computed in the previous layer.
        step : float
            The step size.
        top_down : float | None
            Amount of top-down error to take into account.
        leakage : float | None
            Amount of activity lost every step.
        immediate : bool
            Whether to do the forward pass in "immediate" mode.

        Returns
        -------
        bu_err : tensor (batch_size, channels, reduced_width, reduced_height)
            The bottom-up error that needs to propagate to the next layer.
        """
        # bu_err = F.avg_pool2d(bu_err, self.kernel_size)
        bu_err = F.conv2d(
            bu_err,
            self.weight,
            stride=self.kernel_size,
            groups=self.in_channels,
        )
        return bu_err

    def backward(self, reconstruction):
        """Un-flatten and back-propagate the reconstruction.

        Parameters
        ----------
        reconstruction : tensor (batch_size, channels, reduced_width, reduced_height)
            The reconstruction back-propagated from the next layer.

        Returns
        -------
        reconstruction : tensor (batch_size, channels, original_width, original_height)
            The reconstruction of the state of the units in the previous layer
            that needs to be back-propagated.
        """
        # return F.interpolate(reconstruction, scale_factor=self.kernel_size), 0
        return F.conv_transpose2d(
            reconstruction,
            self.weight,
            stride=self.kernel_size,
            groups=self.in_channels,
        ), 0

    def extra_repr(self):
        """Get some additional information about this module.

        Returns
        -------
        out : str
            Some extra information about this module.
        """
        return f"kernel_size={self.kernel_size}"


class MaxPoolLayer(PCLayer):
    """A predictive-coding layer that performs a max-pool operation.

    Parameters
    ----------
    kernel_size : int
        How large the patch is that should be max-pooled over.
    batch_size : int
        The number of inputs we compute per batch.
    """

    def __init__(self, kernel_size, batch_size=1):
        super().__init__(batch_size)
        self.kernel_size = kernel_size

    def forward(self, bu_err, step=0.1, top_down=None, leakage=None, immediate=False):
        """Flatten and propagate prediction error forward.

        Parameters
        ----------
        bu_err : tensor (batch_size, channels, original_width, original_height)
            The bottom-up error computed in the previous layer.
        step : float
            The step size.
        top_down : float | None
            Amount of top-down error to take into account.
        leakage : float | None
            Amount of activity lost every step.
        immediate : bool
            Whether to do the forward pass in "immediate" mode.

        Returns
        -------
        bu_err : tensor (batch_size, channels, reduced_width, reduced_height)
            The bottom-up error that needs to propagate to the next layer.
        """
        bu_err = F.max_pool2d(bu_err, self.kernel_size)
        return bu_err

    def backward(self, reconstruction):
        """Un-flatten and back-propagate the reconstruction.

        Parameters
        ----------
        reconstruction : tensor (batch_size, channels, reduced_width, reduced_height)
            The reconstruction back-propagated from the next layer.

        Returns
        -------
        reconstruction : tensor (batch_size, channels, original_width, original_height)
            The reconstruction of the state of the units in the previous layer
            that needs to be back-propagated.
        """
        return F.interpolate(reconstruction, scale_factor=self.kernel_size), 0

    def extra_repr(self):
        """Get some additional information about this module.

        Returns
        -------
        out : str
            Some extra information about this module.
        """
        return f"kernel_size={self.kernel_size}"


def weight_norm(weight, norm):
    """Reparameterize the weight into size and direction."""
    orig_shape = weight.shape
    weight = weight.view(orig_shape[0], -1)
    weight = weight / torch.norm(weight, dim=1, keepdim=True)
    weight = norm * weight
    return weight.reshape(orig_shape)


def sparse_norm(weight, norm):
    """Reparameterize the weight for sparsity."""
    return torch.sign(weight) * F.relu(torch.abs(weight) - torch.sigmoid(norm))


def backward_loss(reconstruction, ground_truth) -> float:
    """Compute backward loss."""
    return F.mse_loss(reconstruction, ground_truth) / (
        ground_truth.square().mean() + 1e-6
    )


def cosine_accuracy(pred, target, vectors):
    """Compute cosine accuracy between predicted and target vectors.

    A word is recognized correctly when the predicted vector is closer (in terms of
    cosine distance) to the ground truth vector of the target word than that of any
    other word in the vocabulary.

    Parameters
    ----------
    pred : tensor, shape (batch_size, embedding_length)
        The predicted vectors for a batch of stimuli.
    target : tensor, shape (batch_size,)
        For each stimulus in the batch, an integer index indicating which word was
        written. Use this index with `vectors` to get the ground truth vectors.
    vectors : tensor, shape (vocab_size, embedding_length)
        The ground truth vectors for all the words in the vocabulary.

    Returns
    -------
    n_correct : int
        The number of stimuli in the batch that were correctly predicted.
    accuracy : float
        The percentage of stimuli in the batch that were correctly predicted.
    cosine_similarity : float
        The mean cosine similarity between the predicted and ground truth vectors.
    """
    vectors = torch.as_tensor(vectors).to(pred.device)

    # Normalize both predicted and target vectors
    pred_norm = F.normalize(F.relu(pred), p=2, dim=1)
    vectors_norm = F.normalize(vectors, p=2, dim=1)

    # Compute cosine similarity matrix: (batch_size, vocab_size)
    similarity = pred_norm @ vectors_norm.T

    # For each predicted vector, find the index of the most similar target vector
    predicted_indices = similarity.argmax(dim=1)

    # Accuracy: how many predicted_indices match the ground truth
    n_correct = (predicted_indices == target).sum().item()

    return (
        n_correct,
        n_correct / len(pred),
        similarity[(torch.arange(len(pred)).to(pred.device), target.to(pred.device))]
        .mean()
        .item(),
    )
