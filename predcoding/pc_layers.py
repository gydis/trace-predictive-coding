import torch
import torch.nn as nn

def grad(f):
    def result(x):
        # make leaf variables out of the inputs
        x_ = x.detach().requires_grad_(True) 
        f(x_).backward()
        return x_.grad
    return result

class PCNLayer(nn.Module):
    def __init__(self,
        in_dim,
        out_dim,
        activation_fn=torch.relu,
    ):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.W)
        self.activation_fn = activation_fn
        self.activation_deriv = lambda x: (self.activation_fn(x) > 0).float() if activation_fn == torch.relu else grad(activation_fn)

    def forward(self, x_above):
        a = x_above @ self.W.T
        x_hat = self.activation_fn(a)
        return x_hat, a

class PredictiveCodingNetwork(nn.Module):
    def __init__(self,
        dims,
        output_dim
    ):
        super().__init__()
        self.dims = dims
        self.L = len(dims) - 1
        self.layers = nn.ModuleList([
            PCNLayer(in_dim=dims[l+1],
            out_dim=dims[l])
            for l in range(self.L)
        ])
        self.readout = nn.Linear(dims[-1], output_dim, bias=True)

    def init_latents(self, batch_size, device):
        return [
        torch.randn(batch_size, d, device=device,
        requires_grad=False)
        for d in self.dims[1:]
        ]

    def compute_errors(self, inputs_latents):
        errors, gain_modulated_errors = [], []
        for l, layer in enumerate(self.layers):
            x_hat, a = layer(inputs_latents[l + 1])
            err = inputs_latents[l] - x_hat
            gm_err = err * layer.activation_deriv(a)
            errors.append(err)
            gain_modulated_errors.append(gm_err)
        return errors, gain_modulated_errors