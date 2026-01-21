"""Train a predictive coding model on the MNIST dataset."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from viz import plot_prederr
from tqdm import trange

from layers import (
    AvgPoolLayer,
    ConvLayer,
    FlattenLayer,
    InputLayer,
    MiddleLayer,
    OutputLayer,
)
from models import mnist, run_model


def train(args, model, device, train_loader, epoch, optimizer):
    """Train the model."""
    step = 0.1
    n_zero_steps = 10
    n_steps = 20

    # Switch these off during training (should probably be in the modules themselves).
    old_leakage = model.leakage
    model.leakage = 0

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        zeros = torch.zeros_like(data)
        optimizer.zero_grad()

        model.reset()

        model.release_clamp()
        model.clamp(zeros, None)
        for _ in range(n_zero_steps):
            model.backward()
            model(zeros, step=step)
        loss_zero, _ = model.backward()

        model.release_clamp()
        model.clamp(data, None)
        for _ in range(n_steps):
            model.backward()
            model(data, step=step)
        loss_bw, losses = model.backward()
        output = model(data, step=step)
        loss_fw = F.cross_entropy(output, target)
        ratio = loss_fw / loss_bw
        loss = 10 * (0.1 * loss_zero + 0.1 * loss_bw + 0.8 * loss_fw)
        loss.backward()
        weights = [torch.linalg.norm(p.data).item() for p in model.parameters()]
        grads_fw = [
            torch.linalg.norm(p.grad).item()
            for p in model.parameters()
            if p.grad is not None
        ]
        optimizer.step()

        # Compute accuracy.
        pred = model.layers.output.state.argmax(
            dim=1, keepdim=True
        ).detach()  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100.0 * correct / len(data)

        # Print status update.
        if batch_idx % args.log_interval == 0:
            print(
                ("train epoch: {:02d} [{:05d}/{:05d} ({:.1f}%)]").format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                )
            )

            print(
                (
                    "    loss: {:.6f} (FW: {:.6f} BW: {:.6f} 0: {:.6f} ratio: {:.1f})"
                    "    accuracy: {:03d}/{:03d} ({:05.3f}%)"
                ).format(
                    loss.item(),
                    loss_fw.item(),
                    loss_bw.item(),
                    loss_zero.item(),
                    ratio.item(),
                    correct,
                    len(data),
                    accuracy,
                )
            )
            print(
                ("    BW loss: {}    ({:.6f})").format(
                    " ".join([f"{this_loss:.3f}" for this_loss in losses]),
                    np.sum(losses),
                )
            )
            print(
                ("      grads: {}    ({:.6f})").format(
                    " ".join([f"{this_grad:.3f}" for this_grad in grads_fw]),
                    np.sum(grads_fw),
                )
            )
            print(
                ("    weights: {}    ({:.6f})").format(
                    " ".join([f"{this_weight:.3f}" for this_weight in weights]),
                    np.sum(weights),
                )
            )
            print()
            if args.dry_run:
                break

    # Switch these parameters back on.
    model.leakage = old_leakage


def test(model, device, test_loader, n_iter=100, step=0.1, n_pre_iter=10):
    """Test the performance of the model."""
    model.eval()
    test_loss = np.zeros(n_iter // 10)
    correct = np.zeros(n_iter // 10, dtype="int")
    n = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        zeros = torch.zeros_like(data)
        n += len(data)
        with torch.no_grad():
            model.release_clamp()
            model.reset()
            for _ in range(n_pre_iter):
                model.backward()
                model(zeros, step=step)

            for i in range(n_iter):
                model.backward()
                model(data, step=step)
                if (i + 1) % 10 == 0:
                    test_loss[i // 10] += (
                        F.cross_entropy(
                            model.layers.output.state, target, reduction="sum"
                        )
                        .detach()
                        .item()
                    )
                    pred = model.layers.output.state.argmax(
                        dim=1, keepdim=True
                    ).detach()
                    correct[i // 10] += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= n

    print("Test set:")
    for i, (step_loss, step_correct) in enumerate(zip(test_loss, correct), 1):
        print(
            "step: {:03d}, loss: {:.4f}, accuracy: {}/{} ({:5.1f}%)".format(
                i * 10, step_loss, step_correct, n, 100 * step_correct / n
            )
        )
    return 100.0 * correct[-1] / n


# Training settings
parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--resume", type=str, default=None, help="Resume training of the given model."
)
parser.add_argument(
    "--test", default=None, help="Instead of training a new model, test the given model"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--start-epoch", type=int, default=1, help="Start at this epoch number"
)
parser.add_argument(
    "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--dry-run", action="store_true", default=False, help="quickly check a single pass"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {
    "num_workers": 0,
    "pin_memory": True,
    "shuffle": True,
    "drop_last": True,
    "batch_size": args.batch_size,
}

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

if args.resume:
    model = mnist(state_dict=torch.load(args.resume))
    print("Resumed", args.resume)
elif args.test:
    model = mnist(state_dict=torch.load(args.test))
    print("Testing", args.test)
else:
    model = mnist(batch_size=args.batch_size)
model = model.to(device)
print(model)

model.eval()
test(model, device, test_loader, n_pre_iter=20, n_iter=100, step=0.05)

# Split test data by digit.
digits = [[] for _ in range(10)]
for data, target in iter(test_loader):
    for i in range(10):
        digits[i].append(data[target == i])
for i in range(10):
    digits[i] = torch.vstack(digits[i])

# Create batches for the repetition priming experiment.
first_digit = digits[4][: args.batch_size].to(device)
repeated_digit = digits[4][args.batch_size : 2 * args.batch_size].to(device)
repeated_targets = torch.full((args.batch_size,), 4).to(device)
different_digit = digits[9][: args.batch_size].to(device)
different_targets = torch.full((args.batch_size,), 9).to(device)
zeros = torch.zeros_like(first_digit)

# Perform the repetition priming experiment.
err_repetition, acc = run_model(
    model,
    data=[zeros, first_digit, zeros, repeated_digit],
    n_iter=[50, 100, 50, 300],
    targets=repeated_targets,
    step=0.05,
)
print(f"Accuracy (repetition): {100 * acc}%")
err_different, acc = run_model(
    model,
    data=[zeros, first_digit, zeros, different_digit],
    n_iter=[50, 100, 50, 200],
    targets=different_targets,
    step=0.05,
)
print(f"Accuracy (different): {100 * acc}%")
plot_prederr(
    [err_repetition, err_different], mark=200, labels=["repetition", "different"]
)

##
# model.to("cpu")
# first_digit = digits[3][[0]].to("cpu")
# first_target = 3
# second_digit = digits[4][[0]].to("cpu")
# second_target = 4
# zeros = torch.zeros_like(first_digit)
#
# model.reset(batch_size=1)
# # scales = [1, 1.0, 0.5, 0.5, 0.5, 1.0]
# scales = [1, 1, 1, 1, 1, 1]
# viz(model, zeros, n_steps=500, step_size=0.03, scales=scales).save("zeros.mp4")
# viz(model, first_digit, n_steps=500, step_size=0.03, scales=scales).save(
#     "first_digit.mp4"
# )
# viz(model, second_digit, n_steps=500, step_size=0.03, scales=scales).save(
#     "second_digit.mp4"
# )
# model.reset(args.batch_size)
# model.to(device)
