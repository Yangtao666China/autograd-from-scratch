"""Train a two-moons classifier without PyTorch or downloaded datasets."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .tensor import Tensor


def moons(n, rng):
    angle = rng.uniform(0, np.pi, n)
    label = rng.integers(0, 2, n)
    x = np.column_stack((np.cos(angle), np.sin(angle)))
    x[label == 1] = np.column_stack((1-np.cos(angle[label == 1]),
                                    0.5-np.sin(angle[label == 1])))
    return x + rng.normal(0, 0.13, x.shape), label[:, None].astype(float)


def train(epochs=500, seed=42, output="artifacts"):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    rng = np.random.default_rng(seed)
    x_train, y_train = moons(256, rng)
    x_val, y_val = moons(128, rng)
    x_test, y_test = moons(256, rng)
    weights, biases = [], []
    for din, dout in [(2, 24), (24, 24), (24, 1)]:
        weights.append(Tensor(rng.normal(0, np.sqrt(1/din), (din, dout)), True))
        biases.append(Tensor(np.zeros((1, dout)), True))
    parameters = weights + biases

    def forward(x):
        h = Tensor(x)
        for i, (w, b) in enumerate(zip(weights, biases)):
            h = h @ w + b
            if i < len(weights)-1:
                h = h.tanh()
        return h

    def evaluate(x, y):
        logits = forward(x).data
        return float(np.mean(np.logaddexp(0, logits)-y*logits)), float(np.mean((logits > 0) == y))

    rows, best_loss, best = [], float("inf"), None
    for epoch in range(1, epochs+1):
        logits = forward(x_train)
        loss = (logits.softplus()-logits*y_train).mean()
        for p in parameters:
            p.zero_grad()
        loss.backward()
        for p in parameters:
            p.data -= 0.12*p.grad
        train_loss, train_acc = evaluate(x_train, y_train)
        val_loss, val_acc = evaluate(x_val, y_val)
        rows.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                         train_acc=train_acc, val_acc=val_acc))
        if val_loss < best_loss:
            best_loss = val_loss
            best = (epoch, [p.data.copy() for p in parameters])
    for p, data in zip(parameters, best[1]):
        p.data[...] = data
    test_loss, test_acc = evaluate(x_test, y_test)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metrics = dict(seed=seed, epochs=epochs, selected_epoch=best[0],
                   selection_metric="validation BCE", test_loss=test_loss,
                   test_accuracy=test_acc, train_samples=256, validation_samples=128,
                   test_samples=256, parameters=sum(p.data.size for p in parameters))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    np.savez(out / "model.npz", **{f"parameter_{i}":p.data for i,p in enumerate(parameters)})
    plt.rcParams.update({"font.family":"DejaVu Sans", "axes.spines.top":False,
                         "axes.spines.right":False, "figure.facecolor":"#f8fafc",
                         "axes.facecolor":"#f8fafc"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    grid_x, grid_y = np.meshgrid(np.linspace(-1.6, 2.6, 180), np.linspace(-1.1, 1.6, 140))
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    probability = np.exp(-np.logaddexp(0, -forward(grid).data)).reshape(grid_x.shape)
    axes[0].contourf(grid_x, grid_y, probability, levels=20, cmap="coolwarm", alpha=.65)
    axes[0].scatter(x_test[:,0], x_test[:,1], c=y_test[:,0], cmap="coolwarm", s=16,
                    edgecolors="white", linewidths=.4)
    axes[0].set(title=f"Held-out test accuracy: {test_acc:.1%}", xlabel="Feature 1", ylabel="Feature 2")
    axes[1].plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows], label="Train", color="#2563eb")
    axes[1].plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows], label="Validation", color="#f97316")
    axes[1].axvline(best[0], color="#94a3b8", linestyle="--", label="Selected checkpoint")
    axes[1].set(title="Learning through our own gradients", xlabel="Epoch", ylabel="Binary cross entropy")
    axes[1].legend()
    fig.savefig(out / "training.png", dpi=160)
    plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    print(json.dumps(train(args.epochs, args.seed, args.output), indent=2))


if __name__ == "__main__":
    main()
