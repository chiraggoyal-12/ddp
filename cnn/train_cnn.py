"""
cnn/train_cnn.py - train PyTorch Model A from scratch with the Phase 1 recipe.

Recipe (copied from the original Keras notebook, results/metrics/experiment_config.json):
    Adam, learning rate 1e-3, batch size 64, up to 30 epochs,
    sparse categorical cross-entropy, inputs scaled to [0, 1],
    early stopping on validation loss with patience 5, keep the best epoch.

This script never loads the test set. The test set is scored once, later,
by cnn/score_test.py.

Examples (run from the project folder):
    python -m cnn.train_cnn --seed 42
    python -m cnn.train_cnn --seed 42 --train-fraction 0.1     (learning-curve run)
"""
import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

import common
from cnn.model_a import ModelA, count_parameters, images_to_tensor, use_full_float32

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 5
# Keras' Adam uses epsilon=1e-7, PyTorch's default is 1e-8. Use the Keras value.
ADAM_EPSILON = 1e-7

CHECKPOINT_DIR = os.path.join(common.PROJECT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(common.RESULTS_DIR, "cnn")


def seed_everything(seed):
    """Fix every source of randomness so a rerun gives the same model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the GPU
    # cuDNN can pick different (non-deterministic) algorithms run to run; forbid that.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    use_full_float32()


def run_name(seed, train_fraction):
    if train_fraction == 1.0:
        return f"seed{seed}"
    return f"seed{seed}_frac{train_fraction}"


def evaluate(model, x, y, device, batch_size=512):
    """Return (mean loss, predicted probabilities) on a whole split, without training."""
    model.eval()  # turns dropout off
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    all_probs = []
    with torch.no_grad():  # no gradients needed, saves memory and time
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size].to(device)
            yb = y[start:start + batch_size].to(device)
            logits = model(xb)
            total_loss += loss_fn(logits, yb).item()
            all_probs.append(torch.softmax(logits, dim=1).cpu())
    probs = torch.cat(all_probs).numpy()
    return total_loss / len(x), probs


def train_one_epoch(model, optimizer, x, y, device, generator):
    model.train()  # turns dropout on
    loss_fn = nn.CrossEntropyLoss()
    # New random order every epoch, like Keras fit(shuffle=True).
    order = torch.randperm(len(x), generator=generator)
    total_loss = 0.0
    for start in range(0, len(x), BATCH_SIZE):
        batch_idx = order[start:start + BATCH_SIZE]
        xb = x[batch_idx].to(device)
        yb = y[batch_idx].to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_idx)
    return total_loss / len(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="use only this fraction of the training set (per class)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    name = run_name(args.seed, args.train_fraction)

    train_images, train_labels = common.load_split("train")
    val_images, val_labels = common.load_split("val")
    if args.train_fraction < 1.0:
        keep = common.stratified_subset(train_labels, args.train_fraction, seed=args.seed)
        train_images, train_labels = train_images[keep], train_labels[keep]
    print(f"run {name}: {len(train_labels)} training images, {len(val_labels)} validation images, device {device}")

    x_train = images_to_tensor(train_images)
    y_train = torch.from_numpy(train_labels)
    x_val = images_to_tensor(val_images)
    y_val = torch.from_numpy(val_labels)

    model = ModelA(num_classes=common.NUM_CLASSES).to(device)
    n_params = count_parameters(model)
    print(f"Model A parameters: {n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPSILON)
    shuffle_generator = torch.Generator().manual_seed(args.seed)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, optimizer, x_train, y_train, device, shuffle_generator)
        val_loss, val_probs = evaluate(model, x_val, y_val, device)
        val_acc = float((val_probs.argmax(axis=1) == val_labels).mean())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc})
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  val acc {val_acc:.4f}")

        # Early stopping on validation loss, exactly like Keras EarlyStopping(patience=5,
        # restore_best_weights=True): remember the best epoch, stop after 5 epochs without
        # a new best, and go back to the best weights at the end.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"early stop: no improvement for {PATIENCE} epochs")
                break

    train_seconds = time.time() - start_time
    peak_gpu_mb = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
    model.load_state_dict(best_weights)

    _, val_probs = evaluate(model, x_val, y_val, device)
    val_metrics = common.compute_metrics(val_labels, val_probs.argmax(axis=1))
    common.print_metrics(f"PyTorch Model A {name} - VAL (best epoch {best_epoch})", val_metrics)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"cnn_{name}.pt")
    torch.save(best_weights, checkpoint_path)
    print(f"saved {os.path.relpath(checkpoint_path, common.PROJECT_DIR)}")

    common.save_json(os.path.join(RESULTS_DIR, f"train_{name}.json"), {
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "n_train_images": int(len(train_labels)),
        "n_params": n_params,
        "n_trainable_params": n_params,
        "recipe": {"optimizer": "Adam", "learning_rate": LEARNING_RATE, "adam_epsilon": ADAM_EPSILON,
                   "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
                   "init": "Glorot-uniform weights, zero biases (Keras default)"},
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_seconds": train_seconds,
        "peak_gpu_memory_mb": peak_gpu_mb,
        "history": history,
        "val": val_metrics,
        "val_calibration": common.calibration_summary(val_labels, val_probs),
        "checkpoint": os.path.relpath(checkpoint_path, common.PROJECT_DIR),
    })


if __name__ == "__main__":
    main()
