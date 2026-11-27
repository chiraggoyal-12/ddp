"""
foundation_model/finetune_lora.py - fine-tune CLIP on our training set with LoRA.

LoRA in one paragraph: instead of changing CLIP's 151M original weights, we
freeze them all and add a small "correction" next to some weight matrices. For a
frozen matrix W (say 768 x 768), LoRA learns two thin matrices A (8 x 768) and
B (768 x 8) and uses W + (alpha / r) * B @ A instead of W. Only A and B are
trained, so under 1% of the parameters change (983,040 of 152M = 0.65%), which fits in a 2 GB GPU and
makes it hard to destroy what CLIP already knows. B starts at zero, so at
step 0 the model is exactly zero-shot CLIP.

Where LoRA goes: the query / key / value / output projections of every
attention layer, in BOTH the image encoder and the text encoder. So the model
learns to "see" CT slices differently and to "read" the 11 class sentences
differently.

Training is a plain PyTorch loop (no Trainer class), mirroring cnn/train_cnn.py:
shuffle, cross-entropy on the 11 class similarities, early stopping on
validation loss, keep the best epoch. The test set is never loaded here.

    python -m foundation_model.finetune_lora --seed 42
    python -m foundation_model.finetune_lora --seed 42 --train-fraction 0.1
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

import common
from foundation_model import clip_setup

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]  # attention layers, both encoders
LEARNING_RATE = 2e-4  # a common LoRA learning rate; kept constant, like the CNN's
# Batch 16, not 32: measured on this 2 GB GPU, batch 32 needs ~1.9 GB, and past the
# card's memory the driver silently moves data to system RAM and training crawls.
# Batch 16 peaks at ~1.26 GB. (16-bit "mixed precision" was also measured; it was
# not faster at batch 16 on this card, so plain float32 is kept for simplicity.)
BATCH_SIZE = 16
# Fewer epochs than the CNN's 30: a pre-trained model starts close to a good
# solution, and one CLIP epoch costs far more than one CNN epoch on this GPU.
MAX_EPOCHS = 10
PATIENCE = 3

CHECKPOINT_DIR = os.path.join(common.PROJECT_DIR, "checkpoints")


def seed_everything(seed):
    """Fix every source of randomness so a rerun gives the same model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the GPU (and LoRA's random init of A)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_name(seed, train_fraction):
    if train_fraction == 1.0:
        return f"seed{seed}"
    return f"seed{seed}_frac{train_fraction}"


def chosen_template():
    """Use the prompt template that zero_shot.py picked on the validation set."""
    zero_shot = common.load_json(os.path.join(clip_setup.results_dir(), "zero_shot.json"))
    return zero_shot["chosen_template"]


def add_lora(model):
    config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    # get_peft_model freezes every original weight and inserts the A/B matrices.
    return get_peft_model(model, config)


def validation_loss_and_probs(model, tokenizer, images, labels, template, device):
    probs = clip_setup.predict_probs(model, tokenizer, images, template, device)
    # cross-entropy = average of -log(probability given to the true class)
    true_class_probs = probs[np.arange(len(labels)), labels]
    loss = float(-np.log(np.clip(true_class_probs, 1e-12, 1.0)).mean())
    return loss, probs


def train_one_epoch(model, tokenizer, optimizer, images, labels, prompts, device, generator):
    model.train()  # turns LoRA dropout on
    order = torch.randperm(len(images), generator=generator).numpy()
    total_loss = 0.0
    for start in range(0, len(images), BATCH_SIZE):
        batch_idx = order[start:start + BATCH_SIZE]
        pixel_values = clip_setup.images_to_clip_input(images[batch_idx], device)
        targets = torch.from_numpy(labels[batch_idx]).to(device)
        # The text encoder has LoRA too, so the 11 class vectors change every step
        # and must be recomputed (cheap: only 11 short sentences).
        text_features = clip_setup.encode_text(model, tokenizer, prompts, device)
        logits = clip_setup.class_logits(model, pixel_values, text_features)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_idx)
    return total_loss / len(images)


def copy_trainable_weights(model):
    """Snapshot only the LoRA weights (the frozen ones never change)."""
    snapshot = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            snapshot[name] = p.detach().clone()
    return snapshot


def restore_trainable_weights(model, snapshot):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in snapshot:
                p.copy_(snapshot[name])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    name = run_name(args.seed, args.train_fraction)
    template = chosen_template()
    prompts = clip_setup.class_prompts(template)

    train_images, train_labels = common.load_split("train")
    val_images, val_labels = common.load_split("val")
    if args.train_fraction < 1.0:
        keep = common.stratified_subset(train_labels, args.train_fraction, seed=args.seed)
        train_images, train_labels = train_images[keep], train_labels[keep]
    print(f"run {name}: {len(train_labels)} training images, template {template!r}, device {device}")

    model, tokenizer = clip_setup.load_clip(device)
    model = add_lora(model)
    total_params, trainable_params = clip_setup.count_parameters(model)
    print(f"parameters: {total_params:,} total, {trainable_params:,} trainable "
          f"({100 * trainable_params / total_params:.2f}%)")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)
    shuffle_generator = torch.Generator().manual_seed(args.seed)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Epoch 0 = before any training (= zero-shot), useful to see the starting point.
    val_loss, val_probs = validation_loss_and_probs(model, tokenizer, val_images, val_labels, template, device)
    history = [{"epoch": 0, "train_loss": None, "val_loss": val_loss,
                "val_accuracy": float((val_probs.argmax(axis=1) == val_labels).mean()), "seconds": 0.0}]
    print(f"epoch  0  (no training)        val loss {val_loss:.4f}  val acc {history[0]['val_accuracy']:.4f}")

    best_val_loss = val_loss
    best_epoch = 0
    best_weights = copy_trainable_weights(model)
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, tokenizer, optimizer, train_images, train_labels,
                                     prompts, device, shuffle_generator)
        val_loss, val_probs = validation_loss_and_probs(model, tokenizer, val_images, val_labels, template, device)
        val_acc = float((val_probs.argmax(axis=1) == val_labels).mean())
        seconds = time.time() - epoch_start
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_accuracy": val_acc, "seconds": seconds})
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
              f"val acc {val_acc:.4f}  ({seconds:.0f}s)")

        # Same early-stopping rule as the CNN: keep the epoch with the lowest validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy_trainable_weights(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"early stop: no improvement for {PATIENCE} epochs")
                break

    train_seconds = time.time() - start_time
    peak_gpu_mb = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
    restore_trainable_weights(model, best_weights)

    _, val_probs = validation_loss_and_probs(model, tokenizer, val_images, val_labels, template, device)
    val_metrics = common.compute_metrics(val_labels, val_probs.argmax(axis=1))
    common.print_metrics(f"CLIP + LoRA {name} - VAL (best epoch {best_epoch})", val_metrics)

    adapter_dir = os.path.join(CHECKPOINT_DIR, f"clip_lora_{name}")
    model.save_pretrained(adapter_dir)  # saves only the small LoRA weights + their config
    print(f"saved {os.path.relpath(adapter_dir, common.PROJECT_DIR)}")

    common.save_json(os.path.join(clip_setup.results_dir(), f"finetune_{name}.json"), {
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "n_train_images": int(len(train_labels)),
        "model": clip_setup.MODEL_NAME,
        "template": template,
        "n_params": total_params,
        "n_trainable_params": trainable_params,
        "recipe": {"lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
                   "lora_target_modules": LORA_TARGET_MODULES, "optimizer": "AdamW (weight decay 0.01)",
                   "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
                   "max_epochs": args.max_epochs, "early_stopping_patience": PATIENCE},
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "epochs_run": len(history) - 1,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "stopped_at_epoch_cap": (len(history) - 1 == args.max_epochs and best_epoch == args.max_epochs),
        "train_seconds": train_seconds,
        "peak_gpu_memory_mb": peak_gpu_mb,
        "history": history,
        "val": val_metrics,
        "val_calibration": common.calibration_summary(val_labels, val_probs),
        "adapter_dir": os.path.relpath(adapter_dir, common.PROJECT_DIR),
    })


if __name__ == "__main__":
    main()
