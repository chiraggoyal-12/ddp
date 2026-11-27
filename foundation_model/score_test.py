"""
foundation_model/score_test.py - score the FINAL fine-tuned CLIP on the test set, once.

The final model is chosen before this runs: seed 42, full training set, the
best-validation-loss epoch (the same rule as the CNN). Nothing is changed after
seeing these numbers.

    python -m foundation_model.score_test --seed 42
"""
import argparse
import os
import time

import numpy as np
import torch
from peft import PeftModel

import common
from foundation_model import clip_setup
from foundation_model.finetune_lora import CHECKPOINT_DIR, chosen_template


def load_finetuned_clip(seed, device):
    """Original CLIP weights + the saved LoRA weights on top."""
    base_model, tokenizer = clip_setup.load_clip(device)
    adapter_dir = os.path.join(CHECKPOINT_DIR, f"clip_lora_seed{seed}")
    model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
    model.eval()
    return model, tokenizer, adapter_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, adapter_dir = load_finetuned_clip(args.seed, device)
    template = chosen_template()
    test_images, test_labels = common.load_split("test")

    start_time = time.time()
    test_probs = clip_setup.predict_probs(model, tokenizer, test_images, template, device)
    seconds = time.time() - start_time

    test_metrics = common.compute_metrics(test_labels, test_probs.argmax(axis=1))
    common.print_metrics(f"CLIP + LoRA seed {args.seed} - TEST", test_metrics)

    out_dir = clip_setup.results_dir()
    np.savez_compressed(os.path.join(out_dir, "finetuned_test_probabilities.npz"),
                        probs=test_probs.astype(np.float32), labels=test_labels)
    common.save_json(os.path.join(out_dir, "finetuned_final_test.json"), {
        "model": f"{clip_setup.MODEL_NAME} + LoRA, seed {args.seed}, full training set",
        "adapter_dir": os.path.relpath(adapter_dir, common.PROJECT_DIR),
        "template": template,
        "test_inference_seconds": seconds,
        "test": test_metrics,
        "test_calibration": common.calibration_summary(test_labels, test_probs),
    })


if __name__ == "__main__":
    main()
