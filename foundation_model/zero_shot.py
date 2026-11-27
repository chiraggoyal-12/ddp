"""
foundation_model/zero_shot.py - CLIP with NO training on our data.

This measures what the foundation model knows "out of the box" from its web
pre-training. It is the reference point for how much fine-tuning actually adds.

Steps:
  1. Try each prompt template on the VALIDATION set, keep the best (by macro F1).
     Choosing the prompt is the only decision made, and it uses validation data only.
  2. Score the TEST set once with that template.

    python -m foundation_model.zero_shot
"""
import os
import time

import numpy as np
import torch

import common
from foundation_model import clip_setup


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = clip_setup.load_clip(device)
    total_params, _ = clip_setup.count_parameters(model)
    print(f"{clip_setup.MODEL_NAME}: {total_params:,} parameters, device {device}")

    val_images, val_labels = common.load_split("val")

    # Step 1: pick the prompt template on validation data.
    template_results = []
    for template in clip_setup.PROMPT_TEMPLATES:
        probs = clip_setup.predict_probs(model, tokenizer, val_images, template, device)
        metrics = common.compute_metrics(val_labels, probs.argmax(axis=1))
        print(f"VAL  {template!r:50s} accuracy {metrics['accuracy']:.4f}  macro F1 {metrics['macro_f1']:.4f}")
        template_results.append({"template": template, "val": metrics})

    best = template_results[0]
    for result in template_results:
        if result["val"]["macro_f1"] > best["val"]["macro_f1"]:
            best = result
    best_template = best["template"]
    print(f"chosen template (best validation macro F1): {best_template!r}")
    val_probs = clip_setup.predict_probs(model, tokenizer, val_images, best_template, device)

    # Step 2: score the test set once with the chosen template.
    test_images, test_labels = common.load_split("test")
    start_time = time.time()
    test_probs = clip_setup.predict_probs(model, tokenizer, test_images, best_template, device)
    test_seconds = time.time() - start_time
    test_metrics = common.compute_metrics(test_labels, test_probs.argmax(axis=1))
    common.print_metrics("CLIP zero-shot - VAL", best["val"])
    common.print_metrics("CLIP zero-shot - TEST", test_metrics)

    out_dir = clip_setup.results_dir()
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "zero_shot_test_probabilities.npz"),
                        probs=test_probs.astype(np.float32), labels=test_labels)
    common.save_json(os.path.join(out_dir, "zero_shot.json"), {
        "model": clip_setup.MODEL_NAME,
        "n_params": total_params,
        "templates_tried_on_val": template_results,
        "chosen_template": best_template,
        "val": best["val"],
        "val_calibration": common.calibration_summary(val_labels, val_probs),
        "test": test_metrics,
        "test_calibration": common.calibration_summary(test_labels, test_probs),
        "test_inference_seconds": test_seconds,
    })


if __name__ == "__main__":
    main()
