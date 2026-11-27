"""
cnn/score_test.py - score the FINAL PyTorch Model A on the test set, once.

The final model is chosen before this runs (seed 42, full training data, the
same choice Phase 1 made). Nothing about the model is changed after seeing
these numbers.

    python -m cnn.score_test --seed 42
"""
import argparse
import os
import time

import numpy as np
import torch

import common
from cnn.model_a import ModelA, images_to_tensor, use_full_float32
from cnn.train_cnn import CHECKPOINT_DIR, RESULTS_DIR, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_full_float32()

    model = ModelA(num_classes=common.NUM_CLASSES).to(device)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"cnn_seed{args.seed}.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_images, test_labels = common.load_split("test")
    x_test = images_to_tensor(test_images)
    y_test = torch.from_numpy(test_labels)

    start_time = time.time()
    _, test_probs = evaluate(model, x_test, y_test, device)
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - start_time

    test_metrics = common.compute_metrics(test_labels, test_probs.argmax(axis=1))
    common.print_metrics(f"PyTorch Model A seed {args.seed} - TEST", test_metrics)

    np.savez_compressed(os.path.join(RESULTS_DIR, "test_probabilities.npz"),
                        probs=test_probs.astype(np.float32), labels=test_labels)
    common.save_json(os.path.join(RESULTS_DIR, "final_test.json"), {
        "model": f"PyTorch Model A, seed {args.seed}, full training set",
        "checkpoint": os.path.relpath(checkpoint_path, common.PROJECT_DIR),
        "test_inference_seconds": seconds,
        "test": test_metrics,
        "test_calibration": common.calibration_summary(test_labels, test_probs),
    })


if __name__ == "__main__":
    main()
