"""
common.py - small helpers shared by every script in this project.

What lives here:
  * where the data and results folders are
  * the 11 class names
  * loading one split (train / val / test) from the .npz file
  * the evaluation metrics (accuracy, macro F1, confusion matrix, ...)
  * the calibration numbers and the femur left/right analysis used in the comparison
  * saving / loading JSON

Only numpy and scikit-learn are imported here (no PyTorch), so that the metric
code is exactly the same no matter which model produced the predictions.
"""
import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "organsmnist_64.npz")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Official OrganSMNIST label map, copied from the MedMNIST package (medmnist/info.py).
# Our .npz file's MD5 (53a6d115339d874c25e309a994ff46d3) is the exact checksum MedMNIST
# publishes for organsmnist_64.npz, so these names are confirmed, not guessed.
CLASS_NAMES = [
    "bladder",       # 0
    "femur-left",    # 1
    "femur-right",   # 2
    "heart",         # 3
    "kidney-left",   # 4
    "kidney-right",  # 5
    "liver",         # 6
    "lung-left",     # 7
    "lung-right",    # 8
    "pancreas",      # 9
    "spleen",        # 10
]
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------- data

def load_split(split):
    """Return (images, labels) for 'train', 'val' or 'test'.

    images: uint8 array of shape (N, 64, 64), values 0..255 (grayscale)
    labels: int64 array of shape (N,), values 0..10
    The file stores labels as shape (N, 1), so we flatten them.
    """
    data = np.load(DATA_PATH)  # allow_pickle is False by default, so this only reads plain arrays
    images = data[split + "_images"]
    labels = data[split + "_labels"].flatten().astype(np.int64)
    return images, labels


def stratified_subset(labels, fraction, seed):
    """Pick `fraction` of the indices of every class (at least 1 per class).

    Used for the learning-curve experiment ("how much training data does each
    approach need?"). Sampling per class keeps the class balance of the full
    training set, so a 10% subset is not accidentally missing a rare class.
    """
    rng = np.random.default_rng(seed)
    chosen = []
    for c in range(NUM_CLASSES):
        class_indices = np.where(labels == c)[0]
        n_keep = max(1, int(round(fraction * len(class_indices))))
        picked = rng.choice(class_indices, size=n_keep, replace=False)
        chosen.extend(picked.tolist())
    chosen.sort()
    return np.array(chosen, dtype=np.int64)


# ---------------------------------------------------------------- metrics

def compute_metrics(y_true, y_pred):
    """Accuracy, macro precision/recall/F1, per-class F1 and confusion matrix.

    Same sklearn calls (average="macro", zero_division=0) as the original Keras
    notebook, so the numbers are directly comparable with Phase 1.
    """
    all_labels = list(range(NUM_CLASSES))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=all_labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class_f1": [float(v) for v in per_f1],
        "per_class_recall": [float(v) for v in per_r],
        "confusion_matrix": cm.tolist(),  # rows = true class, columns = predicted class
        "femur": femur_side_analysis(y_true, y_pred),
    }


def femur_side_analysis(y_true, y_pred):
    """Split the class-1/class-2 problem into two simpler questions.

    Class 1 = femur-left, class 2 = femur-right. For every true femur image we ask:
      1. Did the model at least say "femur" (either side)?  -> femur_detected_rate
      2. Of those, did it pick the correct side?            -> side_accuracy
    A side_accuracy near 0.5 means the model is guessing the side, and the
    class 1/2 errors are a left-vs-right problem, not a "what organ is this" problem.
    """
    is_true_femur = (y_true == 1) | (y_true == 2)
    pred_is_femur = (y_pred == 1) | (y_pred == 2)
    n_true_femur = int(is_true_femur.sum())
    detected = is_true_femur & pred_is_femur
    n_detected = int(detected.sum())
    n_side_correct = int((detected & (y_pred == y_true)).sum())
    return {
        "n_true_femur": n_true_femur,
        "femur_detected_rate": n_detected / n_true_femur if n_true_femur else 0.0,
        "side_accuracy": n_side_correct / n_detected if n_detected else 0.0,
        "true_left_pred_right": int(((y_true == 1) & (y_pred == 2)).sum()),
        "true_right_pred_left": int(((y_true == 2) & (y_pred == 1)).sum()),
        "share_of_femur_predictions_that_say_left": (
            int(((y_pred == 1) & is_true_femur).sum()) / n_detected if n_detected else 0.0
        ),
    }


def calibration_summary(y_true, probs, n_bins=10):
    """How well does the model's confidence match how often it is right?

    confidence = the probability the model gave to the class it predicted.
    Returns three things:
      * ECE (expected calibration error): split predictions into 10 equal-width
        confidence bins, and in each bin compare average confidence with
        accuracy. ECE is the weighted average gap. 0 = perfectly calibrated.
      * error_detection_auroc: if you pick one right and one wrong prediction at
        random, how often does the right one have the higher confidence?
        0.5 = confidence tells you nothing, 1.0 = confidence perfectly separates
        right from wrong. This directly answers "does low confidence predict
        wrong answers?".
      * accuracy of the 10% least-confident vs. the other 90% of predictions.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = (predictions == y_true).astype(np.float64)

    bins = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidence >= low) & (confidence <= high)  # last bin includes 1.0
        else:
            in_bin = (confidence >= low) & (confidence < high)
        count = int(in_bin.sum())
        if count == 0:
            bins.append({"low": float(low), "high": float(high), "count": 0,
                         "mean_confidence": None, "accuracy": None})
            continue
        bin_conf = float(confidence[in_bin].mean())
        bin_acc = float(correct[in_bin].mean())
        ece += (count / len(y_true)) * abs(bin_acc - bin_conf)
        bins.append({"low": float(low), "high": float(high), "count": count,
                     "mean_confidence": bin_conf, "accuracy": bin_acc})

    # roc_auc_score needs both right and wrong answers to exist
    if 0 < correct.sum() < len(correct):
        auroc = float(roc_auc_score(correct, confidence))
    else:
        auroc = None

    cutoff = np.quantile(confidence, 0.10)
    low_conf = confidence <= cutoff
    return {
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
        "ece": float(ece),
        "error_detection_auroc": auroc,
        "accuracy_lowest_10pct_confidence": float(correct[low_conf].mean()),
        "accuracy_other_90pct": float(correct[~low_conf].mean()),
        "reliability_bins": bins,
    }


# ---------------------------------------------------------------- printing / saving

def print_metrics(title, metrics):
    print(f"--- {title} ---")
    print(f"  accuracy  {metrics['accuracy']:.4f}")
    print(f"  macro F1  {metrics['macro_f1']:.4f}   (precision {metrics['macro_precision']:.4f}, "
          f"recall {metrics['macro_recall']:.4f})")
    f1 = metrics["per_class_f1"]
    print("  per-class F1: " + "  ".join(f"{c}:{v:.2f}" for c, v in enumerate(f1)))
    femur = metrics["femur"]
    print(f"  femur: detected {femur['femur_detected_rate']:.3f}, side accuracy "
          f"{femur['side_accuracy']:.3f}, left->right {femur['true_left_pred_right']}, "
          f"right->left {femur['true_right_pred_left']}")


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"saved {os.path.relpath(path, PROJECT_DIR)}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
