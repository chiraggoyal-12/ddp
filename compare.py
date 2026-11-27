"""
compare.py - put the approaches side by side, using ONLY saved results.

No model is run here. Every number is read from the JSON / .npz files that the
training and scoring scripts wrote, so this file can be re-run as often as you
like without touching the test set again.

What it produces (in results/comparison/):
    comparison.json   every number below, machine-readable
    summary.md        the same numbers as Markdown tables (pasted into README.md)
    *.png             confusion matrices, femur seeds, learning curve, reliability diagrams

    python compare.py
"""
import os

import matplotlib

matplotlib.use("Agg")  # write PNG files, no window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

import common  # noqa: E402

OUT_DIR = os.path.join(common.RESULTS_DIR, "comparison")
CNN_DIR = os.path.join(common.RESULTS_DIR, "cnn")
FM_DIR = os.path.join(common.RESULTS_DIR, "foundation_model")
KERAS_DIR = os.path.join(common.RESULTS_DIR, "keras_reference")

SEEDS = [42, 7, 123, 2024]
FRACTIONS = [0.05, 0.1, 0.25, 0.5, 1.0]

# Chart colours: one fixed colour per approach, used in every figure.
COLORS = {
    "keras": "#898781",       # muted gray: the old reference, not a competitor
    "cnn": "#2a78d6",         # blue
    "zero_shot": "#eb6834",   # orange
    "finetuned": "#1baf7a",   # aqua
}
LABELS = {
    "keras": "Keras Model A (original)",
    "cnn": "CNN (PyTorch Model A)",
    "zero_shot": "CLIP zero-shot",
    "finetuned": "CLIP + LoRA fine-tuned",
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# ---------------------------------------------------------------- loading

def run_file(folder, prefix, seed, fraction):
    if fraction == 1.0:
        return os.path.join(folder, f"{prefix}_seed{seed}.json")
    return os.path.join(folder, f"{prefix}_seed{seed}_frac{fraction}.json")


def load_test_predictions():
    """Saved test-set probabilities of each approach (each was scored once)."""
    files = {
        "cnn": os.path.join(CNN_DIR, "test_probabilities.npz"),
        "zero_shot": os.path.join(FM_DIR, "zero_shot_test_probabilities.npz"),
        "finetuned": os.path.join(FM_DIR, "finetuned_test_probabilities.npz"),
    }
    probs = {}
    labels = None
    for name, path in files.items():
        data = np.load(path)
        probs[name] = data["probs"]
        if labels is None:
            labels = data["labels"]
        assert np.array_equal(labels, data["labels"]), "all approaches must use the same test labels"
    keras = np.load(os.path.join(KERAS_DIR, "keras_seed42_probabilities.npz"))
    probs["keras"] = keras["test_probs"]
    return probs, labels


# ---------------------------------------------------------------- statistics

def mcnemar_test(y_true, pred_a, pred_b):
    """Is the accuracy difference between two models on the SAME images real?

    Only images where exactly one model is right carry information. If the two
    models were equally good, those disagreements would split 50/50. The exact
    binomial test gives the probability of a split at least this lopsided by chance.
    """
    a_right = pred_a == y_true
    b_right = pred_b == y_true
    only_a = int((a_right & ~b_right).sum())
    only_b = int((~a_right & b_right).sum())
    p_value = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b > 0 else 1.0
    return {"only_first_right": only_a, "only_second_right": only_b, "p_value": float(p_value)}


def bootstrap_macro_f1_difference(y_true, pred_a, pred_b, n_rounds=1000, seed=0):
    """95% interval for macro-F1(a) - macro-F1(b), by resampling test images.

    Each round draws 8,827 test images with replacement (same draw for both
    models) and recomputes both F1 scores. The middle 95% of the differences is
    the interval. If it does not contain 0, the F1 difference is unlikely to be luck.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    differences = []
    for _ in range(n_rounds):
        idx = rng.integers(0, n, size=n)
        f1_a = f1_score(y_true[idx], pred_a[idx], average="macro", zero_division=0)
        f1_b = f1_score(y_true[idx], pred_b[idx], average="macro", zero_division=0)
        differences.append(f1_a - f1_b)
    differences = np.array(differences)
    return {
        "observed": float(f1_score(y_true, pred_a, average="macro", zero_division=0)
                          - f1_score(y_true, pred_b, average="macro", zero_division=0)),
        "ci_low": float(np.percentile(differences, 2.5)),
        "ci_high": float(np.percentile(differences, 97.5)),
    }


def macro_f1_with_femurs_merged(y_true, y_pred):
    """Macro F1 over 10 classes, treating femur-left and femur-right as one "femur" class.

    Shows how good each model is on the part of the task that the data can
    actually answer, once the left-vs-right femur question is taken out.
    """
    merged_true = np.where(y_true == 2, 1, y_true)
    merged_pred = np.where(y_pred == 2, 1, y_pred)
    return float(f1_score(merged_true, merged_pred, average="macro", zero_division=0))


def femur_probability_split(y_true, probs):
    """For true femur images, how does the model split its probability between the two sides?

    If the model has learned "this is a femur, but I can't tell which side", it
    will give roughly half to femur-left and half to femur-right, and its
    confidence will be about 0.5 whether the side it picks is right or wrong.
    """
    is_femur = (y_true == 1) | (y_true == 2)
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    detected = is_femur & ((predictions == 1) | (predictions == 2))
    right_side = detected & (predictions == y_true)
    wrong_side = detected & (predictions != y_true)
    return {
        "mean_prob_femur_left": float(probs[is_femur, 1].mean()),
        "mean_prob_femur_right": float(probs[is_femur, 2].mean()),
        "mean_prob_either_femur": float((probs[is_femur, 1] + probs[is_femur, 2]).mean()),
        "mean_confidence_right_side": float(confidence[right_side].mean()),
        "mean_confidence_wrong_side": float(confidence[wrong_side].mean()),
        "mean_confidence_all_correct_predictions": float(confidence[predictions == y_true].mean()),
    }


def prediction_overlap(y_true, pred_a, pred_b):
    """Do the two models get the SAME images wrong?"""
    a_wrong = pred_a != y_true
    b_wrong = pred_b != y_true
    both_wrong = a_wrong & b_wrong
    return {
        "a_errors": int(a_wrong.sum()),
        "b_errors": int(b_wrong.sum()),
        "both_wrong": int(both_wrong.sum()),
        "both_wrong_same_answer": int((both_wrong & (pred_a == pred_b)).sum()),
        "share_of_a_errors_also_wrong_in_b": float(both_wrong.sum() / max(1, a_wrong.sum())),
    }


def nearest_neighbour_side_probe(left_class, right_class):
    """Model-free test: do raw pixels carry ANY left/right information for this organ?

    For every validation image of the two classes, find the single most similar
    training image of those two classes (smallest pixel-by-pixel squared
    difference) and copy its label. If the left and right versions of an organ
    look the same, this guess is right about 50% of the time.
    Uses train + validation only; the test set is not touched.
    """
    train_x, train_y = common.load_split("train")
    val_x, val_y = common.load_split("val")
    keep_train = (train_y == left_class) | (train_y == right_class)
    keep_val = (val_y == left_class) | (val_y == right_class)
    reference = train_x[keep_train].reshape(-1, 64 * 64).astype(np.float32) / 255.0
    reference_labels = train_y[keep_train]
    queries = val_x[keep_val].reshape(-1, 64 * 64).astype(np.float32) / 255.0
    query_labels = val_y[keep_val]
    n_correct = 0
    for i in range(len(queries)):
        distances = ((reference - queries[i]) ** 2).sum(axis=1)
        nearest = int(np.argmin(distances))
        if reference_labels[nearest] == query_labels[i]:
            n_correct += 1
    return {"classes": [common.CLASS_NAMES[left_class], common.CLASS_NAMES[right_class]],
            "n_val_images": int(len(queries)), "side_accuracy": n_correct / len(queries)}


# ---------------------------------------------------------------- figures

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_confusion_matrices(cms, names, path):
    """Row-normalized test confusion matrices: each row shows where that class's images went."""
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    fig, axes = plt.subplots(1, len(names), figsize=(6.2 * len(names), 6.4), facecolor=SURFACE)
    short = ["blad", "fem-L", "fem-R", "heart", "kid-L", "kid-R", "liver", "lung-L", "lung-R", "panc", "splen"]
    for ax, name in zip(axes, names):
        cm = np.array(cms[name], dtype=float)
        row_share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(row_share, cmap=cmap, vmin=0, vmax=1)
        for i in range(common.NUM_CLASSES):
            for j in range(common.NUM_CLASSES):
                if row_share[i, j] >= 0.05:
                    color = "#ffffff" if row_share[i, j] > 0.55 else INK
                    ax.text(j, i, f"{row_share[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)
        # outline the femur-left / femur-right block
        ax.add_patch(plt.Rectangle((0.5, 0.5), 2, 2, fill=False, edgecolor="#e34948", linewidth=2))
        ax.set_xticks(range(common.NUM_CLASSES))
        ax.set_yticks(range(common.NUM_CLASSES))
        ax.set_xticklabels(short, rotation=60, fontsize=8, color=INK_SECONDARY)
        ax.set_yticklabels(short, fontsize=8, color=INK_SECONDARY)
        ax.set_xlabel("predicted", color=INK_SECONDARY)
        ax.set_ylabel("true", color=INK_SECONDARY)
        ax.set_title(LABELS[name], color=INK, fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Test-set confusion matrices (row = true class, share of that class; red box = femur L/R)",
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))  # leave room for the x-axis label and the title
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_femur_seeds(seed_rows, path):
    """Class 1 (femur-left) vs class 2 (femur-right) validation F1, one marker per seed.

    Hollow markers with a different shape per approach, so markers that land on
    the same spot (it happens: two seeds can make the same decisions) stay visible.
    Per-seed numbers are in summary.md, so the markers are not labelled one by one.
    """
    shapes = {"keras": "s", "cnn": "o", "finetuned": "^"}
    fig, ax = plt.subplots(figsize=(6.6, 5.8), facecolor=SURFACE)
    style_axes(ax)
    for name in ["keras", "cnn", "finetuned"]:
        xs = [row[name]["class1_f1"] for row in seed_rows if name in row]
        ys = [row[name]["class2_f1"] for row in seed_rows if name in row]
        ax.scatter(xs, ys, s=110, marker=shapes[name], facecolors="none", edgecolors=COLORS[name],
                   linewidths=2, label=f"{LABELS[name]} ({len(xs)} seeds)", zorder=3)
    ax.annotate("calls (almost) every femur \"right\"", (0.02, 0.70), fontsize=8, color=INK_SECONDARY)
    ax.annotate("calls (almost) every femur \"left\"", (0.50, 0.08), fontsize=8, color=INK_SECONDARY)
    ax.annotate("splits its guesses", (0.52, 0.52), fontsize=8, color=INK_SECONDARY)
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(-0.03, 1.0)
    ax.set_xlabel("femur-left (class 1) F1 on validation", color=INK_SECONDARY)
    ax.set_ylabel("femur-right (class 2) F1 on validation", color=INK_SECONDARY)
    ax.set_title("Femur left/right: each marker is one training seed", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_learning_curve(curve, zero_shot_val_f1, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor=SURFACE)
    style_axes(ax)
    for name in ["cnn", "finetuned"]:
        xs = [row["n_train_images"] for row in curve[name]]
        ys = [row["val_macro_f1"] for row in curve[name]]
        ax.plot(xs, ys, color=COLORS[name], linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=LABELS[name], zorder=3)
        # direct label at the end of each line: CLIP above its line, CNN below its line
        label_offset = (-8, 10) if name == "finetuned" else (-8, -18)
        ax.annotate(LABELS[name], (xs[-1], ys[-1]), textcoords="offset points", xytext=label_offset,
                    ha="right", fontsize=9, color=INK_SECONDARY)
    ax.axhline(zero_shot_val_f1, color=COLORS["zero_shot"], linewidth=2, linestyle="--", label=LABELS["zero_shot"])
    ax.set_xscale("log")
    ax.set_xlabel("training images used (log scale; same stratified subsets for both)", color=INK_SECONDARY)
    ax.set_ylabel("validation macro F1", color=INK_SECONDARY)
    ax.set_title("How much training data does each approach need?", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_reliability(calibrations, names, path):
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.6), facecolor=SURFACE)
    for ax, name in zip(axes, names):
        style_axes(ax)
        bins = calibrations[name]["reliability_bins"]
        centers = []
        accs = []
        widths = []
        for b in bins:
            if b["count"] > 0:
                centers.append((b["low"] + b["high"]) / 2)
                accs.append(b["accuracy"])
                widths.append((b["high"] - b["low"]) - 0.01)
        ax.bar(centers, accs, width=widths, color=COLORS[name], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.plot([0, 1], [0, 1], color=INK_SECONDARY, linewidth=1.2, linestyle="--", zorder=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("confidence (probability of predicted class)", color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel("accuracy in that confidence bin", color=INK_SECONDARY, fontsize=9)
        c = calibrations[name]
        ax.set_title(f"{LABELS[name]}\nECE {c['ece']:.3f}, error-detection AUROC {c['error_detection_auroc']:.3f}",
                     color=INK, fontsize=10)
    fig.suptitle("Test-set reliability (bars on the dashed line = confidence matches accuracy)",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- tables

def fmt(value, digits=4):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    keras_ref = common.load_json(os.path.join(KERAS_DIR, "keras_reference.json"))
    cnn_test = common.load_json(os.path.join(CNN_DIR, "final_test.json"))
    zero_shot = common.load_json(os.path.join(FM_DIR, "zero_shot.json"))
    ft_test = common.load_json(os.path.join(FM_DIR, "finetuned_final_test.json"))
    cnn_train = common.load_json(run_file(CNN_DIR, "train", 42, 1.0))
    ft_train = common.load_json(run_file(FM_DIR, "finetune", 42, 1.0))
    langchain = common.load_json(os.path.join(FM_DIR, "langchain_check.json"))
    probs, y_test = load_test_predictions()
    preds = {name: p.argmax(axis=1) for name, p in probs.items()}

    # 1. headline numbers (validation + test)
    val_metrics = {"keras": keras_ref["seed42_val"], "cnn": cnn_train["val"],
                   "zero_shot": zero_shot["val"], "finetuned": ft_train["val"]}
    test_metrics = {"keras": keras_ref["seed42_test"], "cnn": cnn_test["test"],
                    "zero_shot": zero_shot["test"], "finetuned": ft_test["test"]}
    order = ["keras", "cnn", "zero_shot", "finetuned"]
    headline_rows = []
    for name in order:
        v, t = val_metrics[name], test_metrics[name]
        headline_rows.append([LABELS[name], fmt(v["accuracy"]), fmt(v["macro_f1"]),
                              fmt(t["accuracy"]), fmt(t["macro_f1"]),
                              fmt(t["femur"]["side_accuracy"], 3)])

    # 1b. per-class test F1, and macro F1 with the two femur classes merged
    per_class_rows = []
    for c in range(common.NUM_CLASSES):
        row = [f"{c} {common.CLASS_NAMES[c]}"]
        for name in order:
            row.append(fmt(test_metrics[name]["per_class_f1"][c], 3))
        per_class_rows.append(row)
    merged_f1 = {}
    merged_row = ["macro F1, femurs merged (10 classes)"]
    for name in order:
        merged_f1[name] = macro_f1_with_femurs_merged(y_test, preds[name])
        merged_row.append(fmt(merged_f1[name], 3))
    per_class_rows.append(merged_row)

    # 2. is the difference real? (paired tests on the same 8,827 test images)
    significance = {
        "finetuned_vs_cnn": {
            "mcnemar_accuracy": mcnemar_test(y_test, preds["finetuned"], preds["cnn"]),
            "bootstrap_macro_f1": bootstrap_macro_f1_difference(y_test, preds["finetuned"], preds["cnn"]),
        },
        "finetuned_vs_zero_shot": {
            "mcnemar_accuracy": mcnemar_test(y_test, preds["finetuned"], preds["zero_shot"]),
            "bootstrap_macro_f1": bootstrap_macro_f1_difference(y_test, preds["finetuned"], preds["zero_shot"]),
        },
    }
    overlap = prediction_overlap(y_test, preds["cnn"], preds["finetuned"])
    femur_split = {}
    femur_split_rows = []
    for name in ["keras", "cnn", "finetuned"]:
        femur_split[name] = femur_probability_split(y_test, probs[name])
        f = femur_split[name]
        femur_split_rows.append([LABELS[name], fmt(f["mean_prob_femur_left"], 3), fmt(f["mean_prob_femur_right"], 3),
                                 fmt(f["mean_prob_either_femur"], 3), fmt(f["mean_confidence_right_side"], 3),
                                 fmt(f["mean_confidence_wrong_side"], 3),
                                 fmt(f["mean_confidence_all_correct_predictions"], 3)])

    # 3. femur bistability across seeds (validation only)
    seed_rows = []
    for seed in SEEDS:
        row = {"seed": seed}
        if seed == 42:
            keras_val = keras_ref["seed42_val"]
        else:
            keras_val = keras_ref["other_seeds_val_only"][str(seed)]["val"]
        sources = {"keras": keras_val}
        cnn_path = run_file(CNN_DIR, "train", seed, 1.0)
        if os.path.exists(cnn_path):
            sources["cnn"] = common.load_json(cnn_path)["val"]
        ft_path = run_file(FM_DIR, "finetune", seed, 1.0)
        if os.path.exists(ft_path):
            sources["finetuned"] = common.load_json(ft_path)["val"]
        for name, m in sources.items():
            row[name] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                         "class1_f1": m["per_class_f1"][1], "class2_f1": m["per_class_f1"][2],
                         "femur_detected_rate": m["femur"]["femur_detected_rate"],
                         "side_accuracy": m["femur"]["side_accuracy"],
                         "share_left": m["femur"]["share_of_femur_predictions_that_say_left"]}
        seed_rows.append(row)
    seed_table_rows = []
    for row in seed_rows:
        for name in ["keras", "cnn", "finetuned"]:
            if name in row:
                r = row[name]
                seed_table_rows.append([row["seed"], LABELS[name], fmt(r["accuracy"]), fmt(r["macro_f1"]),
                                        fmt(r["class1_f1"], 3), fmt(r["class2_f1"], 3),
                                        fmt(r["femur_detected_rate"], 3), fmt(r["side_accuracy"], 3),
                                        fmt(r["share_left"], 3)])
    # mean / min / max of validation macro F1 over the seeds, per approach
    seed_summary = {}
    seed_summary_rows = []
    for name in ["keras", "cnn", "finetuned"]:
        values = [row[name]["macro_f1"] for row in seed_rows if name in row]
        side_values = [row[name]["side_accuracy"] for row in seed_rows if name in row]
        seed_summary[name] = {"n_seeds": len(values), "mean": float(np.mean(values)),
                              "min": float(np.min(values)), "max": float(np.max(values)),
                              "mean_side_accuracy": float(np.mean(side_values))}
        seed_summary_rows.append([LABELS[name], len(values), fmt(np.mean(values)), fmt(np.min(values)),
                                  fmt(np.max(values)), fmt(np.mean(side_values), 3)])
    zs_femur = zero_shot["val"]["femur"]
    probes = [nearest_neighbour_side_probe(1, 2),   # femur
              nearest_neighbour_side_probe(4, 5),   # kidney (control)
              nearest_neighbour_side_probe(7, 8)]   # lung (control)
    probe_rows = [[" vs ".join(p["classes"]), p["n_val_images"], fmt(p["side_accuracy"], 3)] for p in probes]
    seed_table_rows.append(["-", LABELS["zero_shot"], fmt(zero_shot["val"]["accuracy"]),
                            fmt(zero_shot["val"]["macro_f1"]), fmt(zero_shot["val"]["per_class_f1"][1], 3),
                            fmt(zero_shot["val"]["per_class_f1"][2], 3), fmt(zs_femur["femur_detected_rate"], 3),
                            fmt(zs_femur["side_accuracy"], 3),
                            fmt(zs_femur["share_of_femur_predictions_that_say_left"], 3)])

    # 4. learning curve (validation only, seed 42, identical subsets for both)
    curve = {"cnn": [], "finetuned": []}
    for fraction in FRACTIONS:
        for name, folder, prefix in [("cnn", CNN_DIR, "train"), ("finetuned", FM_DIR, "finetune")]:
            path = run_file(folder, prefix, 42, fraction)
            if os.path.exists(path):
                r = common.load_json(path)
                curve[name].append({"fraction": fraction, "n_train_images": r["n_train_images"],
                                    "val_accuracy": r["val"]["accuracy"], "val_macro_f1": r["val"]["macro_f1"],
                                    "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                                    "train_seconds": r["train_seconds"],
                                    "stopped_at_epoch_cap": r.get("stopped_at_epoch_cap", False)})
    curve_rows = []
    for fraction in FRACTIONS:
        c = None
        f = None
        for row in curve["cnn"]:
            if row["fraction"] == fraction:
                c = row
        for row in curve["finetuned"]:
            if row["fraction"] == fraction:
                f = row
        curve_rows.append([f"{fraction:.0%}", c["n_train_images"] if c else "-",
                           fmt(c["val_macro_f1"]) if c else "-", fmt(f["val_macro_f1"]) if f else "-",
                           fmt(c["val_accuracy"]) if c else "-", fmt(f["val_accuracy"]) if f else "-",
                           f"{c['train_seconds']:.0f}s" if c else "-", f"{f['train_seconds']:.0f}s" if f else "-"])

    # 5. compute cost
    cost_rows = [
        [LABELS["cnn"], f"{cnn_train['n_params']:,}", f"{cnn_train['n_trainable_params']:,}",
         cnn_train["epochs_run"], cnn_train["best_epoch"], f"{cnn_train['train_seconds']:.0f}",
         f"{cnn_train['train_seconds'] / cnn_train['epochs_run']:.1f}", f"{cnn_train['peak_gpu_memory_mb']:.0f}",
         f"{cnn_test['test_inference_seconds']:.1f}"],
        [LABELS["zero_shot"], f"{zero_shot['n_params']:,}", "0", "-", "-", "0", "-", "-",
         f"{zero_shot['test_inference_seconds']:.1f}"],
        [LABELS["finetuned"], f"{ft_train['n_params']:,}", f"{ft_train['n_trainable_params']:,}",
         ft_train["epochs_run"], ft_train["best_epoch"], f"{ft_train['train_seconds']:.0f}",
         f"{ft_train['train_seconds'] / ft_train['epochs_run']:.1f}", f"{ft_train['peak_gpu_memory_mb']:.0f}",
         f"{ft_test['test_inference_seconds']:.1f}"],
    ]

    # 6. calibration (test)
    calibrations = {"keras": keras_ref["seed42_test_calibration"], "cnn": cnn_test["test_calibration"],
                    "zero_shot": zero_shot["test_calibration"], "finetuned": ft_test["test_calibration"]}
    calibration_rows = []
    for name in order:
        c = calibrations[name]
        calibration_rows.append([LABELS[name], fmt(c["accuracy"]), fmt(c["mean_confidence"]), fmt(c["ece"]),
                                 fmt(c["error_detection_auroc"]), fmt(c["accuracy_lowest_10pct_confidence"]),
                                 fmt(c["accuracy_other_90pct"])])

    # figures
    plot_confusion_matrices({n: test_metrics[n]["confusion_matrix"] for n in ["cnn", "zero_shot", "finetuned"]},
                            ["cnn", "zero_shot", "finetuned"], os.path.join(OUT_DIR, "test_confusion_matrices.png"))
    plot_femur_seeds(seed_rows, os.path.join(OUT_DIR, "femur_bistability_seeds.png"))
    plot_learning_curve(curve, zero_shot["val"]["macro_f1"], os.path.join(OUT_DIR, "learning_curve.png"))
    plot_reliability(calibrations, ["cnn", "zero_shot", "finetuned"], os.path.join(OUT_DIR, "reliability.png"))

    # write everything
    common.save_json(os.path.join(OUT_DIR, "comparison.json"), {
        "val": val_metrics, "test": test_metrics, "test_macro_f1_femurs_merged": merged_f1,
        "significance_test": significance,
        "error_overlap_cnn_vs_finetuned": overlap, "femur_seeds_val": seed_rows,
        "femur_probability_split_test": femur_split, "seed_summary_val": seed_summary, "nearest_neighbour_side_probe_val": probes,
        "learning_curve_val": curve, "calibration_test": calibrations, "langchain_check": langchain,
    })
    sig = significance["finetuned_vs_cnn"]
    sections = [
        "## Headline (validation and test; test scored once per approach)",
        markdown_table(["approach", "val acc", "val macro F1", "test acc", "test macro F1", "test femur side acc"],
                       headline_rows),
        "## Per-class test F1",
        markdown_table(["class"] + [LABELS[n] for n in order], per_class_rows),
        "## Fine-tuned CLIP vs CNN on the same 8,827 test images",
        f"- McNemar (accuracy): CLIP right & CNN wrong = {sig['mcnemar_accuracy']['only_first_right']}, "
        f"CNN right & CLIP wrong = {sig['mcnemar_accuracy']['only_second_right']}, "
        f"p = {sig['mcnemar_accuracy']['p_value']:.2e}",
        f"- Macro F1 difference (CLIP - CNN) = {sig['bootstrap_macro_f1']['observed']:+.4f}, "
        f"95% bootstrap interval [{sig['bootstrap_macro_f1']['ci_low']:+.4f}, "
        f"{sig['bootstrap_macro_f1']['ci_high']:+.4f}]",
        f"- Errors: CNN {overlap['a_errors']}, CLIP {overlap['b_errors']}, both wrong {overlap['both_wrong']} "
        f"({overlap['share_of_a_errors_also_wrong_in_b']:.1%} of the CNN's errors), "
        f"both wrong with the same answer {overlap['both_wrong_same_answer']}",
        "## Femur left/right across training seeds (validation)",
        markdown_table(["seed", "approach", "val acc", "val macro F1", "class 1 F1", "class 2 F1",
                        "femur detected", "side accuracy", "share of femur preds = left"], seed_table_rows),
        markdown_table(["approach", "seeds", "mean val macro F1", "min", "max", "mean femur side accuracy"],
                       seed_summary_rows),
        "## How the models split probability on true femur images (test)",
        markdown_table(["approach", "mean P(femur-left)", "mean P(femur-right)", "mean P(either femur)",
                        "confidence when side right", "confidence when side wrong",
                        "confidence on all correct predictions"], femur_split_rows),
        "## Model-free check: 1-nearest-neighbour on raw pixels, left vs right (validation)",
        markdown_table(["organ pair", "val images", "side accuracy (0.5 = no information)"], probe_rows),
        "## Learning curve (validation macro F1, seed 42, same subsets for both)",
        markdown_table(["fraction", "train images", "CNN F1", "CLIP+LoRA F1", "CNN acc", "CLIP+LoRA acc",
                        "CNN train time", "CLIP+LoRA train time"], curve_rows),
        "## Compute cost (seed 42, full training set, " + cnn_train["device"] + ")",
        markdown_table(["approach", "parameters", "trained parameters", "epochs run", "best epoch",
                        "train seconds", "seconds / epoch", "peak GPU MB", "test inference seconds"], cost_rows),
        "## Calibration (test)",
        markdown_table(["approach", "accuracy", "mean confidence", "ECE", "error-detection AUROC",
                        "acc. of 10% least confident", "acc. of other 90%"], calibration_rows),
        "## LangChain wrapper check",
        f"- agreement with plain-PyTorch predictions: {langchain['agreement_with_plain_pytorch']:.2%}, "
        f"largest probability difference {langchain['max_probability_difference']:.1e}, "
        f"{langchain['seconds']:.1f}s for {langchain['n_test_images']} images",
    ]
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(sections) + "\n")
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
