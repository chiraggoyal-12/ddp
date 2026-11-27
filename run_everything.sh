#!/usr/bin/env bash
# run_everything.sh - re-create every result in results/, in order.
# Run from the project folder in Git Bash (Windows) or any bash shell, with the
# project venv's python. Total time on the MX570 laptop GPU: roughly 4-5 hours,
# almost all of it CLIP fine-tuning.
#
# The Keras reference (results/keras_reference/) is NOT re-created here: the
# Keras code and checkpoints were deleted on purpose, and those files are the
# saved record of what the original model scored.
set -e
PY=".venv/Scripts/python.exe"   # on Linux/macOS: .venv/bin/python
[ -x "$PY" ] || PY=".venv/bin/python"

# ---- Approach 1: CNN (PyTorch Model A), ~1 minute per run
for seed in 42 7 123 2024; do
    $PY -m cnn.train_cnn --seed $seed                       # validation only
done
for fraction in 0.05 0.1 0.25 0.5; do
    $PY -m cnn.train_cnn --seed 42 --train-fraction $fraction   # learning curve, validation only
done
$PY -m cnn.score_test --seed 42                              # the one test-set evaluation

# ---- Approach 2: CLIP zero-shot, then CLIP + LoRA fine-tuning
$PY -m foundation_model.zero_shot                            # picks the prompt on validation, scores test once
for seed in 42 7 123 2024; do
    $PY -m foundation_model.finetune_lora --seed $seed       # validation only
done
for fraction in 0.05 0.1 0.25 0.5; do
    $PY -m foundation_model.finetune_lora --seed 42 --train-fraction $fraction
done
$PY -m foundation_model.score_test --seed 42                 # the one test-set evaluation
$PY -m foundation_model.langchain_pipeline --seed 42         # same model through LangChain; must match

# ---- Side-by-side comparison (reads saved files only, runs no model)
$PY compare.py
