## Headline (validation and test; test scored once per approach)

| approach | val acc | val macro F1 | test acc | test macro F1 | test femur side acc |
|---|---|---|---|---|---|
| Keras Model A (original) | 0.8907 | 0.8053 | 0.7339 | 0.6840 | 0.499 |
| CNN (PyTorch Model A) | 0.8785 | 0.7950 | 0.7416 | 0.6745 | 0.529 |
| CLIP zero-shot | 0.1154 | 0.0701 | 0.1126 | 0.0731 | 0.481 |
| CLIP + LoRA fine-tuned | 0.9225 | 0.8526 | 0.8014 | 0.7409 | 0.492 |

## Per-class test F1

| class | Keras Model A (original) | CNN (PyTorch Model A) | CLIP zero-shot | CLIP + LoRA fine-tuned |
|---|---|---|---|---|
| 0 bladder | 0.849 | 0.855 | 0.178 | 0.890 |
| 1 femur-left | 0.561 | 0.097 | 0.000 | 0.158 |
| 2 femur-right | 0.250 | 0.630 | 0.155 | 0.619 |
| 3 heart | 0.770 | 0.794 | 0.006 | 0.863 |
| 4 kidney-left | 0.511 | 0.525 | 0.064 | 0.517 |
| 5 kidney-right | 0.439 | 0.461 | 0.054 | 0.619 |
| 6 liver | 0.909 | 0.913 | 0.203 | 0.949 |
| 7 lung-left | 0.890 | 0.843 | 0.000 | 0.954 |
| 8 lung-right | 0.885 | 0.832 | 0.023 | 0.955 |
| 9 pancreas | 0.741 | 0.746 | 0.045 | 0.808 |
| 10 spleen | 0.718 | 0.724 | 0.075 | 0.816 |
| macro F1, femurs merged (10 classes) | 0.760 | 0.760 | 0.090 | 0.833 |

## Fine-tuned CLIP vs CNN on the same 8,827 test images

- McNemar (accuracy): CLIP right & CNN wrong = 1113, CNN right & CLIP wrong = 585, p = 4.81e-38

- Macro F1 difference (CLIP - CNN) = +0.0664, 95% bootstrap interval [+0.0553, +0.0778]

- Errors: CNN 2281, CLIP 1753, both wrong 1168 (51.2% of the CNN's errors), both wrong with the same answer 730

## Femur left/right across training seeds (validation)

| seed | approach | val acc | val macro F1 | class 1 F1 | class 2 F1 | femur detected | side accuracy | share of femur preds = left |
|---|---|---|---|---|---|---|---|---|
| 42 | Keras Model A (original) | 0.8907 | 0.8053 | 0.684 | 0.021 | 0.995 | 0.525 | 0.995 |
| 42 | CNN (PyTorch Model A) | 0.8785 | 0.7950 | 0.037 | 0.648 | 0.995 | 0.485 | 0.015 |
| 42 | CLIP + LoRA fine-tuned | 0.9225 | 0.8526 | 0.214 | 0.617 | 1.000 | 0.487 | 0.131 |
| 7 | Keras Model A (original) | 0.8850 | 0.7973 | 0.019 | 0.643 | 0.960 | 0.482 | 0.016 |
| 7 | CNN (PyTorch Model A) | 0.8801 | 0.8181 | 0.288 | 0.620 | 0.980 | 0.508 | 0.179 |
| 7 | CLIP + LoRA fine-tuned | 0.9299 | 0.8764 | 0.598 | 0.446 | 1.000 | 0.538 | 0.643 |
| 123 | Keras Model A (original) | 0.8789 | 0.7973 | 0.669 | 0.020 | 1.000 | 0.513 | 0.980 |
| 123 | CNN (PyTorch Model A) | 0.8707 | 0.7940 | 0.118 | 0.631 | 0.995 | 0.480 | 0.071 |
| 2024 | Keras Model A (original) | 0.8874 | 0.8169 | 0.690 | 0.170 | 0.985 | 0.556 | 0.944 |
| 2024 | CNN (PyTorch Model A) | 0.8895 | 0.8025 | 0.019 | 0.642 | 0.990 | 0.482 | 0.005 |
| - | CLIP zero-shot | 0.1154 | 0.0701 | 0.000 | 0.028 | 0.035 | 0.571 | 0.000 |

| approach | seeds | mean val macro F1 | min | max | mean femur side accuracy |
|---|---|---|---|---|---|
| Keras Model A (original) | 4 | 0.8042 | 0.7973 | 0.8169 | 0.519 |
| CNN (PyTorch Model A) | 4 | 0.8024 | 0.7940 | 0.8181 | 0.489 |
| CLIP + LoRA fine-tuned | 2 | 0.8645 | 0.8526 | 0.8764 | 0.513 |

## How the models split probability on true femur images (test)

| approach | mean P(femur-left) | mean P(femur-right) | mean P(either femur) | confidence when side right | confidence when side wrong | confidence on all correct predictions |
|---|---|---|---|---|---|---|
| Keras Model A (original) | 0.449 | 0.412 | 0.861 | 0.549 | 0.545 | 0.899 |
| CNN (PyTorch Model A) | 0.366 | 0.478 | 0.844 | 0.537 | 0.532 | 0.877 |
| CLIP + LoRA fine-tuned | 0.435 | 0.523 | 0.957 | 0.545 | 0.546 | 0.941 |

## Model-free check: 1-nearest-neighbour on raw pixels, left vs right (validation)

| organ pair | val images | side accuracy (0.5 = no information) |
|---|---|---|
| femur-left vs femur-right | 199 | 0.503 |
| kidney-left vs kidney-right | 299 | 0.652 |
| lung-left vs lung-right | 536 | 0.873 |

## Learning curve (validation macro F1, seed 42, same subsets for both)

| fraction | train images | CNN F1 | CLIP+LoRA F1 | CNN acc | CLIP+LoRA acc | CNN train time | CLIP+LoRA train time |
|---|---|---|---|---|---|---|---|
| 5% | 697 | 0.6296 | 0.6435 | 0.7219 | 0.7451 | 8s | 186s |
| 10% | 1392 | 0.7027 | 0.7916 | 0.7871 | 0.8597 | 9s | 433s |
| 25% | 3484 | 0.7971 | 0.8390 | 0.8609 | 0.9001 | 20s | 706s |
| 50% | 6966 | 0.7911 | 0.7965 | 0.8670 | 0.8772 | 27s | 836s |
| 100% | 13932 | 0.7950 | 0.8526 | 0.8785 | 0.9225 | 50s | 2334s |

## Compute cost (seed 42, full training set, NVIDIA GeForce MX570)

| approach | parameters | trained parameters | epochs run | best epoch | train seconds | seconds / epoch | peak GPU MB | test inference seconds |
|---|---|---|---|---|---|---|---|---|
| CNN (PyTorch Model A) | 286,219 | 286,219 | 20 | 15 | 50 | 2.5 | 286 | 0.9 |
| CLIP zero-shot | 151,277,313 | 0 | - | - | 0 | - | - | 47.2 |
| CLIP + LoRA fine-tuned | 152,260,353 | 983,040 | 7 | 4 | 2334 | 333.4 | 1265 | 65.4 |

## Calibration (test)

| approach | accuracy | mean confidence | ECE | error-detection AUROC | acc. of 10% least confident | acc. of other 90% |
|---|---|---|---|---|---|---|
| Keras Model A (original) | 0.7339 | 0.8366 | 0.1027 | 0.8353 | 0.3624 | 0.7752 |
| CNN (PyTorch Model A) | 0.7416 | 0.8132 | 0.0717 | 0.8264 | 0.3601 | 0.7840 |
| CLIP zero-shot | 0.1126 | 0.2334 | 0.1208 | 0.4861 | 0.1121 | 0.1127 |
| CLIP + LoRA fine-tuned | 0.8014 | 0.9009 | 0.0995 | 0.8562 | 0.4213 | 0.8437 |

## LangChain wrapper check

- agreement with plain-PyTorch predictions: 100.00%, largest probability difference 0.0e+00, 69.2s for 8827 images
