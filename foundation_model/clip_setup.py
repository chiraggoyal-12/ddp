"""
foundation_model/clip_setup.py - everything the CLIP scripts share.

CLIP in one paragraph: it has an IMAGE encoder (a Vision Transformer) and a TEXT
encoder (a Transformer). Both turn their input into a 512-number vector. During
pre-training (400M image-caption pairs from the web) it learned to make an image's
vector point in the same direction as the vector of a caption describing it.
To classify, we write one sentence per class ("a CT scan of the liver.", ...),
encode all 11, and pick the sentence whose vector is most similar to the image's.
No new output layer is added: the class names themselves, through the text
encoder, are the classifier. That is what makes it a vision-LANGUAGE model.

Model used: openai/clip-vit-base-patch32 (151M parameters). Larger CLIP variants
and generative vision-language models do not fit in this laptop's 2 GB GPU for
training; see README.md.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

import common

# The ~600 MB model is downloaded to the Hugging Face cache. To put it somewhere
# else (e.g. a bigger drive), set the HF_HOME environment variable before running.
from transformers import CLIPModel, CLIPTokenizer

MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_SIZE = 224  # CLIP was pre-trained on 224x224 images

# Plain-English organ names for the text prompts ("femur-left" -> "left femur").
# Same order as common.CLASS_NAMES, so prompt i describes class i.
PROMPT_NAMES = [
    "bladder", "left femur", "right femur", "heart", "left kidney", "right kidney",
    "liver", "left lung", "right lung", "pancreas", "spleen",
]

# Candidate sentence templates. zero_shot.py picks one using the VALIDATION set only;
# the fine-tuning then starts from that same template.
PROMPT_TEMPLATES = [
    "a CT scan of the {organ}.",
    "a sagittal CT slice showing the {organ}.",
    "a grayscale medical image of the {organ}.",
]

# CLIP's own normalization constants (from its preprocessing config).
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def load_clip(device):
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


def images_to_clip_input(images, device):
    """uint8 (N, 64, 64) grayscale -> float (N, 3, 224, 224), the way CLIP expects.

    1. scale to [0, 1]
    2. upsample 64 -> 224 with bicubic interpolation (what CLIP's own preprocessing uses)
    3. copy the single gray channel into R, G and B (CLIP only accepts colour images)
    4. normalize with CLIP's mean / std
    Done with plain tensor operations on the GPU instead of CLIP's image processor,
    which works on one PIL image at a time and is much slower for 25,000 images.
    """
    x = torch.from_numpy(np.ascontiguousarray(images)).float().div(255.0).unsqueeze(1).to(device)
    x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", align_corners=False)
    x = x.clamp(0.0, 1.0)  # bicubic can overshoot slightly outside [0, 1]
    x = x.repeat(1, 3, 1, 1)
    return (x - CLIP_MEAN.to(device)) / CLIP_STD.to(device)


def class_prompts(template):
    return [template.format(organ=name) for name in PROMPT_NAMES]


def encode_text(model, tokenizer, prompts, device):
    """11 prompts -> 11 unit-length text vectors, shape (11, 512)."""
    tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    text_features = model.get_text_features(input_ids=tokens["input_ids"],
                                            attention_mask=tokens["attention_mask"])
    return F.normalize(text_features, dim=-1)


def class_logits(model, pixel_values, text_features):
    """Similarity of every image to every class sentence, shape (N, 11).

    Cosine similarity (dot product of unit vectors) times CLIP's learned
    temperature (logit_scale, = 100 for this model). Softmax over these gives
    class probabilities, exactly like the CNN's output layer.
    """
    image_features = F.normalize(model.get_image_features(pixel_values=pixel_values), dim=-1)
    return model.logit_scale.exp() * image_features @ text_features.T


def predict_probs(model, tokenizer, images, template, device, batch_size=128):
    """Class probabilities for a whole split, no training. Returns numpy (N, 11)."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        text_features = encode_text(model, tokenizer, class_prompts(template), device)
        for start in range(0, len(images), batch_size):
            pixel_values = images_to_clip_input(images[start:start + batch_size], device)
            logits = class_logits(model, pixel_values, text_features)
            all_probs.append(torch.softmax(logits.float(), dim=1).cpu())
    return torch.cat(all_probs).numpy()


def count_parameters(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return total, trainable


def results_dir():
    return os.path.join(common.RESULTS_DIR, "foundation_model")
