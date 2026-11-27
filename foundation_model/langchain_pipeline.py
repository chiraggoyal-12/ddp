"""
foundation_model/langchain_pipeline.py - the finished fine-tuned CLIP, wrapped in LangChain.

LangChain is an inference-side tool. It connects steps such as "fill in a prompt
template", "call the model" and "parse the output" into one pipeline (a "chain")
that you can call with .invoke(). Training is already finished at this point
(finetune_lora.py); this file only RUNS the trained model.

    chain = build_inputs | run_finetuned_clip | parse_answers

    build_inputs        a LangChain PromptTemplate turns the template
                        "a CT scan of the {organ}." into the 11 class sentences
    run_finetuned_clip  the fine-tuned CLIP: images + 11 sentences -> probabilities
    parse_answers       probabilities -> organ name, confidence, a one-line answer

Simplification: LangChain's LLM classes assume "text in, text out". CLIP
does not generate text; it scores images against sentences. So the model step
is a RunnableLambda, LangChain's standard way to put any Python function into
a chain, rather than a custom LLM subclass.

The chain is run over the test set (the comparison set), and its predictions
are checked against the ones score_test.py saved. They should be identical:
the wrapper must not change any result.

    python -m foundation_model.langchain_pipeline --seed 42
"""
import argparse
import os
import time

import numpy as np
import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

import common
from foundation_model import clip_setup
from foundation_model.finetune_lora import chosen_template
from foundation_model.score_test import load_finetuned_clip

CHUNK_SIZE = 128  # images per chain call (same batch size as score_test.py)


def build_chain(model, tokenizer, template, device):
    class_prompt = PromptTemplate.from_template(template)
    answer_prompt = PromptTemplate.from_template(
        "Predicted organ: {organ} (class {label}), confidence {confidence}"
    )

    def build_inputs(request):
        # request = {"images": uint8 array of shape (N, 64, 64)}
        sentences = []
        for organ in clip_setup.PROMPT_NAMES:
            sentences.append(class_prompt.format(organ=organ))
        return {"images": request["images"], "class_sentences": sentences}

    def run_finetuned_clip(inputs):
        with torch.no_grad():
            text_features = clip_setup.encode_text(model, tokenizer, inputs["class_sentences"], device)
            pixel_values = clip_setup.images_to_clip_input(inputs["images"], device)
            logits = clip_setup.class_logits(model, pixel_values, text_features)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        return probs

    def parse_answers(probs):
        answers = []
        for row in probs:
            label = int(row.argmax())
            confidence = float(row[label])
            answers.append({
                "label": label,
                "organ": common.CLASS_NAMES[label],
                "confidence": confidence,
                "probabilities": row,
                "answer": answer_prompt.format(organ=common.CLASS_NAMES[label], label=label,
                                               confidence=f"{confidence:.2f}"),
            })
        return answers

    return RunnableLambda(build_inputs) | RunnableLambda(run_finetuned_clip) | RunnableLambda(parse_answers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, adapter_dir = load_finetuned_clip(args.seed, device)
    template = chosen_template()
    chain = build_chain(model, tokenizer, template, device)

    test_images, test_labels = common.load_split("test")
    start_time = time.time()
    answers = []
    for start in range(0, len(test_images), CHUNK_SIZE):
        answers.extend(chain.invoke({"images": test_images[start:start + CHUNK_SIZE]}))
    seconds = time.time() - start_time

    chain_labels = np.array([a["label"] for a in answers])
    chain_probs = np.stack([a["probabilities"] for a in answers])
    chain_metrics = common.compute_metrics(test_labels, chain_labels)
    common.print_metrics("LangChain pipeline (fine-tuned CLIP) - TEST", chain_metrics)

    # Consistency check against the plain-PyTorch predictions from score_test.py.
    saved = np.load(os.path.join(clip_setup.results_dir(), "finetuned_test_probabilities.npz"))
    saved_labels = saved["probs"].argmax(axis=1)
    agreement = float((saved_labels == chain_labels).mean())
    max_prob_difference = float(np.abs(saved["probs"] - chain_probs).max())
    print(f"agreement with score_test.py predictions: {agreement:.4%}, "
          f"largest probability difference {max_prob_difference:.2e}")
    print("first three answers:")
    for a in answers[:3]:
        print("  " + a["answer"])

    examples = []
    for i in range(5):
        examples.append({"index": i, "true_organ": common.CLASS_NAMES[int(test_labels[i])],
                         "answer": answers[i]["answer"]})
    common.save_json(os.path.join(clip_setup.results_dir(), "langchain_check.json"), {
        "chain": "RunnableLambda(build_inputs: PromptTemplate) | RunnableLambda(run_finetuned_clip) "
                 "| RunnableLambda(parse_answers: PromptTemplate)",
        "adapter_dir": os.path.relpath(adapter_dir, common.PROJECT_DIR),
        "template": template,
        "n_test_images": int(len(test_labels)),
        "seconds": seconds,
        "agreement_with_plain_pytorch": agreement,
        "max_probability_difference": max_prob_difference,
        "test_accuracy_via_chain": chain_metrics["accuracy"],
        "test_macro_f1_via_chain": chain_metrics["macro_f1"],
        "examples": examples,
    })


if __name__ == "__main__":
    main()
