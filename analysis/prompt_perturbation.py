import os
import glob
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from script import CollaborationController, format_prompt, get_shared_embedding

DEVICE = "cuda:0"
ENCODER_NAME = "bert-base-uncased"

DATASET_NAMES = {
    "gsm8k": "gsm8k",
    "mmlu": "cais/mmlu",
    "humaneval": "openai/openai_humaneval"
}

DATASET_CONFIGS = {
    "gsm8k": "main",
    "mmlu": "all",
    "humaneval": "openai_humaneval"
}

for task in ["humaneval", "gsm8k", "mmlu", ]:
    DATASET_NAME = DATASET_NAMES[task]
    DATASET_CONFIG = DATASET_CONFIGS[task]
    DATASET_SPLIT = "test"
    TASK_NAME = task
    CHECKPOINTS = sorted(
          glob.glob(f"checkpoints/{TASK_NAME}_run_for_interp_run1/collaboration_controller_{TASK_NAME}_epoch*.pt", recursive=True),
    )

    OUT_DIR = f"outputs/{TASK_NAME}/perturbation_epochs"
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n\n=== Perturbation analysis for task: {task} ===\n\n")

    def perturb_prompt(prompt, mode):
        if mode == "remove_numbers":
            return "".join(c for c in prompt if not c.isdigit())

        if mode == "mask_numbers":
            return "".join("NUM" if c.isdigit() else c for c in prompt)

        if mode == "shuffle_sentences":
            parts = prompt.split(". ")
            if len(parts) > 1:
                np.random.shuffle(parts)
            return ". ".join(parts)

        if mode == "remove_reasoning":
            return prompt.replace("Let's think step by step.", "")

        return prompt


    PERTURBATIONS = [
        "remove_numbers",
        "mask_numbers",
        "shuffle_sentences",
        "remove_reasoning",
    ]

    def kl_divergence(p, q):
        p = p.clamp(min=1e-9)
        q = q.clamp(min=1e-9)
        return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)

    def routing_entropy(S):
        S = S / S.sum(dim=-1, keepdim=True)
        return -(S * torch.log(S + 1e-9)).sum(dim=-1)

    print("[info] loading dataset")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    prompts = [format_prompt(ds[i], TASK_NAME) for i in range(len(ds))]
    print(f"[info] loaded {len(prompts)} prompts")

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    encoder = AutoModel.from_pretrained(ENCODER_NAME).to(DEVICE)
    encoder.eval()

    results = {} 

    for epoch_idx, ckpt in enumerate(CHECKPOINTS, start=1):
        print(f"[epoch {epoch_idx}] loading checkpoint")

        controller, _ = CollaborationController.from_saved_state(
            ckpt, device=DEVICE
        )
        controller.eval()

        base_emb = get_shared_embedding(
            prompts, encoder, tokenizer
        ).to(DEVICE)

        with torch.no_grad():
            C0, S0, _ = controller(base_emb)

        epoch_results = {}

        for mode in PERTURBATIONS:
            perturbed_prompts = [
                perturb_prompt(p, mode) for p in prompts
            ]

            emb = get_shared_embedding(
                perturbed_prompts, encoder, tokenizer
            ).to(DEVICE)

            with torch.no_grad():
                C, S, _ = controller(emb)

            kl_collab = kl_divergence(
                C0.view(C0.size(0), -1),
                C.view(C.size(0), -1)
            ).mean().item()

            kl_seq = kl_divergence(S0, S).mean().item()

            entropy_shift = (
                routing_entropy(S).mean()
                - routing_entropy(S0).mean()
            ).item()

            epoch_results[mode] = {
                "kl_collab": kl_collab,
                "kl_seq": kl_seq,
                "entropy_shift": entropy_shift
            }

            print(
                f"[epoch {epoch_idx} | {mode}] "
                f"KL(collab)={kl_collab:.4f} "
                f"KL(seq)={kl_seq:.4f} "
                f"Δentropy={entropy_shift:.4f}"
            )

        results[epoch_idx] = epoch_results

    with open(os.path.join(OUT_DIR, "perturbation_results_by_epoch.json"), "w") as f:
        json.dump(results, f, indent=2)

    epochs = sorted(results.keys())

    plt.figure(figsize=(7, 5))
    for mode in PERTURBATIONS:
        plt.plot(
            epochs,
            [results[e][mode]["kl_seq"] for e in epochs],
            marker="o",
            label=mode
        )

    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence (Sequence Head)")
    plt.title("Routing Sensitivity to Prompt Perturbations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "kl_sequence_over_epochs.pdf"))
    plt.close()

    plt.figure(figsize=(7, 5))
    for mode in PERTURBATIONS:
        plt.plot(
            epochs,
            [results[e][mode]["entropy_shift"] for e in epochs],
            marker="o",
            label=mode
        )

    plt.xlabel("Epoch")
    plt.ylabel("Entropy Shift")
    plt.title("Routing Confidence Change Under Perturbations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "entropy_shift_over_epochs.pdf"))
    plt.close()

    print("[DONE] Perturbation analysis saved →", OUT_DIR)
