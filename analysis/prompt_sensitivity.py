import os
import csv
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

for task in ["gsm8k", "mmlu", "humaneval"]:
    DATASET_NAME = DATASET_NAMES[task]
    DATASET_CONFIG = DATASET_CONFIGS[task]
    DATASET_SPLIT = "test[:1%]"
    TASK_NAME = task
    CHECKPOINTS = sorted(
          glob.glob(f"checkpoints/{TASK_NAME}_run_for_interp_run1/collaboration_controller_{TASK_NAME}_epoch*.pt", recursive=True),
    )

    OUT_DIR = f"outputs/{TASK_NAME}/prompt_sensitivity_epochs"
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n\n=== Prompt Sensitivity analysis for task: {task} ===\n\n")

    def routing_entropy(p):
        p = p / p.sum()
        return -(p * torch.log(p + 1e-9)).sum().item()

    def classify_token(tok):
        if any(c.isdigit() for c in tok):
            return "numbers"
        if tok in ["+", "-", "*", "/", "="]:
            return "operators"
        if tok.lower() in ["question", "answer", "q", "a", ":"]:
            return "format"
        return "text"

    def l1_shift(a, b):
        return sum(abs(a[k] - b[k]) for k in a)

    print("[info] loading dataset")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    prompts = [format_prompt(ds[i], TASK_NAME) for i in range(len(ds))]
    print(f"[info] loaded {len(prompts)} prompts")

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    encoder = AutoModel.from_pretrained(ENCODER_NAME).to(DEVICE)
    encoder.eval()

    epoch_token_rows = {} 
    epoch_group_means = {} 
    epoch_entropy_means = {}

    for epoch_idx, ckpt in enumerate(CHECKPOINTS, start=1):
        print(f"[epoch {epoch_idx}] loading checkpoint")

        controller, _ = CollaborationController.from_saved_state(
            ckpt, device=DEVICE
        )
        controller.eval()

        token_rows = []
        group_scores = {
            "numbers": [],
            "operators": [],
            "format": [],
            "text": []
        }
        entropy_vals = []

        for pid, prompt in enumerate(prompts):
            tokens = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True
            ).to(DEVICE)

            out = encoder(**tokens, output_hidden_states=True)
            emb = out.last_hidden_state
            emb.requires_grad_(True)
            emb.retain_grad()

            pooled = emb.mean(dim=1)
            C, S, _ = controller(pooled)
            controller.zero_grad(set_to_none=True)
            encoder.zero_grad(set_to_none=True)

            grads = emb.grad.abs().sum(dim=-1).squeeze(0)
            grads = grads / grads.sum()

            token_strs = tokenizer.convert_ids_to_tokens(
                tokens["input_ids"][0]
            )

            for tok, g in zip(token_strs, grads):
                grp = classify_token(tok)
                token_rows.append([
                    epoch_idx,
                    pid,
                    tok,
                    grp,
                    g.item()
                ])
                group_scores[grp].append(g.item())

            entropy_vals.append(routing_entropy(S.squeeze()))

        epoch_token_rows[epoch_idx] = token_rows
        epoch_group_means[epoch_idx] = {
            k: float(np.mean(v)) for k, v in group_scores.items()
        }
        epoch_entropy_means[epoch_idx] = float(np.mean(entropy_vals))

        print(f"[epoch {epoch_idx}] done")

    csv_path = os.path.join(OUT_DIR, "token_attribution_all_epochs.csv")
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "prompt_id", "token", "group", "importance"])
        for e in epoch_token_rows:
            w.writerows(epoch_token_rows[e])

    with open(os.path.join(OUT_DIR, "group_attribution_by_epoch.json"), "w") as f:
        json.dump(epoch_group_means, f, indent=2)

    with open(os.path.join(OUT_DIR, "routing_entropy_by_epoch.json"), "w") as f:
        json.dump(epoch_entropy_means, f, indent=2)

    epochs = sorted(epoch_group_means.keys())

    plt.figure(figsize=(7, 5))
    for grp in ["numbers", "operators", "format", "text"]:
        plt.plot(
            epochs,
            [epoch_group_means[e][grp] for e in epochs],
            marker="o",
            label=grp
        )

    plt.xlabel("Epoch")
    plt.ylabel("Mean Attribution")
    plt.title("Prompt Feature Sensitivity Across Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "group_attribution_over_epochs.pdf"))
    plt.close()

    plt.figure()
    plt.plot(
        epochs,
        [epoch_entropy_means[e] for e in epochs],
        marker="o"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean Routing Entropy")
    plt.title("Routing Entropy Across Epochs")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "routing_entropy_over_epochs.pdf"))
    plt.close()

    shifts = []
    for i in range(1, len(epochs)):
        shifts.append(
            l1_shift(
                epoch_group_means[epochs[i-1]],
                epoch_group_means[epochs[i]]
            )
        )

    plt.figure()
    plt.plot(epochs[1:], shifts, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Attribution Shift")
    plt.title("Stability of Prompt Sensitivity")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "attribution_shift.pdf"))
    plt.close()

    print("[DONE] Epoch-wise prompt sensitivity analysis saved →", OUT_DIR)
