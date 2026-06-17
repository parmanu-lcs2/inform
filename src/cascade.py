import os
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

OUTPUT_DIR = "cascade_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from datasets import load_dataset
import random

def load_mixed_prompts(
    gsm_n=100,
    mmlu_n=200,
    humaneval_n=164,
    seed=42,
):
    random.seed(seed)
    prompts = []

    gsm = load_dataset("gsm8k", "main", split="test")
    gsm = random.sample(list(gsm), gsm_n)
    for ex in gsm:
        prompts.append({
            "dataset": "gsm8k",
            "prompt": f"Solve step by step:\n\n{ex['question']}"
        })

    mmlu = load_dataset("cais/mmlu", "all", split="test")
    mmlu = random.sample(list(mmlu), mmlu_n)
    for ex in mmlu:
        prompt = (
            f"Question: {ex['question']}\n"
            f"A. {ex['choices'][0]}\n"
            f"B. {ex['choices'][1]}\n"
            f"C. {ex['choices'][2]}\n"
            f"D. {ex['choices'][3]}\n\n"
            "Answer:"
        )
        prompts.append({
            "dataset": "mmlu",
            "prompt": prompt
        })

    humaneval = load_dataset("openai_humaneval", split="test")
    humaneval = list(humaneval)[:humaneval_n]
    for ex in humaneval:
        prompts.append({
            "dataset": "humaneval",
            "prompt": f"Write Python code:\n\n{ex['prompt']}"
        })

    random.shuffle(prompts)
    return prompts

def average_logprob(scores, generated_ids):
    logits = torch.stack(scores, dim=1) 
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(
        -1, generated_ids.unsqueeze(-1)
    ).squeeze(-1)
    return token_log_probs.mean()

class CascadeRouter(nn.Module):
    def __init__(
        self,
        experts: List[nn.Module],
        tokenizers: List[AutoTokenizer],
        stop_threshold=-1.5,
        max_new_tokens=128,
        temperature=0.7,
    ):
        super().__init__()
        self.experts = experts
        self.tokenizers = tokenizers
        self.stop_threshold = stop_threshold
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def forward(self, prompt: str):
        routing_trace = []
        context = prompt

        for idx, (expert, tokenizer) in enumerate(zip(self.experts, self.tokenizers)):
            result = self._run_expert(expert, tokenizer, context, idx)
            
            if result is None:  # Expert is masked
                routing_trace.append({
                    "expert_idx": idx,
                    "confidence": 0.0,
                    "stop_prob": 0.0,
                    "intrinsic": 0.0,
                    "output_text": "",
                })
                continue

            intrinsic = abs(float(result["stop_prob"]) - 1.0) 

            print({
                "expert_idx": idx,
                "confidence": float(result["confidence"]),
                "stop_prob": float(result["stop_prob"]),
                "intrinsic": intrinsic,
                "output_text": result["output_text"],
            })

            routing_trace.append({
                "expert_idx": idx,
                "confidence": float(result["confidence"]),
                "stop_prob": float(result["stop_prob"]),
                "intrinsic": intrinsic,
                "output_text": result["output_text"],
            })


            if result["stop_prob"].item() >= 0.5:
                return result["output_text"], routing_trace

            # Update context more carefully to avoid tokenization issues
            context = prompt + "\n\nPrevious response: " + result["output_text"]

        return routing_trace[-1]["output_text"], routing_trace

    def _run_expert(self, expert, tokenizer, context, expert_idx=None):
        # Check if expert is masked
        if hasattr(expert, '_masked') and expert._masked:
            return None
        
        try:
            inputs = tokenizer(
                context, 
                return_tensors="pt", 
                truncation=True,
                max_length=2048,  # Prevent overly long contexts
                padding=False
            ).to(DEVICE)

            # with torch.no_grad():
            outputs = expert.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            gen_ids = outputs.sequences[:, inputs["input_ids"].shape[1]:]
            output_text = tokenizer.decode(
                gen_ids[0], skip_special_tokens=True
            ).strip()

            # Ensure we have valid output
            if len(gen_ids[0]) == 0 or not output_text:
                output_text = "[No output generated]"
                avg_logp = torch.tensor(-10.0)
            else:
                avg_logp = average_logprob(outputs.scores, gen_ids)
            
            stop_prob = torch.sigmoid(avg_logp - self.stop_threshold)
            logits = torch.stack(outputs.scores, dim=0)   # [T, V]
            rep = logits.mean().item() / np.sqrt(logits.size(-1))   

            return {
                "output_text": output_text,
                "confidence": avg_logp,
                "stop_prob": stop_prob,
                "rep": rep
            }
        
        except Exception as e:
            print(f"Error in _run_expert for expert {expert_idx}: {e}")
            import traceback
            traceback.print_exc()
            # Return a fallback result
            return {
                "output_text": "[Error in generation]",
                "confidence": torch.tensor(-10.0),
                "stop_prob": torch.tensor(0.0),
            }

def mask_expert(expert: nn.Module):
    """Mark expert as masked instead of modifying its generate method"""
    expert._masked = True

def unmask_expert(expert: nn.Module):
    """Unmask expert"""
    if hasattr(expert, '_masked'):
        delattr(expert, '_masked')

class PromptDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if isinstance(sample, dict):
            return sample["prompt"]
        else:  # fallback for plain string lists
            return sample

def evaluate_cascade(router, dataloader):
    traces = []
    stop_indices = []

    for batch in dataloader:
        for prompt in batch:
            try:
                _, trace = router(prompt)
                traces.append(trace)
                # Find the last non-masked expert that was used
                for step in reversed(trace):
                    if step["output_text"] and step["output_text"] not in ["", "[No output generated]", "[Error in generation]"]:
                        stop_indices.append(step["expert_idx"])
                        break
                else:
                    # If all masked or failed, use last expert
                    stop_indices.append(trace[-1]["expert_idx"])
            except Exception as e:
                print(f"Error evaluating prompt '{prompt[:50]}...': {e}")
                continue

    return traces, stop_indices

def stopping_distribution(stop_indices, n):
    if len(stop_indices) == 0:
        return np.ones(n) / n  # Uniform if no data
    counts = np.zeros(n)
    for i in stop_indices:
        counts[i] += 1
    total = counts.sum()
    if total == 0:
        return np.ones(n) / n
    return counts / total

def entropy(p):
    p = np.clip(p, 1e-8, 1.0)
    return -np.sum(p * np.log(p))

def kl_divergence(p, q):
    p = np.clip(p, 1e-8, 1.0)
    q = np.clip(q, 1e-8, 1.0)
    return np.sum(p * np.log(p / q))

def masked_routing(router, dataloader):
    _, base_stops = evaluate_cascade(router, dataloader)
    base_dist = stopping_distribution(base_stops, len(router.experts))

    kl_scores = {}

    for k in range(len(router.experts)):
        print(f"Masking expert {k}...")
        mask_expert(router.experts[k])

        _, masked_stops = evaluate_cascade(router, dataloader)
        masked_dist = stopping_distribution(
            masked_stops, len(router.experts)
        )

        kl_scores[k] = kl_divergence(base_dist, masked_dist)
        unmask_expert(router.experts[k])

    return base_dist, kl_scores

def aggregate_intrinsic(traces):
    scores = {}

    for trace in traces:
        for step in trace:
            idx = step["expert_idx"]
            if step["intrinsic"] > 0:  # Only count non-masked experts
                scores.setdefault(idx, []).append(step["intrinsic"])

    return {k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()}



def plot_kl_vs_intrinsic(kl, intrinsic):
    xs = []
    ys = []

    for k in kl:
        xs.append(intrinsic.get(k, 0.0))
        ys.append(kl[k])

    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, s=70)

    for i, (x, y) in enumerate(zip(xs, ys)):
        plt.text(x, y, f"E{i}", fontsize=10)

    plt.xlabel("Intrinsic Importance")
    plt.ylabel("KL Divergence (Routing Collapse)")
    plt.title("Cascade: Causal Importance vs Routing Dominance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kl_vs_intrinsic.pdf"))
    plt.close()

def save_results(base_dist, intrinsic, kl):
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(
            {
                "stopping_distribution": base_dist.tolist(),
                "entropy": float(entropy(base_dist)),
                "intrinsic_importance": intrinsic,
                "kl_divergence": kl,
            },
            f,
            indent=2,
        )


def main():
    model_names = [
      '/home/models/Llama-3.2-1B-Instruct',
      '/home/models/Qwen2.5-3B-Instruct',
      '/home/models/Mistral-7B-Instruct-v0.1',
    ]

    print("Loading tokenizers and models...")
    tokenizers = []
    experts = []
    
    for m in model_names:
        print(f"Loading {m}...")
        tokenizer = AutoTokenizer.from_pretrained(m)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizers.append(tokenizer)
        
        model = AutoModelForCausalLM.from_pretrained(
            m,
            torch_dtype=torch.float16,
            device_map=DEVICE,
        )
        model.eval()
        experts.append(model)

    router = CascadeRouter(experts, tokenizers, stop_threshold=-0.2)

    prompts = [
        "Solve: If x + y = 10 and x - y = 2, find x and y.",
        "Explain why quicksort has average O(n log n).",
        "Differentiate sin(x) * exp(x).",
        "Write Python code to reverse a linked list.",
        "What is Bayes theorem?",
    ]

    dataloader = DataLoader(
        PromptDataset(prompts), batch_size=1, shuffle=False
    )

    print("Evaluating cascade...")
    traces, stops = evaluate_cascade(router, dataloader)
    intrinsic = aggregate_intrinsic(traces)

    print("Loading datasets...")
    samples = load_mixed_prompts()
    dataset = PromptDataset(samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print("Running masked routing...")
    base_dist, kl = masked_routing(router, dataloader)

    save_results(base_dist, intrinsic, kl)
    plot_kl_vs_intrinsic(kl, intrinsic)

    print("Evaluation complete.")
    print("Results saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
