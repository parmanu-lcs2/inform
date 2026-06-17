import os
import json
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import httpx
import time
import seaborn as sns
import argparse

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "savefig.bbox": "tight"
})

sns.set_context(
    "paper",
    rc={
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    }
)

from script import CollaborationController, format_prompt, get_shared_embedding, generate_text

DEVICE = "cuda:0"
SHAReds_ENCODER_NAME = "/home/models/bert-base-uncased"
TOP_K_LAYERS = 12
EXPERT_TYPES = 'hosted'
proxy_url = "PROXY_URL"
http_client = httpx.Client(
    transport=httpx.HTTPTransport(proxy=proxy_url, verify=False),
    verify=False
)


def safe_normalize_lastdim(x):
    x = x.clamp(min=1e-9)
    s = x.sum(dim=-1, keepdim=True)
    s[s == 0] = 1.0
    return x / s


def kl_scalar_batch(p, q):
    p = p.clamp(min=1e-9)
    q = q.clamp(min=1e-9)
    return torch.sum(p * (p.log() - q.log()), dim=-1)


def kl_head_change_batch(controller_a, controller_b, input_embeds, shaReds_reps, device=DEVICE):
    controller_a.eval()
    controller_b.eval()

    with torch.no_grad():
        C1, S1, _ = controller_a(input_embeds.to(device), shaReds_reps.to(device))
        C2, S2, _ = controller_b(input_embeds.to(device), shaReds_reps.to(device))

        # Flatten collab matrix
        B = C1.size(0)
        Pc = safe_normalize_lastdim(C1.view(B, -1))
        Qc = safe_normalize_lastdim(C2.view(B, -1))

        Ps = safe_normalize_lastdim(S1)
        Qs = safe_normalize_lastdim(S2)

        kl_c = kl_scalar_batch(Pc, Qc).mean().item()
        kl_s = kl_scalar_batch(Ps, Qs).mean().item()

    return kl_c, kl_s


def get_expert_representations(prompts, models, model_names, model_temps, shaReds_encoder, shaReds_tokenizer, device=DEVICE):
    print("[info] generating expert representations via API...")
    
    shaReds_reps_list = []
    
    for model_idx, (model, model_name, temp) in enumerate(zip(models, model_names, model_temps)):
        print(f"[info] calling expert {model_idx+1}/{len(models)}: {model_name} (temp={temp})")
        
        # Generate text from this expert (using your generate_text function)
        expert_outputs = generate_text(
            model, 
            None,  # tokenizer=None for hosted models
            prompts, 
            infer_start=True,  # Short inference for representation
            model_name=model_name,
            temperature=temp  # Use the specific temperature for this model
        )
        
        # Get embeddings for these outputs
        expert_emb = get_shared_embedding(
            expert_outputs, 
            shaReds_encoder, 
            shaReds_tokenizer
        )
        
        shaReds_reps_list.append(expert_emb)
        
        # Brief pause to avoid rate limiting
        time.sleep(1)
    
    # Stack to [N, M, D]
    shaReds_reps = torch.stack(shaReds_reps_list, dim=1).to(device)
    print(f"[info] expert representations shape: {shaReds_reps.shape}")
    
    return shaReds_reps


def analyze_checkpoints_dataset(
    task,
    checkpoints,
    model_names,
    model_temps,  # Added model temperatures parameter
    dataset_name="gsm8k",
    dataset_config="main",
    dataset_split="test",
    output_dir="analysis_results_dataset",
    device=DEVICE,
    top_k_layers=TOP_K_LAYERS,
    batch_size=5  
):
    os.makedirs(output_dir, exist_ok=True)

    print("[info] device:", device)
    print("[info] loading dataset:", dataset_name, dataset_config, dataset_split)
    print(f"[info] model temperatures: {model_temps}")

    if dataset_name == "cais/mmlu":
        ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
    elif dataset_name == "gsm8k":
        ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
    else:
        ds = load_dataset(dataset_name, split=dataset_split)
        
    num_examples = len(ds)
    if num_examples == 0:
        raise ValueError("Dataset slice is empty.")
    print(f"[info] loaded {num_examples} examples")

    shaReds_tokenizer = AutoTokenizer.from_pretrained(SHAReds_ENCODER_NAME)
    shaReds_encoder = AutoModel.from_pretrained(SHAReds_ENCODER_NAME).to(device)

    print("[info] initializing expert models...")
    models = [
        OpenAI(
            base_url="BASE_URL",
            api_key="API_KEY",
            http_client=http_client
        ) for _ in model_names
    ]
    
    prompts = []
    for idx in range(num_examples):
        sample = ds[idx]
        prompt = format_prompt(sample, task)
        prompts.append(prompt)

    print("[info] computing input embeddings from prompts...")
    input_embeddings = get_shared_embedding(prompts, shaReds_encoder, shaReds_tokenizer).cpu()
    print("[info] input embeddings shape:", input_embeddings.shape)

    print("[info] getting expert representations (this may take a while)...")
    all_shaReds_reps = []
    
    for i in range(0, num_examples, batch_size):
        batch_prompts = prompts[i:i+batch_size]
        print(f"[info] processing batch {i//batch_size + 1}/{(num_examples + batch_size - 1)//batch_size}")
        
        batch_shaReds_reps = get_expert_representations(
            batch_prompts,
            models,
            model_names,
            model_temps,  # Pass temperatures to the function
            shaReds_encoder,
            shaReds_tokenizer,
            device=device
        )
        
        all_shaReds_reps.append(batch_shaReds_reps.cpu())
    
    # Concatenate all batches
    shaReds_reps = torch.cat(all_shaReds_reps, dim=0)
    print(f"[info] total shaReds_reps shape: {shaReds_reps.shape}")
    
    # Save representations for reuse
    torch.save({
        'input_embeddings': input_embeddings,
        'shaReds_reps': shaReds_reps,
        'prompts': prompts,
        'model_names': model_names,
        'model_temps': model_temps
    }, os.path.join(output_dir, "cached_representations.pt"))
    print("[info] cached representations saved")

    # Load checkpoints
    controllers = []
    state_dicts = []
    ckpt_names = []

    for ckpt in checkpoints:
        print("[info] loading", ckpt)
        controller, saved = CollaborationController.from_saved_state(ckpt, device=device)
        controller.eval()
        controllers.append(controller)
        state_dicts.append(saved["state_dict"])
        ckpt_names.append(os.path.basename(ckpt))

    num_epochs = len(controllers)
    if num_epochs < 2:
        raise ValueError("Need at least 2 checkpoints.")

    # Per-layer norms & drifts
    layer_norm = OrderedDict()
    layer_l2 = OrderedDict()
    layer_cos = OrderedDict()

    first_sd = state_dicts[0]

    for name in first_sd:
        layer_norm[name] = []
        layer_l2[name] = []
        layer_cos[name] = []

    # Norms per epoch
    for sd in state_dicts:
        for name, t in sd.items():
            t = t.detach().cpu()
            layer_norm[name].append(float(torch.norm(t)))

    # Drift computations
    def l2_change(a, b): 
        return torch.norm(a - b).item()

    def cos_change(a, b):
        v1 = a.flatten()
        v2 = b.flatten()
        if torch.all(v1 == 0) or torch.all(v2 == 0):
            return float("nan")
        return 1 - F.cosine_similarity(v1, v2, dim=0).item()

    for e in range(1, num_epochs):
        prev = state_dicts[e - 1]
        curr = state_dicts[e]
        for name in curr:
            t_prev = prev[name].detach().cpu()
            t_curr = curr[name].detach().cpu()
            layer_l2[name].append(l2_change(t_curr, t_prev))
            layer_cos[name].append(cos_change(t_curr, t_prev))

    # Save CSVs
    csv_metrics = os.path.join(output_dir, "layer_metrics.csv")
    with open(csv_metrics, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "epoch", "norm"])
        for name, vals in layer_norm.items():
            for e, val in enumerate(vals, start=1):
                w.writerow([name, e, val])

    csv_drifts = os.path.join(output_dir, "layer_drifts.csv")
    with open(csv_drifts, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "from_epoch", "to_epoch", "l2_change", "cosine_change"])
        for name in layer_l2:
            for i, (l2v, cosv) in enumerate(zip(layer_l2[name], layer_cos[name])):
                w.writerow([name, i + 1, i + 2, l2v, cosv])

    print("[csv] saved:", csv_metrics, csv_drifts)

    # Averaged Collaboration & Sequence distributions
    collab_avg = []
    seq_avg = []

    for epoch_idx, controller in enumerate(controllers):
        controller.eval()
        with torch.no_grad():
            # Pass both input_embeddings and shaReds_reps
            C, S, _ = controller(
                input_embeddings.to(device), 
                shaReds_reps.to(device)
            )
            C_mean = C.mean(dim=0).cpu().numpy()
            S_mean = S.mean(dim=0).cpu().numpy()
            collab_avg.append(C_mean)
            seq_avg.append(S_mean)
        print(f"[epoch {epoch_idx+1}] computed averages")

    np.save(os.path.join(output_dir, "collab_matrices_avg.npy"), np.array(collab_avg))
    np.save(os.path.join(output_dir, "seq_distributions_avg.npy"), np.array(seq_avg))

    # KL divergences across checkpoints
    kl_collab = []
    kl_seq = []

    for e in range(1, num_epochs):
        k1, k2 = kl_head_change_batch(
            controllers[e - 1], 
            controllers[e], 
            input_embeddings,
            shaReds_reps,
            device=device
        )
        kl_collab.append(k1)
        kl_seq.append(k2)
        print(f"[kl] {e}->{e+1}: collab={k1:.6f}, seq={k2:.6f}")

    # Select top-K layers for plots
    mean_l2 = {name: (np.nanmean(layer_l2[name]) if layer_l2[name] else 0) for name in layer_l2}
    sorted_layers = sorted(mean_l2.items(), key=lambda x: x[1], reverse=True)
    top_layers = [name for name, _ in sorted_layers[:top_k_layers]]

    # Plot: Norm evolution
    plt.figure(figsize=(12, 6))
    for name in top_layers:
        plt.plot(range(1, num_epochs + 1), layer_norm[name], label=name)
    plt.title("Top Layers — Norm Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("L2 Norm")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_layers_norm_evolution.pdf"), dpi=300)
    plt.close()

    # Plot: L2 drift
    plt.figure(figsize=(12, 6))
    for name in top_layers:
        plt.plot(range(2, num_epochs + 1), layer_l2[name], marker="o", label=name)
    plt.title("Top Layers — L2 Drift")
    plt.xlabel("Epoch")
    plt.ylabel("L2 Change")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_layers_l2_drift.pdf"), dpi=300)
    plt.close()

    # Plot: KL divergence
    plt.figure(figsize=(8, 5))
    x = range(2, num_epochs + 1)
    plt.plot(x, kl_collab, marker="o", label="KL Collaboration Head")
    plt.plot(x, kl_seq, marker="o", label="KL Sequence Head")
    plt.title("KL Divergence Across Epoch Transitions")
    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kl_head_divergence.pdf"), dpi=300)
    plt.close()

    # Heatmaps for collab matrix per epoch
    for i, C in enumerate(collab_avg):
        plt.figure(figsize=(8, 6))
        im = plt.imshow(C, aspect="auto", interpolation="nearest", cmap="viridis")
        plt.colorbar(im)

        plt.title(f"Collaboration Matrix — Epoch {i+1}")
        plt.xlabel("To Model")
        plt.ylabel("From Model")

        # Axis ticks
        if len(model_names) == C.shape[0]:
            short_names = [name.split('/')[-1][:15] for name in model_names]
            plt.xticks(range(len(short_names)), short_names, rotation=45, ha="right")
            plt.yticks(range(len(short_names)), short_names)

        # Cell annotations
        for y in range(C.shape[0]):
            for x in range(C.shape[1]):
                plt.text(
                    x, y,
                    f"{C[y, x]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white"
                )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"collab_matrix_epoch{i+1}.pdf"), dpi=300)
        plt.close()
    seq_avg = np.array(seq_avg)
    num_epochs, num_experts = seq_avg.shape

    vmin, vmax = seq_avg.min(), seq_avg.max()

    def heat_color(val):
        # normalize to [0,1]
        alpha = (val - vmin) / (vmax - vmin + 1e-8)
        # convert to LaTeX color mix
        return f"heathigh!{int(alpha * 100)}!heatlow"

    tex_path = os.path.join(
        output_dir,
        f"{task}_seq_distribution_table_heatmap.tex"
    )

    with open(tex_path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write(
            "\\setlength{\\tabcolsep}{6pt}\n"
        )
        f.write(
            "\\begin{tabular}{c" + "c" * num_epochs + "}\n"
        )
        f.write("\\toprule\n")

        # Header
        header = ["Expert"] + [f"Epoch {i+1}" for i in range(num_epochs)]
        f.write(" & ".join(header) + " \\\\\n")
        f.write("\\midrule\n")

        # Rows (experts)
        for e in range(num_experts):
            row = [f"Expert {e+1}"]
            for ep in range(num_epochs):
                val = seq_avg[ep, e]
                color = heat_color(val)
                cell = (
                    f"\\cellcolor{{{color}}}"
                    f"{val:.3f}"
                )
                row.append(cell)
            f.write(" & ".join(row) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(
            f"\\caption{{Transposed expert selection probabilities across epochs for "
            f"\\textsc{{{task.upper()}}}. Cell intensity reflects selection likelihood.}}\n"
        )
        f.write(
            f"\\label{{tab:{task}_seq_distribution_heatmap}}\n"
        )
        f.write("\\end{table}\n")

    print(f"Heatmap LaTeX table written to: {tex_path}")
    
    summary = {
        "num_checkpoints": num_epochs,
        "checkpoints": ckpt_names,
        "model_names": model_names,
        "model_temps": model_temps,
        "dataset_size": num_examples,
        "dataset_split": dataset_split,
        "top_layers": top_layers,
        "kl_collab": kl_collab,
        "kl_seq": kl_seq,
        "final_selection_probs": seq_avg[-1].tolist(),
        "final_collab_matrix": collab_avg[-1].tolist()
    }

    with open(os.path.join(output_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[done] Analysis complete →", output_dir)
    return output_dir


if __name__ == "__main__":
    # Define model names and temperatures
    MODEL_NAMES = [
        'meta-llama/llama-3.1-8b-instruct',
        'qwen/qwen3-8b-fp8',
        'deepseek/deepseek-r1-0528-qwen3-8b',
        'meta-llama/llama-3.1-8b-instruct',
        'qwen/qwen3-8b-fp8',
        'deepseek/deepseek-r1-0528-qwen3-8b',
        'meta-llama/llama-3.1-8b-instruct',
        'qwen/qwen3-8b-fp8',
        'deepseek/deepseek-r1-0528-qwen3-8b',
        'meta-llama/llama-3.1-8b-instruct'
    ]

    MODEL_TEMPS = [0.0, 0.2, 0.5, 0.5, 0.0, 1.0, 1.2, 1.5, 2.0, 1.0]
    task="gsm8k"
    import glob
    checkpoints = sorted(
        glob.glob(f"checkpoints/{task}_run_for_interp_run1/collaboration_controller_{task}_epoch*.pt", recursive=True),
    )
    
    analyze_checkpoints_dataset(
        task,
        checkpoints,
        MODEL_NAMES,
        MODEL_TEMPS,
        dataset_name="gsm8k",
        dataset_config="main",
        dataset_split="test",
        output_dir="outputs/gsm8k/test_10/analysis_shared_reps",
        device=DEVICE,
        top_k_layers=12,
        batch_size=5
    )
