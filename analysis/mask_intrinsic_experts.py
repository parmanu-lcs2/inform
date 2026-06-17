import os
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from scipy.stats import entropy

from script import (
    CollaborationController,
    format_prompt,
    generate_text,
    get_shared_embedding,
    DEVICE
)

DATASET_MAP = {
    "mmlu": ("cais/mmlu", "all"),
    "gsm8k": ("gsm8k", "main"),
    "humaneval": ("openai/openai_humaneval", "openai_humaneval")
}

def kl_div(p, q, eps=1e-8):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return entropy(p, q)


def get_topk_intrinsic(attr_file, k):
    data = torch.load(attr_file, map_location="cpu")
    grad_attr = data["grad_attr"]            # [S, N]
    mean_attr = grad_attr.mean(dim=0)        # [N]
    ranked = torch.argsort(mean_attr, descending=True)
    return ranked[:k].tolist()


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    dataset_name, dataset_cfg = DATASET_MAP[args.task]
    dataset = load_dataset(
        dataset_name,
        dataset_cfg,
        split=f"validation[:{args.num_samples}]" if args.task == "mmlu"
        else f"test[:{args.num_samples}]"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    encoder = AutoModel.from_pretrained(args.encoder).to(DEVICE)
    encoder.eval()

    models = [
        OpenAI(
            base_url=args.base_url,
            api_key="API_KEY",
        )
        for _ in args.model_names
    ]

    ckpts = sorted(
        glob.glob(os.path.join(args.ckpt_dir, "collaboration_controller_*epoch*.pt"))
    )
    if not ckpts:
        raise RuntimeError("No checkpoints found.")

    print(f"[info] Found {len(ckpts)} checkpoints")

    for ckpt_path in ckpts:
        epoch = ckpt_path.split("epoch")[-1].split("_")[0]
        print(f"\n[epoch {epoch}]")

        controller, _ = CollaborationController.from_saved_state(
            ckpt_path, device=DEVICE
        )
        controller.eval()

        attr_file = os.path.join(
            args.attr_dir, f"{args.task}_epoch{epoch}.pt"
        )
        topk = get_topk_intrinsic(attr_file, args.top_k)
        print(f"Masking intrinsic experts: {topk}")

        kl_seq, kl_route = [], []

        for step in tqdm(range(0, len(dataset), args.batch_size)):
            batch = dataset.select(
                range(step, min(step + args.batch_size, len(dataset)))
            )
            prompts = [format_prompt(ex, args.task) for ex in batch]

            with torch.no_grad():
                input_emb = get_shared_embedding(
                    prompts, encoder, tokenizer
                ).to(DEVICE)

            shared_reps = []
            for model, name, temp in zip(
                models, args.model_names, args.model_temps
            ):
                outputs = generate_text(
                    model,
                    tokenizer=None,
                    prompts=prompts,
                    infer_start=True,
                    model_name=name,
                    temperature=temp
                )
                rep = get_shared_embedding(outputs, encoder, tokenizer)
                shared_reps.append(rep)

            shared_reps = torch.stack(shared_reps, dim=1).to(DEVICE)

            with torch.no_grad():
                C_base, s_base, _ = controller(input_emb, shared_reps)

            masked_reps = shared_reps.clone()
            for idx in topk:
                masked_reps[:, idx, :] = 0.0

            with torch.no_grad():
                C_mask, s_mask, _ = controller(input_emb, masked_reps)

            kl_seq.append(
                kl_div(
                    s_base.mean(dim=0).cpu().numpy(),
                    s_mask.mean(dim=0).cpu().numpy()
                )
            )

            kl_route.append(
                kl_div(
                    C_base.mean(dim=(0, 1)).cpu().numpy(),
                    C_mask.mean(dim=(0, 1)).cpu().numpy()
                )
            )

        save_path = os.path.join(
            args.out_dir,
            f"{args.task}_epoch{epoch}_mask_intrinsic_top{args.top_k}.pt"
        )

        torch.save({
            "epoch": int(epoch),
            "masked_experts": topk,
            "kl_sequence": float(np.mean(kl_seq)),
            "kl_routing": float(np.mean(kl_route)),
        }, save_path)

        print(f"[saved] {save_path}")

    print("\n[DONE] Intrinsic expert masking ablation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", required=True,
                        choices=["mmlu", "gsm8k", "humaneval"])
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--attr_dir", required=True)
    parser.add_argument("--out_dir", default="mask_intrinsic_results")

    parser.add_argument("--encoder", default="bert-base-uncased")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=1)

    parser.add_argument("--base_url", default="API_KEY")
    parser.add_argument("--model_names", type=str, required=True)
    parser.add_argument("--model_temps", type=str, required=True)

    args = parser.parse_args()
    args.model_names = eval(args.model_names)
    args.model_temps = eval(args.model_temps)

    assert len(args.model_names) == len(args.model_temps)

    main(args)
