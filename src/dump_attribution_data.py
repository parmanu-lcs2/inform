import os
import glob
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

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

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    dataset_name, dataset_cfg = DATASET_MAP[args.task]
    dataset = load_dataset(
        dataset_name,
        dataset_cfg,
        split=f"validation[:{args.num_samples}]" if args.task == "mmlu"
        else f"test[:{args.num_samples}]"
    )

    shared_tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    shared_encoder = AutoModel.from_pretrained(args.encoder).to(DEVICE)
    shared_encoder.eval()

    models = [
        OpenAI(
            base_url=args.base_url,
            api_key="API_KEY",
        )
        for _ in args.model_names
    ]

    checkpoints = sorted(
        glob.glob(os.path.join(args.ckpt_dir, "collaboration_controller_*epoch*.pt"))
    )
    if not checkpoints:
        raise RuntimeError("No checkpoints found.")

    print(f"[info] Found {len(checkpoints)} checkpoints")

    for ckpt_path in checkpoints:
        epoch_id = ckpt_path.split("epoch")[-1].split("_")[0]
        print(f"\n[epoch {epoch_id}] {ckpt_path}")

        controller, _ = CollaborationController.from_saved_state(
            ckpt_path, device=DEVICE
        )
        controller.eval()

        grad_attr = []
        route_attr = []

        for step in tqdm(range(0, len(dataset), args.batch_size)):
            batch = dataset.select(
                range(step, min(step + args.batch_size, len(dataset)))
            )
            prompts = [format_prompt(ex, args.task) for ex in batch]

            with torch.no_grad():
                input_emb = get_shared_embedding(
                    prompts, shared_encoder, shared_tokenizer
                ).to(DEVICE)

            shared_reps = []
            for model, model_name, temp in zip(
                models, args.model_names, args.model_temps
            ):
                outputs = generate_text(
                    model,
                    tokenizer=None,
                    prompts=prompts,
                    infer_start=True,
                    model_name=model_name,
                    temperature=temp
                )
                rep = get_shared_embedding(
                    outputs, shared_encoder, shared_tokenizer
                )
                shared_reps.append(rep)

            shared_reps = torch.stack(shared_reps, dim=1).to(DEVICE)
            shared_reps.requires_grad_(True)  
            C_soft, seq, _ = controller(input_emb, shared_reps)

            incoming_mass = C_soft.sum(dim=1) 
            route_attr.append(incoming_mass.detach().cpu())

            batch_grad_scores = []

            for i in range(seq.shape[1]):
                controller.zero_grad(set_to_none=True)

                loss = seq[:, i].sum()
                loss.backward(retain_graph=True)

                grad = shared_reps.grad[:, i, :]    
                score = grad.norm(dim=-1)           
                batch_grad_scores.append(score.detach().cpu())

            shared_reps.grad.zero_()

            batch_grad_scores = torch.stack(batch_grad_scores, dim=1)
            grad_attr.append(batch_grad_scores)

        save_path = os.path.join(
            args.out_dir, f"{args.task}_epoch{epoch_id}.pt"
        )

        torch.save({
            "grad_attr": torch.cat(grad_attr, dim=0),    # [S, N]
            "route_attr": torch.cat(route_attr, dim=0),  # [S, N]
        }, save_path)

        print(f"[saved] {save_path}")

    print("\n[DONE] Attribution data dumped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", required=True,
                        choices=["mmlu", "gsm8k", "humaneval"])
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--out_dir", default="attribution_data")
    parser.add_argument("--encoder", default="bert-base-uncased")

    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)

    # Hosted model config
    parser.add_argument("--base_url", default="BASE_URL")
    parser.add_argument("--model_names", type=str, required=True)
    parser.add_argument("--model_temps", type=str, required=True)

    args = parser.parse_args()
    args.model_names = eval(args.model_names)
    args.model_temps = eval(args.model_temps)

    assert len(args.model_names) == len(args.model_temps)

    main(args)
