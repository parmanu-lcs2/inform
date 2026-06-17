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
            api_key=os.environ["OPENAI_API_KEY"],
        )
        for _ in args.model_names
    ]
    tokenizers = [None] * len(args.model_names)

    checkpoints = sorted(
        glob.glob(os.path.join(args.ckpt_dir, "collaboration_controller_*epoch*.pt"))
    )
    print(checkpoints)
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

        epoch_C, epoch_seq = [], []

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

                C_soft, seq, _ = controller(
                    input_emb, shared_reps
                )

            epoch_C.append(C_soft.cpu())
            epoch_seq.append(seq.cpu())

        save_path = os.path.join(
            args.out_dir, f"{args.task}_epoch{epoch_id}.pt"
        )
        torch.save({
            "C": torch.cat(epoch_C, dim=0),
            "seq": torch.cat(epoch_seq, dim=0)
        }, save_path)

        print(f"[saved] {save_path}")

    print("\n[DONE] Emergence data dumped with shared reps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", required=True,
                        choices=["mmlu", "gsm8k", "humaneval"])
    parser.add_argument("--ckpt_dir", required=True,
                        help="Directory containing epoch checkpoints")
    parser.add_argument("--out_dir", default="emergence_data")
    parser.add_argument("--encoder", default="bert-base-uncased")

    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)

    # Hosted model config
    parser.add_argument("--base_url", default="BASE_URL")
    parser.add_argument("--model_names", type=str, required=True,
                        help="Python list of model names")
    parser.add_argument("--model_temps", type=str, required=True,
                        help="Python list of temperatures")

    args = parser.parse_args()
    args.model_names = eval(args.model_names)
    args.model_temps = eval(args.model_temps)

    assert len(args.model_names) == len(args.model_temps)

    main(args)
