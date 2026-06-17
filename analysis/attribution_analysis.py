import os
import torch
import argparse
import pandas as pd
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

def compute_expert_attention_mass(attn_weights):
    """
    attn_weights: [B, H, N, N]
    returns: [B, N] attention mass per expert
    """
    incoming = attn_weights.sum(dim=2)      # [B, H, N]
    incoming = incoming.mean(dim=1)         # average heads → [B, N]
    incoming = incoming / incoming.sum(dim=-1, keepdim=True)
    return incoming

class AttentionInstrumentedController(CollaborationController):
    def forward(self, input_embedding, shared_reps=None, oracle_emb=None):
        h = self.input_proj(input_embedding)

        self.expert_attn_weights = None 

        if shared_reps is not None:
            B, N, D = shared_reps.shape
            shared_reps_proj = self.input_proj(
                shared_reps.view(B * N, D)
            ).view(B, N, -1)

            attn_out, attn_weights = self.model_attention(
                shared_reps_proj,
                shared_reps_proj,
                shared_reps_proj,
                need_weights=True,
                average_attn_weights=False
            )
            self.expert_attn_weights = attn_weights.detach()

            shared_reps_proj = self.norm1(shared_reps_proj + attn_out)

            ffn_out = self.ffn(shared_reps_proj)
            shared_reps_proj = self.norm2(shared_reps_proj + ffn_out)

            attn_out_res = self.residual_proj(shared_reps_proj)
            shared_reps = shared_reps + attn_out_res

            shared_reps_hidden = shared_reps_proj

        if shared_reps is not None:
            sim_matrix = self.compute_cosine_matrix(shared_reps_hidden) \
                if self.use_cosine_bias else None

            C_soft = self.compute_collaboration_matrix(
                h, sim_matrix, shared_reps_hidden
            )
            seq_gumbel = self.compute_sequence_distribution(
                h, C_soft, shared_reps_hidden
            )
        else:
            C_soft = self.compute_collaboration_matrix(h, None, None)
            seq_gumbel = self.compute_sequence_distribution(h, C_soft, None)

        return C_soft, seq_gumbel, {
            "selection_probs": seq_gumbel.detach().cpu().numpy()
        }


CHECKPOINT = None
TASK = None
BATCH_SIZE = 2
NUM_SAMPLES = 20
OUT_DIR = None

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

MODEL_TEMPS = [
    0.000008, 0.2, 0.5,
    0.5, 0.000008, 1.0,
    0.000008, 1.5, 2.0,
    0.000008
]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", choices=["train", "inference", "single"], required=True)
  parser.add_argument("--checkpoint", default="checkpoints/collab_matrix.pt")
  parser.add_argument("--output_dir", default="checkpoints/")
  parser.add_argument("--strategy", choices=["top1", "weighted_vote", "refinement_chain"], default="top1")
  parser.add_argument("--dataset", type=str, default="squad")
  parser.add_argument("--task", type=str, choices=["squad", "mmlu", "humaneval", "gsm8k", "mmlupro"], default="squad")
  parser.add_argument("--refine_with", choices=["full", "last"], default="last",
                  help="Mode for refinement chain: full history or just last model output")
  parser.add_argument("--logfile", type=str, default="multi_slm_collab.log")
  parser.add_argument("--batch_size", type=int, default=2)
  parser.add_argument("--num_experts", type=int, default=3, help="Number of expert models")
  args = parser.parse_args()

  os.makedirs(OUT_DIR, exist_ok=True)

  ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

  controller = AttentionInstrumentedController(
      input_dim=ckpt["config"]["input_dim"],
      hidden_dim=ckpt["config"]["hidden_dim"],
      num_models=ckpt["config"]["num_models"],
      max_seq_len=ckpt["config"]["max_seq_len"],
  ).to(DEVICE)

  controller.load_state_dict(ckpt["state_dict"])
  controller.eval()

  shared_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  shared_encoder = AutoModel.from_pretrained("bert-base-uncased").to(DEVICE)

  dataset = load_dataset(args.dataset, "all", split=f"validation[:{NUM_SAMPLES}]")

  models = [
      OpenAI(
          base_url="BASE_URL",
          api_key=os.environ["OPENAI_API_KEY"]
      )
      for _ in MODEL_NAMES
  ]

  records = []

  def attention_mass(attn):
      inc = attn.sum(dim=2).mean(dim=1)
      return inc / inc.sum(dim=-1, keepdim=True)

  for step in tqdm(range(0, len(dataset), BATCH_SIZE)):
      batch = dataset.select(range(step, min(step + BATCH_SIZE, len(dataset))))
      prompts = [format_prompt(ex, TASK) for ex in batch]

      input_emb = get_shared_embedding(prompts, shared_encoder, shared_tokenizer)

      shared_reps = []
      for m, name, t in zip(models, MODEL_NAMES, MODEL_TEMPS):
          outs = generate_text(
              m, None, prompts,
              infer_start=True,
              model_name=name,
              temperature=t
          )
          shared_reps.append(
              get_shared_embedding(outs, shared_encoder, shared_tokenizer)
          )

      shared_reps = torch.stack(shared_reps, dim=1).to(DEVICE)

      with torch.no_grad():
          _, seq, _ = controller(input_emb, shared_reps)
          attn = controller.expert_attn_weights

      mass = attention_mass(attn)

      for b in range(mass.size(0)):
          for i in range(mass.size(1)):
              records.append({
                  "sample": step + b,
                  "expert": i,
                  "model": MODEL_NAMES[i],
                  "temperature": MODEL_TEMPS[i],
                  "attention_mass": mass[b, i].item(),
                  "selection_prob": seq[b, i].item()
              })

  df = pd.DataFrame(records)
  df.to_csv(f"{OUT_DIR}/expert_attention.csv", index=False)
  print("Saved:", f"{OUT_DIR}/expert_attention.csv")


