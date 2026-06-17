import os
import re
import csv
import re
import time
import httpx
import torch
import random
import datetime
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from openai import OpenAI
from tqdm.auto import tqdm
from rapidfuzz import fuzz
import coloredlogs, logging
import torch.nn.functional as F
from datasets import load_dataset
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
from transformers import get_cosine_schedule_with_warmup
from concurrent.futures import ThreadPoolExecutor, as_completed
from human_eval.execution import check_correctness
from human_eval.data import read_problems
from human_eval.evaluation import evaluate_functional_correctness, estimate_pass_at_k
torch.manual_seed(42)

logger = None

os.environ['HF_TOKEN'] = 'HF_TOKEN'
proxy_url = "HTTPS_PROXY"

http_client = httpx.Client(
    transport=httpx.HTTPTransport(proxy=proxy_url, verify=False),
    verify=False
)

print("SSL DISABLE")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERT_TYPES = 'hosted'

ORACLE_TYPE = 'hosted'
ORACLE_NAME = "openai/gpt-oss-20b"

SHARED_ENCODER_NAME = "bert-base-uncased"
BATCH_SIZE = 2
LAMBDA_SYMM = 0.25
LAMBDA_SPARSITY = 0.01
NUM_SHOTS = 3
USE_QUANTIZED = True
MAX_NEW_TOKENS = 1024

class SelectionLogger:
    def __init__(self, model_names, log_dir="selection_logs"):
        self.model_names = model_names
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.train_log_path = os.path.join(log_dir, f"train_selections_{timestamp}.csv")
        self.inference_log_path = os.path.join(log_dir, f"inference_selections_{timestamp}.csv")

        with open(self.train_log_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'sample_idx', 'model_name', 'selection_prob', 'phase'])

        with open(self.inference_log_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_idx', 'selected_models', 'selection_probs', 'final_model'])

    def log_batch(self, batch_indices, selection_probs, phase='train', selected_indices=None):
        if phase == 'train':
            with open(self.train_log_path, 'a') as f:
                writer = csv.writer(f)
                for i, sample_idx in enumerate(batch_indices):
                    for model_idx, prob in enumerate(selection_probs[i]):
                        writer.writerow([sample_idx, self.model_names[model_idx], prob, phase])
        else:
            with open(self.inference_log_path, 'a') as f:
                writer = csv.writer(f)
                for i, sample_idx in enumerate(batch_indices):
                    selected = selected_indices[i]
                    selected_names = [self.model_names[idx] for idx in selected]
                    probs = selection_probs[i]
                    writer.writerow([sample_idx, '|'.join(selected_names), '|'.join(map(str, probs)), selected_names[-1]])

    def get_selection_stats(self):
        stats = {'total_selections': {name: 0 for name in self.model_names},
                 'average_prob': {name: 0.0 for name in self.model_names}}

        with open(self.train_log_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats['total_selections'][row['model_name']] += 1
                stats['average_prob'][row['model_name']] += float(row['selection_prob'])

        for name in self.model_names:
            if stats['total_selections'][name] > 0:
                stats['average_prob'][name] /= stats['total_selections'][name]

        return stats

class CollaborationDataset(Dataset):
    def __init__(self, args, samples, shared_encoder, shared_tokenizer):
        self.samples = samples
        self.shared_encoder = shared_encoder
        self.shared_tokenizer = shared_tokenizer
        self.args = args

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.args.task == 'humaneval':
            return {
                'prompt': sample['prompt'],
                'answer': sample['answer'],
                'task_id': sample['task_id']
            }
        return {
            'prompt': sample['prompt'],
            'answer': sample['answer'],
        }

class CollaborationController(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_models=3, max_seq_len=3, tau=1.0, use_cosine_bias=True):
        super().__init__()
        self.num_models = num_models
        self.max_seq_len = max_seq_len
        self.tau = tau
        self.use_cosine_bias = use_cosine_bias
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.oracle_proj = nn.Linear(input_dim, hidden_dim)
        self.oracle_attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)

        self.model_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.residual_proj = nn.Linear(hidden_dim, input_dim)

        self.collab_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim)
        )

        self.W_Query = nn.Linear(hidden_dim, hidden_dim) 
        self.W_Key   = nn.Linear(hidden_dim, hidden_dim) 
        
        self.prior_scale = nn.Parameter(torch.tensor(1.0))

        self.seq_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_models)
        )

        for layer in [self.input_proj, self.residual_proj, *self.seq_head, *self.collab_ffn]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)
        
        self.perf_estimator = nn.Linear(hidden_dim, 1)
        
        self.tau_min = 0.5
        self.tau_max = 2.0
        self.tau_decay = 0.999
        self.min_k = 1
        self.current_k = max_seq_len
        self.k_decay = 0.999
        self.k_update_freq = 100
        self.step_count = 0
        self.register_buffer('selection_quality', torch.zeros(1))
        self.quality_alpha = 0.9
        self.length_cost_weight = 0.2
        self.min_length_penalty = 0.1

    def compute_collaboration_matrix(self, encoded_input, sim_matrix=None, shared_reps=None):
        C_logits = torch.randn(
            self.num_models, self.num_models, device=DEVICE
        )
        
        C_logits = C_logits * 10.0 

        if self.use_cosine_bias and sim_matrix is not None:
            C_logits = C_logits + (self.prior_scale * sim_matrix * 5.0)

        mask = torch.eye(self.num_models, device=C_logits.device).bool()
        C_logits = C_logits.masked_fill(mask, float('-inf'))
        
        sm = F.softmax(C_logits, dim=-1)
        return sm


    def compute_cosine_matrix(self, shared_reps):
        B, N, D = shared_reps.shape
        norm_reps = F.normalize(shared_reps, dim=-1)
        return torch.einsum("bid,bjd->bij", norm_reps, norm_reps)

    def compute_sequence_distribution(self, encoded_input, C_soft, shared_reps):
        logits = self.seq_head(encoded_input)

        baseline = torch.ones_like(logits) * 0.1
        logits = logits + baseline

        if shared_reps is not None:
            perf_scores = self.perf_estimator(shared_reps).squeeze(-1)
            logits = logits + perf_scores

        if C_soft is not None:
            collab_importance = 0.5 * (C_soft.mean(dim=1) + C_soft.mean(dim=2))
            logits = logits + collab_importance

        length_penalty = torch.arange(1, self.num_models+1, device=logits.device).float() * self.length_cost_weight
        length_penalty = length_penalty + self.min_length_penalty 
        logits = logits - length_penalty.unsqueeze(0) 

        probs = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)
        probs = probs.clamp(min=1e-4) 
        return probs / probs.sum(dim=-1, keepdim=True) 

    def update_k(self, new_quality):
        self.step_count += 1
        self.selection_quality = (self.quality_alpha * self.selection_quality +
                                (1 - self.quality_alpha) * new_quality)

        if self.step_count % self.k_update_freq == 0:
            if self.selection_quality > 0.7:
                self.current_k = max(self.min_k, int(self.current_k * self.k_decay))

    def forward(self, input_embedding, shared_reps=None, oracle_emb=None):
        h = self.input_proj(input_embedding)

        if shared_reps is not None:
            B, N, D = shared_reps.shape
            shared_reps_proj = self.input_proj(shared_reps.view(B*N, D)).view(B, N, -1)

            attn_out, _ = self.model_attention(shared_reps_proj, shared_reps_proj, shared_reps_proj)
            shared_reps_proj = self.norm1(shared_reps_proj + attn_out)
            
            ffn_out = self.ffn(shared_reps_proj)
            shared_reps_proj = self.norm2(shared_reps_proj + ffn_out) 

            attn_out_res = self.residual_proj(shared_reps_proj)
            shared_reps = shared_reps + attn_out_res

            shared_reps_hidden = shared_reps_proj 

        if shared_reps is not None:
            sim_matrix = self.compute_cosine_matrix(shared_reps_hidden) if self.use_cosine_bias else None
            C_soft = self.compute_collaboration_matrix(h, sim_matrix)
            seq_gumbel = self.compute_sequence_distribution(h, C_soft, shared_reps_hidden)
        else:
            C_soft = self.compute_collaboration_matrix(h, None, None)
            seq_gumbel = self.compute_sequence_distribution(h, C_soft, None)

        print(C_soft, seq_gumbel)

        if not self.training:
            topk_values, topk_indices = torch.topk(seq_gumbel, k=self.current_k, dim=1)
            seq_gumbel = torch.zeros_like(seq_gumbel).scatter(1, topk_indices, topk_values)

        self.tau = max(self.tau_min, self.tau * self.tau_decay)

        with torch.no_grad():
            if self.training:
                selected_counts = (seq_gumbel > 0.5).sum(dim=1).float().mean()
                selected_lengths = (seq_gumbel > 0.5).sum(dim=1).float()
                selection_info = {
                    'selected_counts': selected_counts.item(),
                    'selection_probs': seq_gumbel.detach().cpu().numpy()
                }
                selection_info['length_cost'] = selected_lengths.mean().item() * self.length_cost_weight
            else:
                topk_values, topk_indices = torch.topk(seq_gumbel, k=self.max_seq_len, dim=1)
                selection_info = {
                    'selected_indices': topk_indices.cpu().numpy(),
                    'selection_probs': topk_values.cpu().numpy()
                }
                selection_info['length_cost'] = self.current_k * self.length_cost_weight

        return C_soft, seq_gumbel, selection_info

    @classmethod
    def from_saved_state(cls, save_path, device=DEVICE):
        checkpoint = torch.load(save_path, map_location=device)

        controller = cls(
            input_dim=checkpoint['config']['input_dim'],
            hidden_dim=checkpoint['config']['hidden_dim'],
            num_models=checkpoint['config']['num_models'],
            max_seq_len=checkpoint['config']['max_seq_len'],
            tau=checkpoint['tau'],  
            use_cosine_bias=True
        ).to(device)

        controller.load_state_dict(checkpoint['state_dict'])
        controller.current_k = checkpoint['current_k']
        controller.tau = checkpoint['tau']
        controller.selection_quality = torch.tensor(checkpoint['selection_quality'], device=device)
        controller.step_count = checkpoint['step_count']

        for param, value in checkpoint['config'].items():
            setattr(controller, param, value)

        return controller, checkpoint

def prepare_datasets(args):
    if args.task == "mmlu":
        dataset = load_dataset(args.dataset, 'all', split="validation")
    elif args.task == "mmlupro":
        dataset = load_dataset(args.dataset, split="validation")
    elif args.task == "gsm8k":
        dataset = load_dataset(args.dataset, 'main', split="test")
    else:
        dataset = load_dataset(args.dataset, split="test")

    samples = []
    for item in dataset:
        if args.task == 'squad':
            prompt = construct_few_shot_prompt(0, dataset) + f"Q: {item['question']}\nContext: {item['context']}\nA:"
            answer = item["answers"]["text"][0] if item["answers"]["text"] else ""
        elif args.task == 'mmlu':
            prompt = format_prompt(item, 'mmlu')
            answer = ["(A)", "(B)", "(C)", "(D)"][item["answer"]] + " " + item["choices"][item["answer"]]
        elif args.task == 'mmlupro':
            prompt = format_prompt(item, 'mmlupro')
            answer = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)"][item["answer_index"]] + " " + item["options"][item["answer_index"]]
        elif args.task == 'humaneval':
            prompt = format_prompt(item, 'humaneval')
            answer = item["canonical_solution"]
        elif args.task == 'gsm8k':
            prompt = format_prompt(item, 'gsm8k')
            answer = item["answer"]
        else:
            raise ValueError("Unsupported task")

        if args.task == 'humaneval':
            samples.append({
                'prompt': prompt,
                'answer': answer,
                'task_id': item['task_id']
            })
        else:
            samples.append({'prompt': prompt, 'answer': answer})

    return samples

def extract_oracle_signals(oracle_model, oracle_tokenizer, prompts, shared_encoder, shared_tokenizer, models, tokenizers):
    B = len(prompts)
    N = len(MODEL_NAMES)

    oracle_outputs = generate_text(oracle_model, oracle_tokenizer, prompts)
    oracle_emb = get_shared_embedding(oracle_outputs, shared_encoder, shared_tokenizer)

    model_outputs = []
    for name, model, tokenizer in zip(MODEL_NAMES, models, tokenizers):
        outputs = generate_text(model, tokenizer, prompts, name)
        model_outputs.append(outputs)

    oracle_matrix = torch.zeros(B, N, N, device=DEVICE)
    importance_scores = torch.zeros(B, N, device=DEVICE)

    for i in range(N):
        model_emb = get_shared_embedding(model_outputs[i], shared_encoder, shared_tokenizer)
        importance_scores[:, i] = F.cosine_similarity(oracle_emb, model_emb, dim=1)

        for j in range(N):
            if i != j:
                other_emb = get_shared_embedding(model_outputs[j], shared_encoder, shared_tokenizer)
                combined_emb = (model_emb + other_emb)/2
                oracle_matrix[:,i,j] = F.cosine_similarity(combined_emb, oracle_emb, dim=1)

    return F.softmax(oracle_matrix, dim=-1), F.softmax(importance_scores, dim=-1)

def train_controller(args):
    logging.info(f"Starting training with lambda_symm={args.lambda_symm}, lambda_sparse={args.lambda_sparse}, lambda_oracle={args.lambda_oracle}")
    if EXPERT_TYPES == 'hosted':
        models = [
            OpenAI(
                base_url="BASE_URL",
                api_key="API_KEY",
                http_client=http_client
            ) for _ in MODEL_NAMES
        ]
        tokenizers = [None] * len(MODEL_NAMES) 
    else:
        models = [load_model_quantized(name) if USE_QUANTIZED else AutoModelForCausalLM.from_pretrained(name).to(DEVICE) for name in MODEL_NAMES]
        tokenizers = [AutoTokenizer.from_pretrained(name) for name in MODEL_NAMES]
        for t in tokenizers:
            t.pad_token = t.eos_token
            t.padding_side = 'left'

    if ORACLE_TYPE == 'hosted':
        oracle_tokenizer = None
        oracle_model = OpenAI(
                base_url="BASE_URL",
                api_key="API_KEY",
                http_client=http_client

        )
    else:
        oracle_tokenizer = AutoTokenizer.from_pretrained(ORACLE_NAME)
        oracle_tokenizer.pad_token = oracle_tokenizer.eos_token
        oracle_tokenizer.padding_side = 'left'
        oracle_model = load_model_quantized(ORACLE_NAME) if USE_QUANTIZED else AutoModelForCausalLM.from_pretrained(ORACLE_NAME).to(DEVICE)

    shared_tokenizer = AutoTokenizer.from_pretrained(SHARED_ENCODER_NAME)
    shared_encoder = AutoModel.from_pretrained(SHARED_ENCODER_NAME).to(DEVICE)

    samples = prepare_datasets(args)
    dataset = CollaborationDataset(args, samples, shared_encoder, shared_tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    controller = CollaborationController(num_models=len(models), max_seq_len=len(models)-1, use_cosine_bias=True).to(DEVICE)
    optimizer = torch.optim.Adam(controller.parameters(), lr=1e-3)

    num_training_steps = len(dataloader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    loss_weights = {
        'util': 0.5,
        'distill': 0.5,
        'symm': args.lambda_symm,
        'sparse': args.lambda_sparse,
        'diversity': 0.1,
        'oracle': args.lambda_oracle,
        'selection': 1.0,
        'length': 0.5
    }
    selection_logger = SelectionLogger(MODEL_NAMES)

    for epoch in range(args.epochs):
        controller.train()
        epoch_loss = 0.0
        epoch_C_accum = torch.zeros(len(models), len(models), device='cpu')
        epoch_sample_count = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            prompts = batch['prompt']

            if args.task == 'humaneval':
                golds = [{'prompt': p, 'answer': a, 'task_id': i} for (p, a, i) in zip(batch['prompt'], batch['answer'], batch['task_id'])]
            else:
                golds = batch['answer']

            with torch.no_grad():
                input_embeds = get_shared_embedding(prompts, shared_encoder, shared_tokenizer)

                if EXPERT_TYPES == 'hosted':
                    shared_reps = torch.stack([
                        get_shared_embedding(generate_text(m, t, prompts, n), shared_encoder, shared_tokenizer)
                        for (n, m, t) in zip(MODEL_NAMES, models, tokenizers)
                    ], dim=1).to(DEVICE)
                else:
                    shared_reps = torch.stack([
                        get_shared_embedding(generate_text(m, t, prompts), shared_encoder, shared_tokenizer)
                        for (m, t) in zip(models, tokenizers)
                    ], dim=1).to(DEVICE)

                oracle_matrix, importance_scores = extract_oracle_signals(
                    oracle_model, oracle_tokenizer, prompts, shared_encoder, shared_tokenizer, models, tokenizers
                )
                oracle_emb = get_shared_embedding(
                    generate_text(oracle_model, oracle_tokenizer, prompts),
                    shared_encoder, shared_tokenizer
                ).to(DEVICE)

            C_soft, seq_gumbel, selection_info = controller(input_embeds, shared_reps, oracle_emb)
            with torch.no_grad():
                epoch_C_accum += C_soft.sum(dim=0).detach().cpu()
                epoch_sample_count += C_soft.size(0)

            selection_logger.log_batch(
                batch_indices=batch_idx * args.batch_size + torch.arange(len(prompts)),
                selection_probs=selection_info['selection_probs'],
                phase='train'
            )

            if batch_idx % 2 == 0:
                logger.info(f"Selection probs: {seq_gumbel.mean(dim=0).detach().cpu().numpy()}")
                logger.info(f"Max prob: {seq_gumbel.max().item():.4f}, Min prob: {seq_gumbel.min().item():.4f}")

            oracle_alignment_loss = F.mse_loss(C_soft, oracle_matrix) + F.mse_loss(seq_gumbel, importance_scores)
            batch_qualities = []
            final_outputs = []
            distillation_loss = 0
            selected_counts = torch.zeros(len(models), device=DEVICE)

            for b in range(len(prompts)):
                current_prompt = prompts[b]
                reasoning = ""

                topk_probs, topk_indices = torch.topk(seq_gumbel[b], k=min(args.seq_len, seq_gumbel[b].size(0)))
                selected_counts.scatter_add_(0, topk_indices, topk_probs)

                oracle_emb = get_shared_embedding(
                    generate_text(oracle_model, oracle_tokenizer, [current_prompt])[0],
                    shared_encoder, shared_tokenizer
                ).to(DEVICE)

                model_outputs = []
                model_shared_reps = []
                for k in range(min(args.seq_len, len(topk_indices))):
                    selected_idx = topk_indices[k].item()
                    model = models[selected_idx]

                    full_input = reasoning + "\n" + current_prompt if reasoning else current_prompt

                    if EXPERT_TYPES != 'hosted':
                        tokenizer = tokenizers[selected_idx]
                        full_input = tokenizer.apply_chat_template([{'role': 'user', 'content': full_input}], tokenize=False, add_generation_prompt=False)
                    else:
                        tokenizer = None

                    logger.debug(full_input)

                    out = generate_text(model, tokenizer, [full_input], False, MODEL_NAMES[selected_idx])[0]

                    model_outputs.append(out.strip())
                    reasoning += f"\nAssistant {k+1}'s Response: {out}"

                    out_shared_rep = get_shared_embedding(out, shared_encoder, shared_tokenizer)
                    model_shared_reps.append(out_shared_rep)
                    out_shared_rep_avg = out_shared_rep.mean(dim=0)
                    oracle_emb_avg = oracle_emb.mean(dim=0) 

                    distillation_loss += F.mse_loss(out_shared_rep_avg, oracle_emb_avg)

                similarities = [
                    F.cosine_similarity(
                        oracle_emb.mean(dim=0).unsqueeze(0),
                        e.mean(dim=0).unsqueeze(0), 
                        dim=1
                    )
                    for e in model_shared_reps
                ]
                batch_quality = torch.mean(torch.stack(similarities))
                batch_qualities.append(batch_quality)

                final_output = model_outputs[-1] if model_outputs else ""
                final_outputs.append(final_output)

            avg_quality = torch.mean(torch.stack(batch_qualities)).item()
            controller.update_k(avg_quality)

            util_loss = compute_utility_loss(final_outputs, golds, args.task)
            symm_loss = (C_soft - C_soft.transpose(1, 2)).abs().mean()
            epsilon = 1e-8
            entropy = -torch.sum(C_soft * torch.log(C_soft + epsilon), dim=-1)
            sparse_loss = entropy.mean()
            diversity_loss = -selected_counts.var() / len(models)
            selection_loss = -seq_gumbel.mean()  # Encourage higher probabilities
            length_cost = selection_info['length_cost']

            total_loss = (
                loss_weights['util'] * util_loss +
                loss_weights['distill'] * distillation_loss +
                loss_weights['symm'] * symm_loss +
                loss_weights['sparse'] * sparse_loss +
                loss_weights['diversity'] * diversity_loss +
                loss_weights['oracle'] * oracle_alignment_loss +
                loss_weights['selection'] * selection_loss +
                loss_weights['length'] * length_cost
            )

            
            # log all loss components and total loss as a dict
            logger.info({
                'k': controller.current_k,
                'util_loss': util_loss,
                'distill_loss': distillation_loss,
                'symm_loss': symm_loss,
                'sparse_loss': sparse_loss,
                'diversity_loss': diversity_loss,
                'oracle_alignment_loss': oracle_alignment_loss,
                'selection_loss': selection_loss,
                'length_cost': length_cost,
                'total_loss': total_loss.item()
            })

            epoch_loss += total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()

            # Gradient monitoring
            grad_norms = [p.grad.norm().item() for p in controller.parameters() if p.grad is not None]
            logger.info(f"Gradient norms - Min: {min(grad_norms):.4f}, Max: {max(grad_norms):.4f}, Mean: {np.mean(grad_norms):.4f}")

            torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=15.0)
            optimizer.step()
            scheduler.step()

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1} | Batch {batch_idx} | Loss={total_loss.item():.4f} | "
                    f"Util={util_loss:.4f} | OracleAlign={oracle_alignment_loss:.4f} | "
                    f"Selection={selection_loss:.4f}"
                )

        # End of epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} completed | Avg Loss: {avg_epoch_loss:.4f}")
        avg_C_matrix = epoch_C_accum / epoch_sample_count
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the matrix tensor
        matrix_save_path = f"{args.output_dir}/avg_C_matrix_{args.task}_epoch{epoch+1}_{timestamp}.pt"
        torch.save(avg_C_matrix, matrix_save_path)
        logger.info(f"Saved average Collaboration Matrix to {matrix_save_path}")
        controller_state = {
            'state_dict': controller.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'current_k': controller.current_k,
            'tau': controller.tau,
            'selection_quality': controller.selection_quality,
            'step_count': controller.step_count,
            'epoch': epoch + 1,
            'loss_weights': loss_weights,
            'model_names': MODEL_NAMES,
            'config': {
                'input_dim': controller.input_dim,
                'hidden_dim': controller.hidden_dim,
                'num_models': controller.num_models,
                'max_seq_len': controller.max_seq_len,
                'tau_min': controller.tau_min,
                'tau_max': controller.tau_max,
                'min_k': controller.min_k,
                'k_decay': controller.k_decay,
                'k_update_freq': controller.k_update_freq,
                'quality_alpha': controller.quality_alpha,
                'length_cost_weight': controller.length_cost_weight,
                'min_length_penalty': controller.min_length_penalty
            }
        }

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{args.output_dir}/collaboration_controller_{args.task}_epoch{epoch+1}_{timestamp}.pt"
        torch.save(controller_state, save_path)

    final_save_path = f"{args.output_dir}/collaboration_controller_{args.task}_final_{timestamp}.pt"
    torch.save(controller_state, final_save_path)
    logger.info(f"Saved final controller state to {final_save_path}")

def clean_and_extract_number(s):
    # Remove all punctuation and keep only digits
    s = s.replace(',', '')
    s = s.split()[0]
    s = re.sub(r"[^\d]", "", s)

    return int(s) if s else None

def extract_final_number(text):
    """
    Extracts the final numeric value from a string.
    Handles cases like:
    - "29-1=<<29-1=28>>28 years old."
    - "The answer is: $840$."
    - "Therefore, the final answer is \\boxed{54}."
    """
    if pd.isna(text):
        return None
    text = str(text)

    # Priority 1: boxed or LaTeX-style final answer
    match = re.search(r'\\boxed\{(\d+(\.\d+)?)\}', text)
    if match:
        return float(match.group(1))

    # Priority 2: $number$ style
    match = re.search(r'\$(\d+(?:\.\d+)?)\$', text)
    if match:
        return float(match.group(1))

    # Priority 3: <<...=number>>number pattern
    match = re.findall(r'>>(\d+(?:\.\d+)?)', text)
    if match:
        return float(match[-1])

    # Fallback: get the last number in the string
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    return float(numbers[-1]) if numbers else None

def load_model_quantized(name):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.float16
    )
    return AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb_config, device_map="auto")

def compute_mmlu_loss(preds, golds):
    def index_to_option(idx):
        return f"({chr(ord('A') + int(idx))})"

    def extract_option_from_response(response):
        response = response.split('\n\n')[0]
        for opt in ['(A)', '(B)', '(C)', '(D)']:
            if opt in response.split():
                return opt
        return None
    correct = 0
    for pred, gold in zip(preds, golds):
        _pred = extract_option_from_response(pred)
        _gold = extract_option_from_response(gold)
        if _pred is not None and _gold is not None and _gold in _pred:
            correct += 1
    acc = correct / len(preds)
    return 1.0 - acc

def compute_gsm8k_loss(preds, golds):
    correct = 0
    for pred, gold in zip(preds, golds):
        pred_val = extract_final_number(pred)
        gold_val = extract_final_number(gold)
        if pred_val is not None and gold_val is not None and abs(pred_val - gold_val) < 1e-3:
            correct += 1
    acc = correct / len(preds)
    return 1.0 - acc
def clean_pred(pred: str) -> str:
    """Return cleaned Python code string with code-fence / <think> wrappers removed."""
    if pred is None:
        return ""
    s = pred.replace("\r\n", "\n")  # normalize newlines
    s = s.strip()

    # Remove a leading <think> ... </think> block (if present)
    s = re.sub(r'^\s*<think>.*?</think>\s*', '', s, flags=re.DOTALL)

    # Remove opening code fence: ``` or ```python or ``` py (allow optional newline after)
    s = re.sub(r'^\s*```(?:\s*\w+)?\s*\n?', '', s, flags=re.IGNORECASE)

    # Remove trailing closing fence: ```
    s = re.sub(r'\n?\s*```\s*$', '', s)

    # Also handle fence style with tildes ~~~ just in case
    s = re.sub(r'^\s*~~~(?:\s*\w+)?\s*\n?', '', s)
    s = re.sub(r'\n?\s*~~~\s*$', '', s)

    return s.strip()


def compute_humaneval_loss(preds, golds, k=1):
    print(golds[0])
    preds = [clean_pred(p) for p in preds]
    from human_eval.data import stream_jsonl
    import os

    evalset_file = os.path.join(
    "human-eval", "data", "HumanEval.jsonl.gz"
    )
    all_problems = {task["task_id"]: task for task in stream_jsonl(evalset_file)}
    task_ids_in_batch = {g["task_id"] for g in golds}
    problems = {tid: all_problems[tid] for tid in task_ids_in_batch if tid in all_problems}
    completions = {g['task_id']: [p] for g, p in zip(golds, preds)}
    timeout=3.0
    n_workers=4
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        from collections import Counter, defaultdict

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for task_id, completion in zip(completions.keys(), completions.values()):
            print(type(problems[task_id]))
            print(problems[task_id])
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for future in as_completed(futures):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))


    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result if r[1] is not None and "passed" in r[1]]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = [k]
    print(total)
    print(correct)
    results = {f"pass@{kk}": estimate_pass_at_k(total, correct, kk).mean()
                 for kk in ks if len(total) > 0 and np.all(total >= kk)}

    acc = results[f'pass@{k}']
    return 1.0 - acc

def fallback_token_loss(preds, golds, tokenizer):
    """
    Compute cross-entropy loss over tokens when exact-match is not possible.
    """
    input_ids = tokenizer(preds, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    target_ids = tokenizer(golds, padding=True, truncation=True, return_tensors="pt")["input_ids"]

    input_ids = input_ids[:, :target_ids.shape[1]].to(target_ids.device)
    loss = F.cross_entropy(input_ids.float(), target_ids, reduction='mean')
    return loss.item()

def compute_utility_loss(preds, golds, task_name, tokenizer=None, evaluator=None):
    """
    Task-specific utility loss: lower is better.
    """
    task_name = task_name.lower()

    if "mmlu" in task_name:
        return compute_mmlu_loss(preds, golds)

    elif "gsm8k" in task_name:
        return compute_gsm8k_loss(preds, golds)

    elif "humaneval" in task_name:
        return compute_humaneval_loss(preds, golds, 1)

    else:
        if tokenizer is None:
            raise ValueError("Tokenizer required for fallback token-level loss")
        return fallback_token_loss(preds, golds, tokenizer)

def generate_text(model, tokenizer, prompts, infer_start=False, model_name='openai/gpt-oss-20b', temperature=0.0):
    print(tokenizer, model_name)
    import time
    time.sleep(random.randint(0,10))
    if tokenizer is None or model_name is not None:
        responses = []
        for prompt in prompts:
            retry_count = 0
            while retry_count < 5:
                try:
                    response = model.chat.completions.create(
                        model=model_name,
                        max_tokens=512 if not infer_start else 30,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature
                    )
                    print(response.choices[0].message.content)
                    responses.append(response.choices[0].message.content.replace('Answer:', '').strip())
                    break
                except Exception as e:
                    print(e)
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    print(f"Retry {retry_count}/5 after error: {e}. Waiting {wait_time}s.")
                    time.sleep(10 * wait_time + random.randint(0, 10))
            else:
                print("[ERROR: Max retries exceeded]")
        return responses

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    outputs = [out[len(prompt):].strip() for out, prompt in zip(outputs, prompts)]
    return outputs

def get_shared_embedding(texts, encoder, tokenizer, device=DEVICE):
    if not texts or not all(isinstance(t, str) for t in texts):
        batch_size = len(texts) if isinstance(texts, list) else 1
        return torch.zeros(batch_size, encoder.config.hidden_size, device=device)

    cleaned_texts = []
    for text in texts:
        if not isinstance(text, str):
            text = str(text)
        cleaned_texts.append(text[:1000000])

    try:
        inputs = tokenizer(
            cleaned_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
            return_attention_mask=True
        ).to(device)

        if (inputs['input_ids'] >= tokenizer.vocab_size).any():
            invalid_ids = inputs['input_ids'][inputs['input_ids'] >= tokenizer.vocab_size]
            logger.warning(f"Found {len(invalid_ids)} invalid token IDs: {invalid_ids}")
            inputs['input_ids'][inputs['input_ids'] >= tokenizer.vocab_size] = tokenizer.unk_token_id

        if 'attention_mask' not in inputs:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

        with torch.no_grad():
            outputs = encoder(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]

            if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                logger.warning("Generated embeddings contain NaN/inf values")
                return torch.zeros(len(texts), encoder.config.hidden_size, device=device)

            return embeddings

    except Exception as e:
        print(f"Embedding failed: {str(e)}")
        return torch.zeros(len(texts), encoder.config.hidden_size, device=device)

def evaluate(preds, golds):
    em_total, f1_total = 0.0, 0.0
    for pred, gold in zip(preds, golds):
        pred, gold = pred.strip().lower(), gold.strip().lower()
        em = int(pred == gold)
        em_total += em
        pred_tokens = set(pred.split())
        gold_tokens = set(gold.split())
        common = pred_tokens & gold_tokens
        f1 = 2 * len(common) / (len(pred_tokens) + len(gold_tokens) + 1e-8)
        f1_total += f1
    return em_total / len(preds), f1_total / len(preds)

def construct_few_shot_prompt(current_idx, dataset, num_shots=3):
    few_shots = []
    random_indices = torch.randperm(len(dataset)).tolist()
    random_indices = random_indices[:num_shots]
    for i in random_indices:
        if i == current_idx:
            continue

        q, a = dataset[i]["question"], dataset[i]["answer"]
        r = a.split('####')[0].strip()
        a = a.split('####')[-1].strip()
        shot = f"#### Example\nQuestion {q}\nReason: {r}\nAnswer: {a}"
        few_shots.append(shot)
    return "\n\n".join(few_shots)

def run_batch_collaborative_inference_selected(
    prompts, task, controller, models, model_tokenizers, tokenizer, encoder,
    device="cuda", top_k=None, threshold=0.0, early_stopping=True
):
    B = len(prompts)
    N = len(models)

    input_emb = get_shared_embedding(prompts, encoder, tokenizer).to(device)

    if EXPERT_TYPES == "hosted":
        shared_reps = torch.stack([
            get_shared_embedding(generate_text(m, t, prompts, True, n), encoder, tokenizer)
            for (n,m,t) in zip(MODEL_NAMES, models, model_tokenizers)
        ], dim=1).to(device)
    else:
        shared_reps = torch.stack([
            get_shared_embedding(generate_text(m, t, prompts, True), encoder, tokenizer)
            for (m,t) in zip(models, model_tokenizers)
        ], dim=1).to(device)

    controller.eval()
    start_time = time.time()
    with torch.no_grad():
        C_soft, seq_gumbel, selection_info = controller(input_emb, shared_reps)

    final_outputs = []
    all_selected_models = []
    stopping_points = []

    for i in range(B):
        current_prompt = prompts[i]
        selected_models = []
        previous_output = None
        stop_early = False
        reasoning_chain = []

        # Get top-k models for this sample
        topk_values, topk_indices = torch.topk(seq_gumbel[i], k=controller.current_k)

        for k, model_idx in enumerate(topk_indices):
            if stop_early and early_stopping:
                break

            model_idx = model_idx.item()
            model = models[model_idx]
            tokenizer = model_tokenizers[model_idx]

            # Prepare input with previous reasoning if available
            full_input = current_prompt
            if reasoning_chain:
                full_input = "\n\n".join(reasoning_chain) + "\n\n" + current_prompt

            temp = args.model_temps[model_idx]

            output = generate_text(model, tokenizer, [full_input], False, MODEL_NAMES[model_idx], temperature=temp)[0]
            reasoning_chain.append(f"Assistant {k+1} (Model {model_idx}): {output}")

            if previous_output is not None and early_stopping:
                string_similarity = fuzz.ratio(output, previous_output)/100

                if string_similarity > 0.8:
                    logger.info(f"Early stopping at model {k+1} due to similar outputs")
                    logger.info(f"String similarity: {string_similarity:.2f}")
                    stop_early = True
                    stopping_points.append(k+1)
                    break

            previous_output = output
            selected_models.append(model_idx)

        final_output = previous_output if previous_output else ""
        final_outputs.append(final_output)
        all_selected_models.append(selected_models)

    if early_stopping and stopping_points:
        avg_stopping_point = sum(stopping_points)/len(stopping_points)
        logger.info(f"Early stopping triggered {len(stopping_points)} times")
        logger.info(f"Average stopping point: {avg_stopping_point:.1f} models")
    else:
        pass

    end_time = time.time()
    elapsed_time = (end_time - start_time) / B
    logger.info(f"Total inference time: {elapsed_time:.2f} seconds")
    controller.last_C_soft = C_soft.detach()

    return final_outputs, selection_info, all_selected_models, stopping_points if early_stopping else None, elapsed_time

def run_batch_collaborative_inference_static_C(
    prompts, task, controller, models, model_tokenizers, tokenizer, encoder,
    device="cuda", top_k=None, threshold=0.0, early_stopping=True
):
    B = len(prompts)
    N = len(models)

    input_emb = get_shared_embedding(prompts, encoder, tokenizer).to(device)

    if EXPERT_TYPES == "hosted":
        shared_reps = torch.stack([
            get_shared_embedding(generate_text(m, t, prompts, True, n), encoder, tokenizer)
            for (n,m,t) in zip(MODEL_NAMES, models, model_tokenizers)
        ], dim=1).to(device)
    else:
        shared_reps = torch.stack([
            get_shared_embedding(generate_text(m, t, prompts, True), encoder, tokenizer)
            for (m,t) in zip(models, model_tokenizers)
        ], dim=1).to(device)

    controller.eval()
    start_time = time.time()
    with torch.no_grad():
        C_soft, seq_gumbel, selection_info = controller(input_emb, shared_reps)

    final_outputs = []
    all_selected_models = []
    stopping_points = []

    for i in range(B):
        current_prompt = prompts[i]
        selected_models = []
        previous_output = None
        stop_early = False
        reasoning_chain = []

        topk_values, topk_indices = torch.topk(seq_gumbel[i], k=controller.current_k)

        for k, model_idx in enumerate(topk_indices):
            if stop_early and early_stopping:
                break

            model_idx = model_idx.item()
            model = models[model_idx]
            tokenizer = model_tokenizers[model_idx]

            full_input = current_prompt
            if reasoning_chain:
                full_input = "\n\n".join(reasoning_chain) + "\n\n" + current_prompt

            temp = args.model_temps[model_idx]

            output = generate_text(model, tokenizer, [full_input], False, MODEL_NAMES[model_idx], temperature=temp)[0]
            reasoning_chain.append(f"Assistant {k+1} (Model {model_idx}): {output}")

            if previous_output is not None and early_stopping:
                string_similarity = fuzz.ratio(output, previous_output)/100
                if string_similarity > 0.8:
                    logger.info(f"Early stopping at model {k+1} due to similar outputs")
                    logger.info(f"String similarity: {string_similarity:.2f}")
                    stop_early = True
                    stopping_points.append(k+1)
                    break

            previous_output = output
            selected_models.append(model_idx)

        final_output = previous_output if previous_output else ""
        final_outputs.append(final_output)
        all_selected_models.append(selected_models)

    if early_stopping and stopping_points:
        avg_stopping_point = sum(stopping_points)/len(stopping_points)
        logger.info(f"Early stopping triggered {len(stopping_points)} times")
        logger.info(f"Average stopping point: {avg_stopping_point:.1f} models")
    else:
        pass

    end_time = time.time()
    elapsed_time = (end_time - start_time) / B
    logger.info(f"Total inference time: {elapsed_time:.2f} seconds")
    controller.last_C_soft = C_soft.detach()

    return final_outputs, selection_info, all_selected_models, stopping_points if early_stopping else None, elapsed_time

def run_batch_collaborative_inference_static(
    prompts, task, controller, models, model_tokenizers, tokenizer, encoder, static_seq,
    device="cuda", top_k=None, threshold=0.0, early_stopping=True
):
    B = len(prompts)
    N = len(models)

    input_emb = get_shared_embedding(prompts, encoder, tokenizer).to(device)

    if EXPERT_TYPES == "hosted":
        shared_reps = torch.stack([
            get_shared_embedding(generate_text(m, t, prompts, True, n), encoder, tokenizer)
            for (n,m,t) in zip(MODEL_NAMES, models, model_tokenizers)
        ], dim=1).to(device)
    else:
        shared_reps = torch.stack([
            get_shared_embedding(generate_text(m, t, prompts, True), encoder, tokenizer)
            for (m,t) in zip(models, model_tokenizers)
        ], dim=1).to(device)

    controller.eval()
    start_time = time.time()
    with torch.no_grad():
        C_soft, seq_gumbel, selection_info = controller(input_emb, shared_reps)

    final_outputs = []
    all_selected_models = []
    stopping_points = []

    for i in range(B):
        current_prompt = prompts[i]
        selected_models = []
        previous_output = None
        stop_early = False
        reasoning_chain = []

        topk_values, topk_indices = torch.topk(seq_gumbel[i], k=controller.current_k)

        for k, model_idx in enumerate(static_seq):
            if stop_early and early_stopping:
                break

            model_idx = model_idx.item()
            model = models[model_idx]
            tokenizer = model_tokenizers[model_idx]

            full_input = current_prompt
            if reasoning_chain:
                full_input = "\n\n".join(reasoning_chain) + "\n\n" + current_prompt

            temp = args.model_temps[model_idx]

            output = generate_text(model, tokenizer, [full_input], False, MODEL_NAMES[model_idx], temperature=temp)[0]
            reasoning_chain.append(f"Assistant {k+1} (Model {model_idx}): {output}")

            if previous_output is not None and early_stopping:
                string_similarity = fuzz.ratio(output, previous_output)/100

                if string_similarity > 0.8:
                    logger.info(f"Early stopping at model {k+1} due to similar outputs")
                    logger.info(f"String similarity: {string_similarity:.2f}")
                    stop_early = True
                    stopping_points.append(k+1)
                    break

            previous_output = output
            selected_models.append(model_idx)

        final_output = previous_output if previous_output else ""
        final_outputs.append(final_output)
        all_selected_models.append(selected_models)

    if early_stopping and stopping_points:
        avg_stopping_point = sum(stopping_points)/len(stopping_points)
        logger.info(f"Early stopping triggered {len(stopping_points)} times")
        logger.info(f"Average stopping point: {avg_stopping_point:.1f} models")
    else:
        pass

    end_time = time.time()
    elapsed_time = (end_time - start_time) / B
    logger.info(f"Total inference time: {elapsed_time:.2f} seconds")
    controller.last_C_soft = C_soft.detach()

    return final_outputs, selection_info, all_selected_models, stopping_points if early_stopping else None, elapsed_time

def format_prompt(example, task):
    if task == "mmlu":
        return """Above is the conversation history, with the most recent model output at the top.
Each model should carefully read *all previous outputs* and decide how to contribute next.
Your role is to coordinate with earlier outputs by either:
1. Building upon correct reasoning.
2. Correcting or refining mistakes.
3. Adding missing details.
4. Passing an intermediate or final answer if complete.

Always state explicitly what you are doing and why.
Avoid repeating identical reasoning unless you are clarifying or improving it. /no_think

Answer the following question as accurately as possible. Put your final answer as (A), (B), (C), or (D). All questions are single choice.\n\n""" + f"Question: {example['question']}\n(A) {example['choices'][0]}\n(B) {example['choices'][1]}\n(C) {example['choices'][2]}\n(D) {example['choices'][3]}\nAnswer: "
    elif task == "mmlupro":
        return """Above is the conversation history, with the most recent model output at the top.
Each model should carefully read *all previous outputs* and decide how to contribute next.
Your role is to coordinate with earlier outputs by either:
1. Building upon correct reasoning.
2. Correcting or refining mistakes.
3. Adding missing details.
4. Passing an intermediate or final answer if complete.

Always state explicitly what you are doing and why.
Avoid repeating identical reasoning unless you are clarifying or improving it.

Answer the following question as accurately as possible. Put your final answer as (A), (B), (C), (D), (E), (F), (G), (H), (I), (J). All questions are single choice.\n\n""" + f"Question: {example['question']}\n(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n(F) {example['options'][5]}\n(G) {example['options'][6]}\n(H) {example['options'][7]}\n(I) {example['options'][8]}\n(J) {example['options'][9]}\nAnswer: "
    elif task == "humaneval":
        return """Above is the conversation history, with the most recent model output at the top.
Each model should carefully read *all previous outputs* and decide how to contribute next.
Your role is to coordinate with earlier outputs by either:
1. Building upon correct reasoning.
2. Correcting or refining mistakes.
3. Adding missing details.
4. Passing an intermediate or final answer if complete.

Always state explicitly what you are doing and why.
Avoid repeating identical reasoning unless you are clarifying or improving it. /no_think

Read the function signature and comments.Please dont give any explanation and implement the function only. You should only return the code. Put your solution enclosed in backticks.\n\n""" + "Question: ```python\n" + example["prompt"] + "```\nAnswer: ```python\n"
    elif task == "gsm8k":
        answer = example["answer"].split('####')[-1].strip()
        reasoning = example["answer"].split('####')[0].strip()
        return """Above is the conversation history, with the most recent model output at the top.
Each model should carefully read *all previous outputs* and decide how to contribute next.
Your role is to coordinate with earlier outputs by either:
1. Building upon correct reasoning.
2. Correcting or refining mistakes.
3. Adding missing details.
4. Passing an intermediate or final answer if complete.

Always state explicitly what you are doing and why.
Avoid repeating identical reasoning unless you are clarifying or improving it. /no_think

Follow the given examples and answer the mathematics problem.\n\n""" + """Problem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = 6 trees that were planted. The answer is 6.
###
Problem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. Then 2 more cars arrive. Now 3 + 2 = 5 cars are in the parking lot. The answer is 5.
###
Problem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39 pieces left in total. The answer is 39.
###
Problem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason had 20 lollipops originally. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops. The answer is 8.\n\n""" + f"Question: {example['question']}\nAnswer: "
    else:
        context = example.get("context", "")
        return f"Question: {example['question']}\nContext: {context}"

def extract_answer(example, task):
    if task == "mmlu":
        return example["answer"]
    elif task == "mmlupro":
        return example["answer_index"]
    elif task == "humaneval":
        return example["canonical_solution"]
    else:
        return example["answers"]["text"][0] if example["answers"]["text"] else ""

def inference(args):
    logger.info("Starting inference, will save results to %s", args.output_dir)
    if EXPERT_TYPES == 'hosted':
        models = [
            OpenAI(
                base_url="BASE_URL",
                api_key="API_KEY",
                http_client=http_client
            ) for _ in (MODEL_NAMES)
        ]
        tokenizers = [None] * len(MODEL_NAMES)
    else:
        models = [load_model_quantized(name) if USE_QUANTIZED else AutoModelForCausalLM.from_pretrained(name).to(DEVICE) for name in MODEL_NAMES]
        tokenizers = [AutoTokenizer.from_pretrained(name) for name in MODEL_NAMES]
        for t in tokenizers:
            t.pad_token = t.eos_token
            t.padding_side = 'left'

    shared_tokenizer = AutoTokenizer.from_pretrained(SHARED_ENCODER_NAME)
    shared_encoder = AutoModel.from_pretrained(SHARED_ENCODER_NAME).to(DEVICE)

    if args.task == "mmlu":
        dataset = load_dataset(args.dataset, 'all', split="validation")
    elif args.task == "gsm8k":
        dataset = load_dataset(args.dataset, 'main', split="test")
    else:
        dataset = load_dataset(args.dataset, split="test")

    controller = CollaborationController(num_models=len(models), max_seq_len=len(models)-1, use_cosine_bias=True).to(DEVICE)
    controller, checkpoint = CollaborationController.from_saved_state(args.checkpoint)
    controller.to(DEVICE)
    logger.info(f"Loading controller trained for {checkpoint['epoch']} epochs")
    logger.info(f"Current k: {controller.current_k}")
    logger.info(f"Current tau: {controller.tau:.4f}")
    logger.info(f"Selection quality: {controller.selection_quality.item():.4f}")

    optimizer = torch.optim.Adam(controller.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    final_outputs = []
    gold_answers = []
    all_selected_models = []
    all_stopping_points = []
    prompts = []

    selection_logger = SelectionLogger(MODEL_NAMES)

    times = []

    for step in tqdm(range(0, len(dataset), args.batch_size)):
        batch = dataset.select(range(step, min(step + args.batch_size, len(dataset))))

        for idx in range(len(batch)):
            # base = construct_few_shot_prompt(idx, dataset)
            base = ''
            if args.task == "squad":
                q = dataset[step + idx]["question"]
                ctx = dataset[step + idx].get("context", "")
                prompt = base + f"Question: {q}\nContext: {ctx}\nAnswer: "
                answer = dataset[step + idx]["answers"]["text"][0] if dataset[step + idx]["answers"]["text"] else "No Answer"
            elif args.task == "mmlu":
                prompt = format_prompt(dataset[step + idx], task="mmlu")
                _idx = dataset[step + idx]["answer"]
                _gold = dataset[step + idx]["choices"][_idx]
                answer = ["(A)", "(B)", "(C)", "(D)"][_idx] + " " + _gold
            elif args.task == "mmlupro":
                prompt = format_prompt(dataset[step + idx], task="mmlupro")
                _idx = dataset[step + idx]["answer_index"]
                _gold = dataset[step + idx]["options"][_idx]
                answer = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)"][_idx] + " " + _gold
            elif args.task == "humaneval":
                prompt = format_prompt(dataset[step + idx], task="humaneval")
                answer = dataset[step + idx]["canonical_solution"]
            elif args.task == "gsm8k":
                prompt = format_prompt(dataset[step + idx], task="gsm8k")
                answer = dataset[step + idx]["answer"].split('####')[-1].strip()
            else:
                raise ValueError("Unsupported task")
            prompts.append(prompt)
            print(prompt, answer)
            gold_answers.append(answer)

        if args.static_seq is True:
            outputs, selection_info, selections, stopping_points, elapsed_time = run_batch_collaborative_inference_static(
                prompts=prompts[step:step + args.batch_size],
                task=args.task,
                controller=controller,
                models=models,
                model_tokenizers=tokenizers,
                tokenizer=shared_tokenizer,
                encoder=shared_encoder,
                static_seq=[0,1,2],
                device=DEVICE,
                top_k=controller.current_k 
            )

        else:
            outputs, selection_info, selections, stopping_points, elapsed_time = run_batch_collaborative_inference_selected(
                prompts=prompts[step:step + args.batch_size],
                task=args.task,
                controller=controller,
                models=models,
                model_tokenizers=tokenizers,
                tokenizer=shared_tokenizer,
                encoder=shared_encoder,
                device=DEVICE,
                top_k=controller.current_k 
            )

        if elapsed_time is not None:
            times.append(elapsed_time)
            logger.info(f"Avg time per sample so far: {sum(times)/len(times):.2f} seconds")

        selection_logger.log_batch(
            batch_indices=range(step, min(step + args.batch_size, len(dataset))),
            selection_probs=selection_info['selection_probs'],
            selected_indices=selection_info['selected_indices'],
            phase='inference'
        )

        logger.info(outputs)
        final_outputs.extend(outputs)
        all_selected_models.extend(selections)
        if stopping_points is not None:
            all_stopping_points.extend(stopping_points)
        torch.cuda.empty_cache()

    print(len(final_outputs), len(gold_answers), len(prompts))

    avg_models_used = sum(len(m) for m in all_selected_models)/len(all_selected_models)
    logger.info(f"Average models used: {avg_models_used:.2f}")
    if all_stopping_points:
        avg_stopping_point = sum(all_stopping_points)/len(all_stopping_points)
        logger.info(f"Average stopping point: {avg_stopping_point:.2f}")
        logger.info(f"Early stopping rate: {len(all_stopping_points)/len(all_selected_models):.1%}")

    stats = selection_logger.get_selection_stats()
    logger.info("Model Selection Statistics:")
    for name in MODEL_NAMES:
        logger.info(f"{name}: Selected {stats['total_selections'][name]} times | Avg prob: {stats['average_prob'][name]:.3f}")
    print("Inference completed. Saving results...")
    logger.info(f"Average Time per Sample: {sum(times)/len(times) if times else 'N/A'}")

    timestamp_unix = int(time.time())
    csv_file = os.path.join(args.output_dir, f"{args.task}_predictions.csv.{timestamp_unix}")
    with open(csv_file, mode="w", encoding="utf-8") as file:
        writer = csv.writer(file)
        if args.task in ['mmlu', 'squad', 'gsm8k']:
            writer.writerow(["Context", "Question", "Gold Answer", "Prediction", "Selections"])
        elif args.task == 'humaneval':
            writer.writerow(["task_id", "completion", "canonical", "selections"])
        for i, prompt in enumerate(prompts):
            if args.task == "squad":
                ctx = dataset[i].get("context", "")
                q = dataset[i].get("question", "")
            elif args.task == "mmlu":
                ctx = prompt 
                q = dataset[i].get("question", "")
            elif args.task == "mmlupro":
                ctx = prompt
                q = dataset[i].get("question", "")
            elif args.task == "humaneval":
                ctx = prompt
                q = dataset[i].get("prompt", "")
            else:
                ctx, q = prompt, ""

            if args.task == 'humaneval':
                writer.writerow([dataset[i]['task_id'], final_outputs[i], gold_answers[i], selections[i] if i < len(selections) else ""])
            else:
                writer.writerow([ctx, q, gold_answers[i], final_outputs[i], selections[i] if i < len(selections) else ""])

def single_model_inference(args):
    model_name = args.model_name if hasattr(args, 'model_name') else MODEL_NAMES[0]
    logger.info(f"Running single model inference with: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    if USE_QUANTIZED:
        model = load_model_quantized(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

    if args.task == "mmlu":
        dataset = load_dataset(args.dataset, 'all', split="validation")
    elif args.task == "mmlupro":
        dataset = load_dataset(args.dataset, split="validation")
    elif args.task == "gsm8k":
        dataset = load_dataset(args.dataset, 'main', split="test")
    else:
        dataset = load_dataset(args.dataset, split="test")

    final_outputs = []
    gold_answers = []
    prompts = []

    for step in tqdm(range(0, len(dataset), args.batch_size)):
        batch = dataset.select(range(step, min(step + args.batch_size, len(dataset))))

        batch_prompts = []
        batch_answers = []

        for idx in range(len(batch)):
            if args.task == "squad":
                q = batch[idx]["question"]
                ctx = batch[idx].get("context", "")
                prompt = f"Question: {q}\nContext: {ctx}\nAnswer: "
                answer = batch[idx]["answers"]["text"][0] if batch[idx]["answers"]["text"] else "No Answer"
            elif args.task == "mmlu":
                prompt = format_prompt(batch[idx], task="mmlu")
                _idx = batch[idx]["answer"]
                _gold = batch[idx]["choices"][_idx]
                answer = ["(A)", "(B)", "(C)", "(D)"][_idx] + " " + _gold
            elif args.task == "mmlupro":
                prompt = format_prompt(dataset[step + idx], task="mmlupro")
                _idx = dataset[step + idx]["answer_index"]
                _gold = dataset[step + idx]["options"][_idx]
                answer = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)"][_idx] + " " + _gold
            elif args.task == "humaneval":
                prompt = format_prompt(batch[idx], task="humaneval")
                answer = batch[idx]["canonical_solution"]
            elif args.task == "gsm8k":
                prompt = format_prompt(batch[idx], task="gsm8k")
                answer = batch[idx]["answer"].split('####')[-1].strip()
            else:
                raise ValueError("Unsupported task")

            batch_prompts.append(prompt)
            batch_answers.append(answer)

        outputs = generate_text(model, tokenizer, batch_prompts)

        final_outputs.extend(outputs)
        gold_answers.extend(batch_answers)
        prompts.extend(batch_prompts)

        if step == 0:
            for i in range(min(3, len(outputs))):
                logger.info(f"Prompt: {batch_prompts[i][:200]}...")
                logger.info(f"Gold Answer: {batch_answers[i]}")
                logger.info(f"Model Output: {outputs[i]}")
                logger.info("="*50)

    timestamp_unix = int(time.time())
    csv_file = os.path.join(args.output_dir, f"{args.task}_{model_name.replace('/','_')}_predictions.csv.{timestamp_unix}")

    with open(csv_file, mode="w", encoding="utf-8") as file:
        writer = csv.writer(file)
        if args.task in ['mmlu', 'squad', 'gsm8k']:
            writer.writerow(["Context", "Question", "Gold Answer", "Prediction"])
        elif args.task == 'humaneval':
            writer.writerow(["task_id", "completion", "canonical"])

        for i, prompt in enumerate(prompts):
            if args.task == "squad":
                ctx = dataset[i].get("context", "")
                q = dataset[i].get("question", "")
            elif args.task == "mmlu":
                ctx = prompt
                q = dataset[i].get("question", "")
            elif args.task == "mmlupro":
                ctx = prompt
                q = dataset[i].get("question", "")
            elif args.task == "humaneval":
                ctx = prompt
                q = dataset[i].get("prompt", "")
            else:
                ctx, q = prompt, ""

            if args.task == 'humaneval':
                writer.writerow([dataset[i]['task_id'], final_outputs[i], gold_answers[i]])
            else:
                writer.writerow([ctx, q, gold_answers[i], final_outputs[i]])

    logger.info(f"Results saved to {csv_file}")

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
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lambda_symm", type=float, default=0.05)
    parser.add_argument("--lambda_sparse", type=float, default=0.1)
    parser.add_argument("--lambda_oracle", type=float, default=0.5)
    parser.add_argument("--static_seq", action='store_true', default=False)
    parser.add_argument("--static_C", action='store_true', default=False)
    parser.add_argument("--seq_len", type=int, default=4, help="Initial length of model sequence")
    parser.add_argument("--num_experts", type=int, default=3, help="Number of expert models")
    parser.add_argument("--model_name", type=str, help="Model name for inference")
    parser.add_argument("--model_names", type=str)
    parser.add_argument("--model_temps", type=str, help="List of temperatures for each model")
    args = parser.parse_args()
    args.model_names = eval(args.model_names)
    args.model_temps = eval(args.model_temps)
    file_handler = logging.FileHandler(args.logfile)
    file_handler.setLevel(logging.DEBUG)
    formatter = coloredlogs.ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("multi_slm_collab")
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    coloredlogs.install()
    import ast
    MODEL_NAMES = ast.literal_eval(args.model_names)

    args.seq_len = len(MODEL_NAMES) - 1
    args.batch_size = 2

    if args.mode == "train":
        train_controller(args)
    elif args.mode == "inference":
        inference(args)
    elif args.mode == "single":
        single_model_inference(args)
