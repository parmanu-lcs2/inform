import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import logging
import coloredlogs
import os
import csv
import ast

# Import necessary components from your original script
from script import (
    CollaborationController, 
    load_model_quantized, 
    prepare_datasets, 
    get_shared_embedding, 
    generate_text, 
    format_prompt, 
    extract_answer, 
    compute_utility_loss,
    AutoTokenizer, 
    AutoModel,
    OpenAI,
    DEVICE,
    SHARED_ENCODER_NAME,
    http_client
)

# Setup Logging
logger = logging.getLogger("probe_steer")
coloredlogs.install(level='INFO')

class SteerableController(nn.Module):
    """
    Wrapper around the CollaborationController to enable activation steering.
    """
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.steering_vector = None
        self.steering_strength = 0.0
        self.hook_handle = None
        self.captured_hidden = None

    def _hook_fn(self, module, input, output):
        """Capture hidden state and inject steering vector if present."""
        # output is 'h' (batch, hidden_dim)
        self.captured_hidden = output.detach().clone()
        
        if self.steering_vector is not None and self.steering_strength != 0.0:
            # h' = h + alpha * w
            steered_h = output + (self.steering_strength * self.steering_vector.to(output.device))
            return steered_h
        return output

    def register_hook(self):
        # We hook into input_proj because that creates the internal latent representation 'h'
        self.hook_handle = self.controller.input_proj.register_forward_hook(self._hook_fn)

    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()

    def forward(self, input_embedding, shared_reps=None, oracle_emb=None):
        return self.controller(input_embedding, shared_reps, oracle_emb)

def collect_activations(args, wrapper, dataset, shared_encoder, shared_tokenizer, models, model_tokenizers, model_temps):
    """
    Phase 1: Run inference on a subset to collect (Hidden State, Success/Fail) pairs.
    """
    logger.info("Phase 1: Collecting activations and failure modes...")
    
    activations = []
    labels = [] # 1 for Success, 0 for Failure
    
    wrapper.steering_strength = 0.0 # Ensure no steering during collection
    
    limit = args.limit if args.limit > 0 else len(dataset)
    
    for i in tqdm(range(limit)):
        item = dataset[i]
        
        # Prepare Inputs
        if args.task == "gsm8k":
            prompt = format_prompt(item, "gsm8k")
            gold = item["answer"].split('####')[-1].strip()
        elif args.task == "mmlu":
            prompt = format_prompt(item, "mmlu")
            gold = item["answer"] # index
        elif args.task == "humaneval":
            prompt = format_prompt(item, "humaneval")
            gold = item["canonical_solution"]
        else:
            continue

        # 1. Get Embedding
        with torch.no_grad():
            input_emb = get_shared_embedding([prompt], shared_encoder, shared_tokenizer).to(DEVICE)
            
            # 2. Run Controller (Forward Pass triggers Hook)
            # We trigger the hook to capture 'h'
            _ = wrapper(input_emb, None) 
            
            hidden_state = wrapper.captured_hidden.cpu().numpy().flatten()
            
            # 3. Get selection probabilities to simulate routing
            _, seq_gumbel, _ = wrapper.controller(input_emb, None)
            top_choice_idx = torch.argmax(seq_gumbel, dim=1).item()
            
            selected_model = models[top_choice_idx]
            selected_tokenizer = model_tokenizers[top_choice_idx]
            model_name = args.model_names[top_choice_idx]
            
            # Retrieve the specific temperature for this model
            current_temp = model_temps[top_choice_idx]
            
            # Generate Answer with correct temperature
            response = generate_text(
                selected_model, 
                selected_tokenizer, 
                [prompt], 
                model_name=model_name,
                temperature=current_temp
            )[0]
            
            # 4. Check Correctness
            loss = compute_utility_loss([response], [gold], args.task)
            is_correct = 1 if loss < 0.1 else 0 # Threshold for "Success"
            
            activations.append(hidden_state)
            labels.append(is_correct)
            
    return np.array(activations), np.array(labels)

def train_probe(X, y):
    """
    Phase 2: Train a Logistic Regression probe.
    """
    logger.info(f"Phase 2: Training Probe on {len(y)} samples ({np.sum(y)} successes)...")
    
    if np.sum(y) == 0 or np.sum(y) == len(y):
        logger.warning("Classes are not balanced (all success or all failure). Probe cannot learn.")
        return torch.zeros(X.shape[1]), 0.0

    # Simple Logistic Regression
    clf = LogisticRegression(random_state=42, C=1.0, solver='liblinear')
    clf.fit(X, y)
    
    acc = accuracy_score(y, clf.predict(X))
    logger.info(f"Probe Accuracy on Train Set: {acc:.4f}")
    
    # The coefficient vector is the direction normal to the decision boundary.
    steering_vec = torch.tensor(clf.coef_, dtype=torch.float32).to(DEVICE)
    
    return steering_vec, acc

def evaluate_steered(args, wrapper, steering_vec, dataset, shared_encoder, shared_tokenizer, models, model_tokenizers, model_temps):
    """
    Phase 3: Run inference with steering enabled.
    """
    logger.info(f"Phase 3: Running Steered Inference (Alpha={args.alpha})...")
    
    wrapper.steering_vector = steering_vec
    wrapper.steering_strength = args.alpha # Inject the "Success" direction
    
    success_count = 0
    limit = args.limit if args.limit > 0 else len(dataset)
    
    for i in tqdm(range(limit)):
        item = dataset[i]
        
        if args.task == "gsm8k":
            prompt = format_prompt(item, "gsm8k")
            gold = item["answer"].split('####')[-1].strip()
        elif args.task == "mmlu":
            prompt = format_prompt(item, "mmlu")
            gold = item["answer"]
        elif args.task == "humaneval":
            prompt = format_prompt(item, "humaneval")
            gold = item["canonical_solution"]
        else:
            continue

        with torch.no_grad():
            input_emb = get_shared_embedding([prompt], shared_encoder, shared_tokenizer).to(DEVICE)
            
            # The wrapper automatically injects the vector now during this forward pass
            _, seq_gumbel, _ = wrapper(input_emb, None)
            
            # Did the routing change?
            top_choice_idx = torch.argmax(seq_gumbel, dim=1).item()
            
            selected_model = models[top_choice_idx]
            selected_tokenizer = model_tokenizers[top_choice_idx]
            model_name = args.model_names[top_choice_idx]
            current_temp = model_temps[top_choice_idx]
            
            response = generate_text(
                selected_model, 
                selected_tokenizer, 
                [prompt], 
                model_name=model_name,
                temperature=current_temp
            )[0]
            
            loss = compute_utility_loss([response], [gold], args.task)
            if loss < 0.1:
                success_count += 1
                
    accuracy = success_count / limit
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--limit", type=int, default=50, help="Number of samples for probe training")
    parser.add_argument("--alpha", type=float, default=2.0, help="Steering strength")
    parser.add_argument("--model_names", type=str, required=True, help="List of model names string")
    parser.add_argument("--model_temps", type=str, required=True, help="List of model temperatures string")
    
    args = parser.parse_args()
    
    # Parse list strings
    args.model_names = ast.literal_eval(args.model_names)
    args.model_temps = ast.literal_eval(args.model_temps)

    if len(args.model_names) != len(args.model_temps):
        raise ValueError(f"Mismatch: {len(args.model_names)} models vs {len(args.model_temps)} temps.")

    # 1. Load Resources
    logger.info("Loading Orchestrator and Models...")
    controller, _ = CollaborationController.from_saved_state(args.checkpoint)
    controller.eval()
    
    # Wrap the controller
    steerable_controller = SteerableController(controller)
    steerable_controller.register_hook()
    
    # Load Encoder
    shared_tokenizer = AutoTokenizer.from_pretrained(SHARED_ENCODER_NAME)
    shared_encoder = AutoModel.from_pretrained(SHARED_ENCODER_NAME).to(DEVICE)
    
    # Load Experts
    models = []
    tokenizers = []
    
    # Check if we are using hosted models or local models
    if 'openai' in args.model_names[0] or 'hosted' in str(args.checkpoint):
        logger.info("Initializing Hosted Models...")
        models = [
            OpenAI(
                base_url="BASE_URL",
                api_key="API_KEY",
                http_client=http_client
            ) for _ in args.model_names
        ]
        tokenizers = [None] * len(args.model_names)
    else:
        logger.info("Initializing Local Models (this may take time)...")
        # Assuming USE_QUANTIZED is True as per your original script logic
        for name in args.model_names:
            models.append(load_model_quantized(name))
            tokenizer = AutoTokenizer.from_pretrained(name)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            tokenizers.append(tokenizer)
    
    # 2. Prepare Data
    if args.task == "gsm8k":
        ds_name = "openai/gsm8k" 
    elif args.task == "mmlu":
        ds_name = "cais/mmlu"
    elif args.task == "humaneval":
        ds_name = "openai/openai_humaneval"
    else:
        ds_name = "squad" # default fallback
        
    samples = prepare_datasets(argparse.Namespace(task=args.task, dataset=ds_name))
    
    # 3. Phase 1: Collect
    X, y = collect_activations(
        args, steerable_controller, samples, shared_encoder, shared_tokenizer, 
        models, tokenizers, args.model_temps
    )
    logger.info(f"Collected {len(X)} samples. Success rate: {np.mean(y):.2%}")

    # 4. Phase 2: Train Probe
    steering_vector, probe_acc = train_probe(X, y)
    
    # 5. Phase 3: Steer
    steered_acc = evaluate_steered(
        args, steerable_controller, steering_vector, samples, shared_encoder, shared_tokenizer, 
        models, tokenizers, args.model_temps
    )
    
    logger.info("="*30)
    logger.info(f"Baseline Accuracy: {np.mean(y):.2%}")
    logger.info(f"Steered Accuracy:  {steered_acc:.2%}")
    logger.info(f"Improvement:       {steered_acc - np.mean(y):.2%}")
    logger.info("="*30)
    
    # Save vector
    torch.save(steering_vector, f"{args.task}_steering_vec.pt")

if __name__ == "__main__":
    main()
