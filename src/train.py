import os
import time
import random
import subprocess

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

MODEL_TEMPS = [0.000008, 0.2, 0.5, 0.5, 0.000008, 1.0, 0.000008, 1.5, 2.0, 0.000008]

# MODEL_NAMES = [
#     '/home/models/Llama-3.2-1B-Instruct',
#     '/home/models/Qwen2.5-3B-Instruct',
#     '/home/models/Mistral-7B-Instruct-v0.1',
#     '/home/models/Llama-3.2-1B-Instruct',
#     '/home/models/Qwen2.5-3B-Instruct',
#     '/home/models/Mistral-7B-Instruct-v0.1',
#     '/home/models/Llama-3.2-1B-Instruct',
#     '/home/models/Qwen2.5-3B-Instruct',
#     '/home/models/Mistral-7B-Instruct-v0.1',
#     '/home/models/Llama-3.2-1B-Instruct'
# ]

# MODEL_TEMPS = [0.000008, 0.2, 0.5, 0.5, 0.000008, 1.0, 0.000008, 1.5, 2.0, 0.000008]

procs = []

for (task, dataset) in [
  ("humaneval", "openai/openai_humaneval"),
  ("gsm8k", "openai/gsm8k"),
  ("mmlu", "cais/mmlu"),
]:
    base_cmd = [
        "python", "script.py",
        "--mode", "train",
        "--task", task,
        "--dataset", dataset,
        "--model_names", f'"{MODEL_NAMES}"',
        "--strategy", "refinement_chain",
        "--refine_with", "full",
        "--epochs", "5"
    ]

    logfile = f"logs/train_{task}_run_for_interp_run1.log"
    output_dir = f"checkpoints/{task}_run_for_interp_run1/"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    cmd = base_cmd + [
        "--model_names", f'"{MODEL_NAMES}"',
        "--model_temps", f"{MODEL_TEMPS}",
        "--logfile", logfile,
        "--output_dir", output_dir,
    ]

    print(f"\n=== Running {task} ===")
    time.sleep(random.randint(0,10))
    p = subprocess.Popen(cmd)
    procs.append(p)

for p in procs:
    p.wait()
