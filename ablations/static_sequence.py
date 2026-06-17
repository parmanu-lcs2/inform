import os
import glob
import subprocess

SCRIPT_PATH = "final-script/ablation/script.py"

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

MODEL_TEMPS = [0.00008, 0.2, 0.5, 0.5, 0.00008, 1.0, 1.2, 1.5, 2.0, 1.0]

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

for task in ["mmlu", "gsm8k", "humaneval"]:
  print("[STATIC_SEQUENCE_ABLATION] TASK:", task)
  checkpoints = glob.glob(
    f"checkpoints/{task}_run_for_interp_run1/collaboration_*{task}_epoch*.pt"
  )

  for i, ckpt in enumerate(checkpoints):
    print("\tEPOCH", i+1)
    os.makedirs(f"logs_{task}", exist_ok=True)
    os.makedirs(f"outputs_{task}/epoch_{i+1}", exist_ok=True)

    subprocess.run([
      "python", "script_static_sequence.py",
        "--mode", "inference",
        "--static_seq",
        "--checkpoint", ckpt,
        "--model_names", f'"{MODEL_NAMES}"',
        "--model_temps", f"{MODEL_TEMPS}",
        "--task", task,
        "--dataset", DATASET_NAMES[task],
        "--strategy", "refinement_chain",
        "--refine_with", "full",
        "--logfile", f"logs_{task}/epoch_{i+1}.log",
        "--output_dir", f"outputs_{task}/epoch_{i+1}"
    ])
