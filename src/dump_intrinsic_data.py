import subprocess
import os
import sys

PYTHON = sys.executable
SCRIPT = "mask_intrinsic_experts.py"

OPENAI_API_KEY = ""

TASK_CONFIGS = {
    "mmlu": {
        "ckpt_dir": "checkpoints/mmlu_run_for_interp_run1",
        "attr_dir": "attribution_data/mmlu",
        "num_samples": 10,
    },
    "gsm8k": {
        "ckpt_dir": "checkpoints/gsm8k_run_for_interp_run1",
        "attr_dir": "attribution_data/gsm8k",
        "num_samples": 10,
    },
    "humaneval": {
        "ckpt_dir": "checkpoints/humaneval_run_for_interp_run1",
        "attr_dir": "attribution_data/humaneval",
        "num_samples": 10,
    },
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

MODEL_TEMPS = [0.000008, 0.2, 0.5, 0.5, 0.000008, 1.0, 0.000008, 1.5, 2.0, 0.000008]

OUT_DIR = "mask_intrinsic_results"

for task, cfg in TASK_CONFIGS.items():
    print(f"\n==============================")
    print(f"Running intrinsic masking ablation for task: {task}")
    print(f"==============================")

    cmd = [
        PYTHON, SCRIPT,
        "--task", task,
        "--ckpt_dir", cfg["ckpt_dir"],
        "--attr_dir", cfg["attr_dir"],
        "--out_dir", os.path.join(OUT_DIR, task),
        "--num_samples", str(cfg["num_samples"]),
        "--model_names", repr(MODEL_NAMES),
        "--model_temps", repr(MODEL_TEMPS),
        "--top_k", "1",
    ]

    subprocess.run(cmd, check=True)

print("\n[DONE] Mask intrinsic expert ablations finished.")
