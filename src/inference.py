import os
import glob
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

for task in ["mmlu","gsm8k", "humaneval"]:
    print("[info] Running task", task)

    ckpt_list = glob.glob(
        f"checkpoints/{task}_run_for_interp_run1/collaboration_controller_{task}_final*.pt"
    )

    if not ckpt_list:
        print(f"No {task} checkpoint found for experts. Skipping...")
        exit()

    ckpt = ckpt_list[0]
    print(f"Using checkpoint file: {ckpt}")
    os.makedirs(f"checkpoints", exist_ok=True)

    cmd = [
        "python", "ablations/script_static_sequence.py",
        "--mode", "inference",
        "--checkpoint", ckpt,
        "--model_names", f'"{MODEL_NAMES}"',
        "--model_temps", f"{MODEL_TEMPS}",
        "--task", task,
        "--dataset", DATASET_NAMES[task],
        "--strategy", "refinement_chain",
        "--refine_with", "full",
        "--logfile", f"checkpoints/{task}_run_for_interp_run1/inference_log.log",
        "--output_dir", f"checkpoints/{task}_run_for_interp_run1/"
    ]

    subprocess.run(cmd)

