import argparse
import json
import os
import sys
from pathlib import Path

# Set CUDA allocation configuration to avoid fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
from tqdm import tqdm

from animacy.models import TransformersModelConfig, VLLMModelConfig
from animacy.prompts import (
    create_inference_engine,
    create_roles_from_df,
    create_tasks_for_role,
)
from animacy.responses import sample_responses

# Add src to path to ensure imports work if package not installed
# This assumes the script is located at results/q_responses/scripts/run_experiment.py
# and the project root is 3 levels up.
project_root = Path(__file__).resolve().parents[3]
if str(project_root / "src") not in sys.path:
    sys.path.append(str(project_root / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run animacy experiment")

    # Backend Selection
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Inference backend to use (transformers or vllm)",
    )

    # Model Configuration
    parser.add_argument(
        "--model_name", type=str, required=True, help="HuggingFace model name"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024, help="Maximum tokens to generate"
    )

    # Transformers-specific arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="[Transformers only] Device to run on (cuda, cpu, auto)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        help="[Transformers only] Torch dtype (float16, bfloat16, auto)",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="[Transformers only] Load model in 8-bit quantization",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="[Transformers only] Load model in 4-bit quantization",
    )

    # vLLM-specific arguments
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="[vLLM only] Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="[vLLM only] Fraction of GPU memory to use (0.0-1.0)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="[vLLM only] Data type for model weights (auto, half, float16, bfloat16, float, float32)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="[vLLM only] Maximum sequence length (reduces KV cache memory usage)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model",
    )

    # Data and Output
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/selected_words.csv",
        help="Path to input CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/q_responses/data",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples per task"
    )

    args = parser.parse_args()

    # Resolve paths relative to project root if they are relative
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = project_root / csv_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    print(f"Initializing model: {args.model_name} (backend: {args.backend})...")

    # Create appropriate config based on backend
    if args.backend == "transformers":
        config = TransformersModelConfig(
            model_name=args.model_name,
            device=args.device,
            torch_dtype=args.torch_dtype,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.backend == "vllm":
        config = VLLMModelConfig(
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    with create_inference_engine(config) as engine:
        roles = list(create_roles_from_df(df))
        print(f"Processing {len(roles)} roles...")

        for role in tqdm(roles, desc="Roles"):
            role_results = []
            tasks = list(create_tasks_for_role(role))

            # Calculate total samples for this role
            total_samples = len(tasks) * args.num_samples

            with tqdm(
                total=total_samples, desc=f"Samples ({role.role_name})", leave=False
            ) as pbar:
                for task in tasks:
                    responses = sample_responses(
                        engine, task, num_samples=args.num_samples
                    )
                    pbar.update(len(responses))

                    for i, response in enumerate(responses):
                        role_results.append(
                            {
                                "role_name": response.role_name,
                                "task_name": response.task_name,
                                "sample_idx": i + 1,
                                "response": response.response,
                            }
                        )

            # Save results for this role
            # Sanitize filename
            safe_role_name = "".join(
                x for x in role.role_name if x.isalnum() or x in (" ", "-", "_")
            ).strip()
            output_file = output_dir / f"{safe_role_name}.json"

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(role_results, f, indent=2, ensure_ascii=False)

    print(f"Done! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
