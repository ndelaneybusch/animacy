import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from animacy.models import TransformersModelConfig
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

    # Model Configuration
    parser.add_argument(
        "--model_name", type=str, required=True, help="HuggingFace model name"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to run on (cuda, cpu, auto)"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        help="Torch dtype (float16, bfloat16, auto)",
    )
    parser.add_argument(
        "--load_in_8bit", action="store_true", help="Load model in 8-bit quantization"
    )
    parser.add_argument(
        "--load_in_4bit", action="store_true", help="Load model in 4-bit quantization"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024, help="Maximum tokens to generate"
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

    print(f"Initializing model: {args.model_name}...")
    config = TransformersModelConfig(
        model_name=args.model_name,
        device=args.device,
        torch_dtype=args.torch_dtype,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    with create_inference_engine(config) as engine:
        roles = list(create_roles_from_df(df))
        print(f"Processing {len(roles)} roles...")

        for role in tqdm(roles, desc="Generating responses"):
            role_results = []
            tasks = create_tasks_for_role(role)

            for task in tasks:
                responses = sample_responses(engine, task, num_samples=args.num_samples)

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
