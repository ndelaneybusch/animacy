"""
Example call:
# Qwen find OOM steering strengths
python ~/animacy/results/steering/scripts/run_steering_experiment.py \
    --input_dir ~/animacy/results/q_responses/data/Qwen3-30B-A3B-Instruct-2507/ \
    --output_file results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_responses/avg_response_sys_diff.csv \
    --model_name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --role_vectors_file ~/animacy/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_vectors_avg_response_sys_diff.pkl \
    --roles napkin scarf hair foot umpire butler \
    --magnitudes 1 1.5 2 2.5 3 \
    --no_system_prompt \
    --batch_size 24

"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from animacy.steering.evaluation import evaluate_steered_logits


def load_role_vectors(file_path: Path) -> dict[str, dict[int, torch.Tensor]]:
    """
    Load role vectors from a pickle file.
    Expected format: dict[role_name, dict[layer_idx, vector]]
    Converts numpy arrays to torch tensors if needed.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Convert numpy arrays to torch tensors if needed
    for role_name, layer_vectors in data.items():
        for layer_idx, vector in layer_vectors.items():
            if not isinstance(vector, torch.Tensor):
                data[role_name][layer_idx] = torch.from_numpy(vector)

    return data


def load_samples(input_dir: Path) -> list[dict[str, Any]]:
    """
    Load all JSON samples from the input directory.
    """
    samples = []
    files = list(input_dir.glob("*.json"))
    files.sort()

    for file_path in tqdm(files, desc="Loading samples"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_data = json.load(f)
                # file_data is a list of dicts
                samples.extend(file_data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run steering experiments and extract log-probabilities."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the folder containing response JSON files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output DataFrame (e.g., .csv or .pkl).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name or path of the model to use.",
    )
    parser.add_argument(
        "--role_vectors_file",
        type=str,
        required=True,
        help="Path to the pickle file containing role vectors.",
    )
    parser.add_argument(
        "--roles",
        type=str,
        nargs="+",
        help="List of roles to process. If not provided, process all roles found in vectors and samples.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="List of layers to steer. If not provided, steer all available layers for the role.",
    )
    parser.add_argument(
        "--magnitudes",
        type=float,
        nargs="+",
        default=[1.0],
        help="List of steering magnitudes to try. Default: [1.0].",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--no_system_prompt",
        action="store_true",
        help="Do not use the system prompt when extracting log-probabilities.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation. Default: 8.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    role_vectors_path = Path(args.role_vectors_file)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        sys.exit(1)

    if not role_vectors_path.exists():
        print(f"Error: Role vectors file {role_vectors_path} does not exist.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading role vectors from {role_vectors_path}...")
    role_vectors = load_role_vectors(role_vectors_path)
    available_roles = set(role_vectors.keys())
    print(f"Loaded vectors for {len(available_roles)} roles.")

    print(f"Loading samples from {input_dir}...")
    all_samples = load_samples(input_dir)
    print(f"Loaded {len(all_samples)} samples.")

    # Filter roles
    if args.roles:
        target_roles = set(args.roles)
        # Verify roles exist in vectors
        missing_roles = target_roles - available_roles
        if missing_roles:
            print(
                f"Warning: The following requested roles have no vectors: {missing_roles}"
            )

        roles_to_process = target_roles.intersection(available_roles)
    else:
        roles_to_process = available_roles

    print(f"Processing {len(roles_to_process)} roles.")

    print(f"Loading model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    all_results = []

    # Iterate over magnitudes
    for magnitude in tqdm(args.magnitudes, desc="Magnitudes"):
        # Iterate over roles
        for role in tqdm(
            roles_to_process, desc=f"Roles (mag={magnitude})", leave=False
        ):
            # Filter samples for this role
            role_samples = [s for s in all_samples if s.get("role_name") == role]

            if not role_samples:
                continue

            # Get vectors for this role
            vectors = role_vectors[role]
            available_layers = set(vectors.keys())

            # Determine layers to steer
            if args.layers:
                target_layers = set(args.layers)
                layers_to_steer = list(target_layers.intersection(available_layers))
                if not layers_to_steer:
                    print(
                        f"Warning: No valid layers to steer for role {role} (requested "
                        f"{args.layers}, available {list(available_layers)})"
                    )
                    continue
            else:
                layers_to_steer = list(available_layers)

            # Prepare steering vectors dict for this specific run
            # We only include the layers we want to steer
            current_steering_vectors = {l: vectors[l] for l in layers_to_steer}

            # Run evaluation
            try:
                results = evaluate_steered_logits(
                    model=model,
                    tokenizer=tokenizer,
                    steering_vectors=current_steering_vectors,
                    layers=layers_to_steer,
                    magnitude=magnitude,
                    samples=role_samples,
                    use_system_prompt=not args.no_system_prompt,
                    batch_size=args.batch_size,
                )

                # Add metadata and convert to dict
                for res in results:
                    res_dict = (
                        res.model_dump() if hasattr(res, "model_dump") else res.dict()
                    )
                    res_dict["steering_magnitude"] = magnitude
                    res_dict["steered_layers"] = str(sorted(layers_to_steer))
                    all_results.append(res_dict)

            except Exception as e:
                print(f"Error processing role {role} at magnitude {magnitude}: {e}")
                import traceback

                traceback.print_exc()

    # Save results
    if not all_results:
        print("No results generated.")
        return

    df = pd.DataFrame(all_results)
    print(f"Saving {len(df)} results to {output_file}...")

    if output_file.suffix == ".csv":
        df.to_csv(output_file, index=False)
    elif output_file.suffix == ".pkl":
        df.to_pickle(output_file)
    elif output_file.suffix == ".parquet":
        df.to_parquet(output_file)
    else:
        print("Unknown extension, saving as CSV.")
        df.to_csv(output_file, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
