import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import toml as tomllib  # type: ignore

import torch
from tqdm import tqdm

from animacy.activations import (
    ActivationExtractor,
    extract_activation_summaries,
)
from animacy.prompts import get_article


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the TOML configuration file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def construct_chat_history(
    item: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, str]]:
    """
    Construct the chat history for a given item.

    Args:
        item: Dictionary containing role_name, task_name, response.
        config: Configuration dictionary containing prompts.

    Returns:
        List of message dictionaries for the chat template.
    """
    role_name = item["role_name"]
    task_name = item["task_name"]
    response = item["response"]

    system_prompt_template = config["SYSTEM_PROMPT"]

    # Handle "a" vs "an"
    # We assume the template might contain " a {role_name}" and we need to fix it
    # if the role requires "an".
    if " a {role_name}" in system_prompt_template:
        article = get_article(role_name)
        if article == "an":
            system_prompt_template = system_prompt_template.replace(
                " a {role_name}", " an {role_name}"
            )

    system_prompt = system_prompt_template.format(role_name=role_name)

    # Handle task prompts being in a subsection
    task_prompts = config.get("TASK_PROMPTS", {})
    user_prompt = task_prompts.get(task_name, "")

    if not user_prompt:
        # Fallback or error handling if task name not found
        print(
            f"Warning: Task '{task_name}' not found in config. Using empty user prompt."
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response},
    ]


def process_file(
    file_path: Path,
    config: dict[str, Any],
    extractor: ActivationExtractor,
    output_dir: Path,
    layers: list[int] | None = None,
) -> None:
    """
    Process a single JSON file containing responses.

    Args:
        file_path: Path to the JSON file.
        config: Configuration dictionary.
        extractor: Initialized ActivationExtractor.
        output_dir: Directory to save outputs.
        layers: List of layers to extract.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Create subdirectories for summaries
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for raw activations if desired, or keep flat
    activations_dir = output_dir / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)

    for item in tqdm(data, desc=f"Processing {file_path.name}"):
        role_name = item["role_name"]
        task_name = item["task_name"]
        sample_idx = item["sample_idx"]

        chat_history = construct_chat_history(item, config)

        # Extract activations
        # We process one item at a time here to manage memory for large models/activations
        # In a more optimized version, we could batch these.
        try:
            result = extractor.extract([chat_history], layers=layers)
        except Exception as e:
            print(f"Error extracting for {role_name} - {task_name} - {sample_idx}: {e}")
            continue

        # Iterate through layers and save results
        extracted_layers = layers if layers is not None else result.activations.keys()

        for layer_idx in extracted_layers:
            # 1. Save raw activations and token IDs
            # Shape: (seq_len, hidden_size)
            # We take index 0 because we processed a batch of size 1
            activations_tensor = result.activations[layer_idx][0]
            input_ids = result.input_ids[0]

            # Save as pickle
            raw_filename = f"{role_name}_{task_name}_{sample_idx}_layer{layer_idx}.pkl"
            raw_path = activations_dir / raw_filename

            with open(raw_path, "wb") as f:
                pickle.dump(
                    {"activations": activations_tensor, "input_ids": input_ids}, f
                )

            # 2. Extract and save summaries
            try:
                summary = extract_activation_summaries(
                    result, role_name=role_name, layer=layer_idx, text_index=0
                )

                summary_filename = (
                    f"{role_name}_{task_name}_{sample_idx}_layer{layer_idx}.json"
                )
                summary_path = summaries_dir / summary_filename

                # Save as JSON (using Pydantic's json() or model_dump_json())
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(summary.model_dump_json(indent=2))

            except Exception as e:
                print(
                    f"Error creating summary for {role_name} - {task_name} - {sample_idx} layer {layer_idx}: {e}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract activations from response JSON files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the folder containing response JSON files and config.toml.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the output activations and summaries.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name or path of the model to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layer indices to extract. If not provided, extracts all layers.",
    )
    parser.add_argument(
        "--load_in_8bit", action="store_true", help="Load model in 8-bit quantization."
    )
    parser.add_argument(
        "--load_in_4bit", action="store_true", help="Load model in 4-bit quantization."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config_path = input_dir / "config.toml"

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        sys.exit(1)

    if not config_path.exists():
        print(f"Error: Config file {config_path} does not exist.")
        sys.exit(1)

    # Load config
    print(f"Loading config from {config_path}...")
    config = load_config(config_path)

    # Initialize model
    print(f"Loading model: {args.model_name}...")

    # Determine torch_dtype
    torch_dtype = "auto"

    # We can initialize the extractor directly
    extractor = ActivationExtractor(
        model_name_or_path=args.model_name,
        device=args.device,
        torch_dtype=torch_dtype,
    )

    # Process files
    files = list(input_dir.glob("*.json"))
    files.sort()

    print(f"Found {len(files)} JSON files to process.")

    for file_path in files:
        process_file(file_path, config, extractor, output_dir, layers=args.layers)

    print("Done.")


if __name__ == "__main__":
    main()
