import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from animacy.analysis.logits import LogitExtractor, ResponseLogits

# Add src to path to ensure imports work
project_root = Path(__file__).resolve().parents[3]
if str(project_root / "src") not in sys.path:
    sys.path.append(str(project_root / "src"))


def process_item(item: dict[str, Any], extractor: LogitExtractor) -> ResponseLogits:
    """
    Process a single item (dict) and return a ResponseLogits object.

    Args:
        item: A dictionary containing the following keys:
            - role_name: The name of the role (e.g., "angel").
            - task_name: The name of the task (e.g., "meaning_of_life").
            - sample_idx: The index of the sample.
            - response: The generated response text.
        extractor: A LogitExtractor object.

    Returns:
        ResponseLogits object populated with calculated logits.
    """
    return extractor.extract_logits(
        role_name=item["role_name"],
        task_name=item["task_name"],
        sample_idx=item["sample_idx"],
        response_text=item["response"],
        use_system_prompt=True,
    )


def process_file(
    file_path: Path, extractor: LogitExtractor
) -> Iterable[ResponseLogits]:
    """
    Process a file and return an iterable of ResponseLogits.

    Args:
        file_path: Path to the JSON file containing the items.
        extractor: A LogitExtractor object.

    Returns:
        Iterable of ResponseLogits objects.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        yield process_item(item, extractor)


def process_folder(folder_path: Path, extractor: LogitExtractor) -> pd.DataFrame:
    """
    Process a folder and return the data frame.

    Args:
        folder_path: Path to the folder containing the JSON files.
        extractor: A LogitExtractor object.

    Returns:
        DataFrame containing the logits for all items.
    """
    all_logits = []
    files = list(folder_path.glob("*.json"))

    # Sort files for reproducibility
    files.sort()

    for file_path in tqdm(files, desc="Processing files"):
        try:
            file_logits = process_file(file_path, extractor)
            # Use model_dump() for Pydantic v2, fallback to dict() if needed
            if hasattr(ResponseLogits, "model_dump"):
                all_logits.extend([l.model_dump() for l in file_logits])
            else:
                all_logits.extend([l.dict() for l in file_logits])
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    return pd.DataFrame(all_logits)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract logits from response JSON files."
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--load_in_8bit", action="store_true", help="Load model in 8-bit quantization."
    )
    parser.add_argument(
        "--load_in_4bit", action="store_true", help="Load model in 4-bit quantization."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name}...")

    # Handle quantization
    # We assume the environment has bitsandbytes if these flags are used, or transformers
    # handles it.

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    model_kwargs = {
        "device_map": args.device,
        "torch_dtype": "auto",
        "trust_remote_code": True,
    }

    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    elif args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    extractor = LogitExtractor(model, tokenizer)

    print(f"Processing responses from {input_dir}...")
    df = process_folder(input_dir, extractor)

    print(f"Saving results to {output_file}...")
    if output_file.suffix == ".csv":
        df.to_csv(output_file, index=False)
    elif output_file.suffix == ".pkl":
        df.to_pickle(output_file)
    elif output_file.suffix == ".parquet":
        df.to_parquet(output_file)
    else:
        # Default to CSV if unknown extension
        print("Unknown extension, saving as CSV.")
        df.to_csv(output_file, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
