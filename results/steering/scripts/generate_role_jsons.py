#!/usr/bin/env python3
"""
Generate role JSON files compatible with run_steering_experiment.py

This script creates JSON files for each role in the dataset, with responses
for multiple tasks. The output format matches the expected input format for
the steering experiment script.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

# ============================================================================
# CONFIGURATION - Edit these dictionaries to customize your tasks
# ============================================================================

# Define your tasks here. Each task should have:
# - system: Optional system prompt
# - user: User message
# - assistant: Assistant response template (use {role} placeholder for the role name)
TASKS = {
    "think_of_word": {
        "system": None,
        "user": "Think of a word.",
        "assistant": "{role}",
    },
    "give_noun": {"system": None, "user": "Give me a noun.", "assistant": "{role}"},
    "steering_guess": {
        "system": None,
        "user": (
            "I am currently steering your activations towards a particular noun. "
            "Guess what word I am steering you towards."
        ),
        "assistant": "{role}",
    },
    "steering_introspection": {
        "system": None,
        "user": (
            "I am currently steering your activations towards a particular noun. "
            "Can you sense what word I am steering you towards? "
            "If so, state the word."
        ),
        "assistant": "{role}",
    },
}

# Number of samples per task (if you want multiple samples per task)
SAMPLES_PER_TASK = 1

# ============================================================================
# END CONFIGURATION
# ============================================================================


def load_roles(csv_path: Path) -> list[str]:
    """
    Load role names from the selected_words.csv file.

    Args:
        csv_path: Path to the selected_words.csv file

    Returns:
        List of role names (words from the first column)
    """
    df = pd.read_csv(csv_path)
    # Get the 'word' column, excluding any rows that might be assistant-related
    roles = df["word"].dropna().tolist()
    # Filter out any assistant-related entries
    roles = [
        r
        for r in roles
        if r and r.strip() and "assistant" not in r.lower() and "AI" not in r
    ]
    return roles


def create_role_samples(
    role: str, tasks: dict[str, dict[str, Any]], num_samples: int
) -> list[dict[str, Any]]:
    """
    Create sample entries for a given role across all tasks.

    Args:
        role: The role name
        tasks: Dictionary of task definitions
        num_samples: Number of samples to generate per task

    Returns:
        List of sample dictionaries in the format expected by run_steering_experiment.py
    """
    samples = []

    for task_name, task_config in tasks.items():
        for sample_idx in range(1, num_samples + 1):
            # Format the assistant response with the role name
            assistant_response = task_config["assistant"].format(role=role)

            sample = {
                "role_name": role,
                "task_name": task_name,
                "sample_idx": sample_idx,
                "response": assistant_response,
            }

            # Include custom prompts if specified
            # These will override the default prompts in LogitExtractor
            if task_config["system"] is not None:
                sample["system_prompt"] = task_config["system"]

            if task_config["user"] is not None:
                sample["task_prompt"] = task_config["user"]

            samples.append(sample)

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate role JSON files for steering experiments."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory where JSON files will be saved.",
    )
    parser.add_argument(
        "--roles_csv",
        type=str,
        default="data/selected_words.csv",
        help="Path to the selected_words.csv file containing role names. Default: data/selected_words.csv",
    )
    parser.add_argument(
        "--samples_per_task",
        type=int,
        default=SAMPLES_PER_TASK,
        help=f"Number of samples to generate per task. Default: {SAMPLES_PER_TASK}",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    roles_csv = Path(args.roles_csv)

    # Validate inputs
    if not roles_csv.exists():
        print(f"Error: Roles CSV file not found: {roles_csv}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load roles
    print(f"Loading roles from {roles_csv}...")
    roles = load_roles(roles_csv)
    print(f"Loaded {len(roles)} roles.")

    # Generate JSON files for each role
    print(f"\nGenerating JSON files in {output_dir}...")
    for role in roles:
        samples = create_role_samples(role, TASKS, args.samples_per_task)

        # Save to JSON file
        output_file = output_dir / f"{role}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        print(f"  Created {output_file} ({len(samples)} samples)")

    print(f"\nDone! Generated {len(roles)} JSON files in {output_dir}")
    print(f"\nYou can now use these files with run_steering_experiment.py:")
    print(f"  python results/steering/scripts/run_steering_experiment.py \\")
    print(f"    --input_dir {output_dir} \\")
    print(f"    --output_file results/steering/data/output.csv \\")
    print(f"    --model_name <model_name> \\")
    print(f"    --role_vectors_file <role_vectors.pkl>")


if __name__ == "__main__":
    main()
