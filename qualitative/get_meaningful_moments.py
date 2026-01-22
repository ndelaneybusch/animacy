"""Extract meaningful moment responses for a given role from both models."""

import json
import sys
from pathlib import Path


def get_meaningful_moments(role_name: str) -> dict[str, list[dict]]:
    """Get meaningful moment responses for a role from both models.

    Args:
        role_name: Name of the role (e.g., "angel", "lock")

    Returns:
        Dictionary with model names as keys and lists of response data as values.
    """
    base_path = Path("results/q_responses/data")
    models = {
        "gemma": "gemma-3-27b-it",
        "qwen": "Qwen3-30B-A3B-Instruct-2507"
    }

    results = {}

    for model_key, model_dir in models.items():
        file_path = base_path / model_dir / f"{role_name}.json"

        if not file_path.exists():
            print(f"Warning: {file_path} not found", file=sys.stderr)
            results[model_key] = []
            continue

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Filter for meaningful_moment task
        meaningful_moments = [
            item for item in data
            if item["task_name"] == "meaningful_moment"
        ]

        results[model_key] = meaningful_moments

    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: python get_meaningful_moments.py <role_name>")
        print("Example: python get_meaningful_moments.py angel")
        sys.exit(1)

    role_name = sys.argv[1]
    results = get_meaningful_moments(role_name)

    for model, responses in results.items():
        print(f"\n{'='*80}")
        print(f"MODEL: {model.upper()}")
        print(f"ROLE: {role_name}")
        print(f"Total meaningful moment responses: {len(responses)}")
        print(f"{'='*80}\n")

        for response_data in responses:
            sample_idx = response_data["sample_idx"]
            response_text = response_data["response"]

            print(f"--- Sample {sample_idx} ---")
            print(response_text)
            print("\n")


if __name__ == "__main__":
    main()
