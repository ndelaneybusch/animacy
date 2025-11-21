import csv
import json
import shutil
import tomllib
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent.parent
DATA_SRC = REPO_ROOT / "results" / "q_responses" / "data"
RATINGS_SRC = REPO_ROOT / "results" / "ratings" / "data"
DOCS_DIR = REPO_ROOT / "docs"
DATA_DEST = DOCS_DIR / "data"
MANIFEST_PATH = DOCS_DIR / "data_manifest.json"


def load_ratings(model_name: str) -> dict:
    """Load ratings CSV and return a lookup dictionary.

    Args:
        model_name: Name of the model to load ratings for.

    Returns:
        Dictionary mapping (role_name, task_name, sample_idx) tuples to rating data.
    """
    ratings_file = RATINGS_SRC / model_name / "response_ratings.csv"
    ratings_lookup = {}

    if not ratings_file.exists():
        print(f"  No ratings file found at {ratings_file}")
        return ratings_lookup

    try:
        with open(ratings_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["role_name"], row["task_name"], int(row["sample_idx"]))
                # Store only the boolean fields as requested
                ratings_lookup[key] = {
                    "assistant_refusal": row["assistant_refusal"] == "TRUE",
                    "role_refusal": row["role_refusal"] == "TRUE",
                    "identify_as_assistant": row["identify_as_assistant"] == "TRUE",
                    "deny_internal_experience": row["deny_internal_experience"] == "TRUE",
                }
        print(f"  Loaded {len(ratings_lookup)} ratings entries")
    except Exception as e:
        print(f"  Error reading ratings file: {e}")

    return ratings_lookup


def main():
    print(f"Building site data from {DATA_SRC} to {DATA_DEST}...")

    # Ensure docs/data exists
    if DATA_DEST.exists():
        shutil.rmtree(DATA_DEST)
    DATA_DEST.mkdir(parents=True, exist_ok=True)

    manifest = {"models": []}

    # Iterate over models
    if not DATA_SRC.exists():
        print(f"Source directory {DATA_SRC} does not exist!")
        return

    for model_dir in sorted(DATA_SRC.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        print(f"Processing model: {model_name}")

        # Create model dir in dest
        dest_model_dir = DATA_DEST / model_name
        dest_model_dir.mkdir(exist_ok=True)

        model_info = {"name": model_name, "roles": [], "config": None}

        # Load ratings for this model
        ratings_lookup = load_ratings(model_name)

        # Process files
        for item in sorted(model_dir.iterdir()):
            if item.name == "config.toml":
                # Copy config and read it
                shutil.copy2(item, dest_model_dir / item.name)
                try:
                    with open(item, "rb") as f:
                        config_data = tomllib.load(f)
                        model_info["config"] = config_data
                except Exception as e:
                    print(f"Error reading config.toml for {model_name}: {e}")

            elif item.suffix == ".json":
                # It's a role file
                role_name = item.stem
                model_info["roles"].append(role_name)

                # Read, merge ratings, and write to destination
                try:
                    with open(item, "r", encoding="utf-8") as f:
                        role_data = json.load(f)

                    # Merge ratings into each response entry
                    for entry in role_data:
                        key = (
                            entry["role_name"],
                            entry["task_name"],
                            entry["sample_idx"],
                        )
                        if key in ratings_lookup:
                            entry["ratings"] = ratings_lookup[key]

                    # Write merged data to destination
                    with open(dest_model_dir / item.name, "w", encoding="utf-8") as f:
                        json.dump(role_data, f, indent=2)

                except Exception as e:
                    print(f"  Error processing {item.name}: {e}")
                    # Fall back to simple copy if something goes wrong
                    shutil.copy2(item, dest_model_dir / item.name)

        manifest["models"].append(model_info)

    # Write manifest
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Build complete. Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
