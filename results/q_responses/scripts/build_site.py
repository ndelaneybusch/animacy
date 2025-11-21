import os
import json
import shutil
import tomllib
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent.parent
DATA_SRC = REPO_ROOT / "results" / "q_responses" / "data"
DOCS_DIR = REPO_ROOT / "docs"
DATA_DEST = DOCS_DIR / "data"
MANIFEST_PATH = DOCS_DIR / "data_manifest.json"


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
                shutil.copy2(item, dest_model_dir / item.name)

        manifest["models"].append(model_info)

    # Write manifest
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Build complete. Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
