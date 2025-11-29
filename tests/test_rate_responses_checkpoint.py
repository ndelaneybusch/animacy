import sys
import json
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts directory to path to import rate_responses
# Assuming the tests are run from the root of the repo or we can find the file relative to this test file
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "results" / "ratings" / "scripts"
sys.path.append(str(SCRIPTS_DIR))

import rate_responses


@pytest.fixture
def dummy_data():
    return [
        {
            "role_name": "RoleA",
            "task_name": "Task1",
            "sample_idx": 0,
            "response": "Response A1",
        },
        {
            "role_name": "RoleA",
            "task_name": "Task2",
            "sample_idx": 0,
            "response": "Response A2",
        },
        {
            "role_name": "RoleB",
            "task_name": "Task1",
            "sample_idx": 0,
            "response": "Response B1",
        },
    ]


@pytest.fixture
def dummy_checkpoint_df():
    return pd.DataFrame(
        [
            {
                "role_name": "RoleA",
                "task_name": "Task1",
                "sample_idx": 0,
                "response": "Response A1",
                "assistant_refusal": False,
                "role_refusal": False,
                "identify_as_assistant": False,
                "deny_internal_experience": False,
                "role_adherence": 100,
                "error": None,
            },
            {
                "role_name": "RoleA",
                "task_name": "Task2",
                "sample_idx": 0,
                "response": "Response A2",
                "assistant_refusal": None,
                "role_refusal": None,
                "identify_as_assistant": None,
                "deny_internal_experience": None,
                "role_adherence": None,
                "error": "Error processing",
            },
        ]
    )


def test_load_checkpoint(tmp_path, dummy_checkpoint_df):
    checkpoint_path = tmp_path / "checkpoint.csv"
    dummy_checkpoint_df.to_csv(checkpoint_path, index=False)

    df, skip_keys = rate_responses.load_checkpoint(checkpoint_path)

    assert len(df) == 1
    assert df.iloc[0]["role_name"] == "RoleA"
    assert df.iloc[0]["task_name"] == "Task1"

    assert len(skip_keys) == 1
    assert ("RoleA", "Task1", 0) in skip_keys
    assert ("RoleA", "Task2", 0) not in skip_keys


def test_process_folder_with_checkpoint(tmp_path, dummy_data, dummy_checkpoint_df):
    # Setup input files
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    with open(input_dir / "data.json", "w") as f:
        json.dump(dummy_data, f)

    # Setup checkpoint
    checkpoint_path = tmp_path / "checkpoint.csv"
    dummy_checkpoint_df.to_csv(checkpoint_path, index=False)

    # Mock process_item
    def mock_process_item(item, model_name, provider):
        return {
            "role_name": item["role_name"],
            "task_name": item["task_name"],
            "sample_idx": item["sample_idx"],
            "response": item["response"],
            "assistant_refusal": False,
            "role_refusal": False,
            "identify_as_assistant": False,
            "deny_internal_experience": False,
            "role_adherence": 50,  # Distinguishable from checkpoint's 100
        }

    with patch("rate_responses.process_item", side_effect=mock_process_item):
        # We can't easily call main() because it parses args.
        # But we can replicate the main logic or refactor main to be testable.
        # For now, let's replicate the integration logic from main.

        # 1. Load checkpoint
        existing_df, skip_keys = rate_responses.load_checkpoint(checkpoint_path)

        # 2. Process folder
        new_results_df = rate_responses.process_folder(
            input_dir, model_name="test-model", provider="openai", skip_keys=skip_keys
        )

        # 3. Combine
        final_df = pd.concat([existing_df, new_results_df], ignore_index=True)

        # Verify
        assert len(final_df) == 3

        # Row 1: From checkpoint
        row1 = final_df[
            (final_df["role_name"] == "RoleA")
            & (final_df["task_name"] == "Task1")
            & (final_df["sample_idx"] == 0)
        ]
        assert len(row1) == 1
        assert row1.iloc[0]["role_adherence"] == 100

        # Row 2: Re-processed (was error in checkpoint)
        row2 = final_df[
            (final_df["role_name"] == "RoleA")
            & (final_df["task_name"] == "Task2")
            & (final_df["sample_idx"] == 0)
        ]
        assert len(row2) == 1
        assert row2.iloc[0]["role_adherence"] == 50

        # Row 3: New
        row3 = final_df[
            (final_df["role_name"] == "RoleB")
            & (final_df["task_name"] == "Task1")
            & (final_df["sample_idx"] == 0)
        ]
        assert len(row3) == 1
        assert row3.iloc[0]["role_adherence"] == 50


if __name__ == "__main__":
    # Allow running this script directly
    pytest.main([__file__])
