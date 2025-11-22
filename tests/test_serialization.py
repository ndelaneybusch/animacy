import json

import numpy as np
import pytest

from animacy.activations import ActivationSummaries


def test_activation_summaries_serialization():
    """Test that ActivationSummaries can be serialized to JSON even with numpy arrays."""

    # Create dummy data
    summary = ActivationSummaries(
        avg_system_prompt=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        avg_user_prompt=np.array([0.4, 0.5, 0.6], dtype=np.float32),
        avg_response=np.array([0.7, 0.8, 0.9], dtype=np.float32),
        avg_response_first_10_tokens=np.array([1.0, 1.1], dtype=np.float32),
        at_role=None,
        at_role_period=None,
        at_end_system_prompt=None,
        at_end_user_prompt=None,
        at_start_agent_response=None,
    )

    # Try to serialize
    try:
        json_str = summary.model_dump_json()

        # Verify it's valid JSON and contains expected values
        data = json.loads(json_str)
        # Floating point comparison
        assert np.allclose(data["avg_system_prompt"], [0.1, 0.2, 0.3], atol=1e-6)

    except Exception as e:
        pytest.fail(f"Serialization failed: {e}")
