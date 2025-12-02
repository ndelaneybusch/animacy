"""
Tests for the steering module.
"""

import pytest
import torch
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer

from animacy.steering.core import SteeringManager
from animacy.steering.evaluation import evaluate_steered_logits


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize(
    "model_name", ["Qwen/Qwen2.5-0.5B-Instruct", "google/gemma-3-270m-it"]
)
def test_steering_effect(model_name):
    """
    Test that steering actually changes the model output.
    """
    print(f"\nTesting steering with {model_name}...")

    device = get_device()

    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=None, torch_dtype="auto", trust_remote_code=True
        ).to(device)
    except (GatedRepoError, RepositoryNotFoundError, OSError) as e:
        pytest.skip(f"Skipping {model_name} due to access error: {e}")

    manager = SteeringManager(model, tokenizer)

    # Find a middle layer to steer
    layers = manager._layers
    layer_idx = len(layers) // 2
    hidden_size = model.config.hidden_size

    # Create a random steering vector
    steering_vector = torch.randn(hidden_size, device=model.device, dtype=model.dtype)
    steering_vectors = {layer_idx: steering_vector}

    # Input text
    text = "Hello, world!"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Run unsteered
    with torch.no_grad():
        output_unsteered = model(**inputs).logits

    # Run steered
    with manager.apply_steering(steering_vectors, [layer_idx], magnitude=10.0):
        with torch.no_grad():
            output_steered = model(**inputs).logits

    # Check that outputs are different
    diff = torch.norm(output_steered - output_unsteered)
    print(f"Difference norm: {diff.item()}")
    assert diff.item() > 0.1, "Steering should significantly change the output"

    # Check that hooks are removed
    with torch.no_grad():
        output_after = model(**inputs).logits

    diff_after = torch.norm(output_after - output_unsteered)
    print(f"Difference after hook removal: {diff_after.item()}")
    assert diff_after.item() < 1e-5, "Hooks should be removed after context exit"


def test_multi_layer_steering():
    """
    Test steering multiple layers.
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\nTesting multi-layer steering with {model_name}...")

    device = get_device()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=None, torch_dtype="auto", trust_remote_code=True
        ).to(device)
    except Exception as e:
        pytest.skip(f"Skipping due to model load error: {e}")

    manager = SteeringManager(model, tokenizer)
    layers = manager._layers
    hidden_size = model.config.hidden_size

    # Steer two layers
    layer_indices = [len(layers) // 3, 2 * len(layers) // 3]
    steering_vector = torch.randn(hidden_size, device=model.device, dtype=model.dtype)
    steering_vectors = {idx: steering_vector for idx in layer_indices}

    text = "Hello, world!"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Run unsteered
    with torch.no_grad():
        output_unsteered = model(**inputs).logits

    # Run steered
    with manager.apply_steering(steering_vectors, layer_indices, magnitude=5.0):
        with torch.no_grad():
            output_steered = model(**inputs).logits

    # Check difference
    diff = torch.norm(output_steered - output_unsteered)
    print(f"Difference norm: {diff.item()}")
    assert diff.item() > 0.1


def test_evaluate_steered_logits():
    """
    Test the high-level evaluation function.
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\nTesting evaluate_steered_logits with {model_name}...")

    device = get_device()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=None, torch_dtype="auto", trust_remote_code=True
        ).to(device)
    except Exception as e:
        pytest.skip(f"Skipping due to model load error: {e}")

    hidden_size = model.config.hidden_size
    layer_idx = 10
    steering_vector = torch.randn(hidden_size, device=model.device, dtype=model.dtype)
    steering_vectors = {layer_idx: steering_vector}

    samples = [
        {
            "role_name": "assistant",
            "task_name": "meaning_of_life",
            "sample_idx": 0,
            "response": "The meaning of life is 42.",
        }
    ]

    # We need to ensure TASK_PROMPTS has "meaning_of_life"
    # Assuming it's in the codebase, otherwise this test might fail if imports fail or key missing
    # But let's assume it works as per existing code structure

    results = evaluate_steered_logits(
        model=model,
        tokenizer=tokenizer,
        steering_vectors=steering_vectors,
        layers=[layer_idx],
        magnitude=1.0,
        samples=samples,
        use_system_prompt=True,
    )

    assert len(results) == 1
    assert results[0].task_name == "meaning_of_life"
    assert results[0].average_log_probs is not None
    print("Evaluation successful")


if __name__ == "__main__":
    # Allow running directly
    test_steering_effect("Qwen/Qwen2.5-0.5B-Instruct")
    test_steering_effect("google/gemma-3-270m-it")
    test_multi_layer_steering()
    test_evaluate_steered_logits()
