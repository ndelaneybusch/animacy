"""
Test to verify if steering has any impact on Gemma 3 models.
"""

import torch
import pytest
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from animacy.steering.core import SteeringManager
from animacy.steering.evaluation import evaluate_steered_logits


@pytest.mark.parametrize("model_name", ["google/gemma-3-270m-it"])
def test_gemma_random_steering_impact(model_name):
    """
    Test that applying a random steering vector with high magnitude
    has a significant impact on the model's logits.
    """
    print(f"\nTesting steering impact on {model_name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True
    )

    # Define diverse prompts
    prompts = [
        {
            "role_name": "assistant",
            "task_name": "greeting",
            "sample_idx": 0,
            "response": "Hello! How can I help you today?",
            "system_prompt": "You are a helpful assistant.",
            "task_prompt": "Hello",
        },
        {
            "role_name": "assistant",
            "task_name": "math",
            "sample_idx": 1,
            "response": "The answer is 4.",
            "system_prompt": "You are a math tutor.",
            "task_prompt": "What is 2+2?",
        },
        {
            "role_name": "assistant",
            "task_name": "joke",
            "sample_idx": 2,
            "response": "Why did the chicken cross the road? To get to the other side!",
            "system_prompt": "You are a comedian.",
            "task_prompt": "Tell me a joke",
        },
    ]

    # Initialize steering manager to find layers
    manager = SteeringManager(model, tokenizer)
    layers = manager._find_layers()
    num_layers = len(layers)
    target_layer = num_layers // 2

    print(f"Model has {num_layers} layers. Targeting layer {target_layer}.")

    # Generate random steering vector
    hidden_size = model.config.hidden_size
    random_vector = torch.randn(hidden_size)
    steering_vectors = {target_layer: random_vector}

    # Baseline: No steering (magnitude 0)
    print("Running baseline (magnitude 0)...")
    baseline_results = evaluate_steered_logits(
        model=model,
        tokenizer=tokenizer,
        steering_vectors=steering_vectors,
        layers=[target_layer],
        magnitude=0.0,
        samples=prompts,
        use_system_prompt=True,
    )

    # Steered: High magnitude
    # Gemma 3 has very large activation norms (~40k), so we need a large magnitude
    magnitude = 5000.0
    print(f"Running steered (magnitude {magnitude})...")
    steered_results = evaluate_steered_logits(
        model=model,
        tokenizer=tokenizer,
        steering_vectors=steering_vectors,
        layers=[target_layer],
        magnitude=magnitude,
        samples=prompts,
        use_system_prompt=True,
    )

    # Compare logits
    for i, (base, steered) in enumerate(zip(baseline_results, steered_results)):
        print(f"\nPrompt {i}: {prompts[i]['task_prompt']}")

        # Check if average_log_probs is None (which might happen if role detection fails)
        if base.average_log_probs is None or steered.average_log_probs is None:
            print("WARNING: average_log_probs is None. Skipping logit comparison.")
            continue

        diff = abs(base.average_log_probs - steered.average_log_probs)
        print(f"Baseline avg logit: {base.average_log_probs}")
        print(f"Steered avg logit: {steered.average_log_probs}")
        print(f"Difference: {diff}")

        # Assert significant difference
        # With magnitude 100, we expect a massive disruption
        assert diff > 1.0, (
            f"Steering had insufficient impact on prompt {i}. Diff: {diff}"
        )


if __name__ == "__main__":
    test_gemma_random_steering_impact("google/gemma-3-270m-it")
