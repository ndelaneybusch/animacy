"""
Evaluation tools for steered models.
"""

from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from animacy.analysis.logits import LogitExtractor, ResponseLogits

from .core import SteeringManager


def evaluate_steered_logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    steering_vectors: dict[int, torch.Tensor],
    layers: list[int],
    magnitude: float,
    samples: list[dict[str, Any]],
    use_system_prompt: bool = True,
) -> list[ResponseLogits]:
    """
    Evaluate logits for a set of samples while applying steering vectors.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer.
        steering_vectors: Dictionary mapping layer indices to steering vectors.
        layers: List of layers to steer.
        magnitude: Steering magnitude.
        samples: List of dictionaries containing sample data. Each dict must have:
                 - role_name (str | None)
                 - task_name (str)
                 - sample_idx (int)
                 - response (str)
        use_system_prompt: Whether to use the system prompt in logit extraction.

    Returns:
        List of ResponseLogits objects.
    """
    steering_manager = SteeringManager(model, tokenizer)
    logit_extractor = LogitExtractor(model, tokenizer)

    results = []

    # Apply steering context
    with steering_manager.apply_steering(steering_vectors, layers, magnitude):
        for sample in samples:
            # Extract fields from sample dict
            role_name = sample.get("role_name")
            task_name = sample["task_name"]
            sample_idx = sample["sample_idx"]
            response_text = sample["response"]

            # Extract logits
            # Note: extract_logits handles tokenization and model forward pass internally
            # Since we are inside the context manager, the model forward pass will be steered
            logits_result = logit_extractor.extract_logits(
                role_name=role_name,
                task_name=task_name,
                sample_idx=sample_idx,
                response_text=response_text,
                use_system_prompt=use_system_prompt,
            )
            results.append(logits_result)

    return results
