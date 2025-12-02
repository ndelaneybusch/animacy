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
    batch_size: int = 1,
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
                 - system_prompt (str, optional): Custom system prompt
                 - task_prompt (str, optional): Custom task/user prompt
        use_system_prompt: Whether to use the system prompt in logit extraction.
        batch_size: Batch size for processing.

    Returns:
        List of ResponseLogits objects.
    """
    steering_manager = SteeringManager(model, tokenizer)
    logit_extractor = LogitExtractor(model, tokenizer)

    results = []

    # Pre-process vectors once
    prepared_vectors = steering_manager.prepare_vectors(
        steering_vectors, layers, magnitude
    )

    # Apply steering context
    with steering_manager.apply_steering(
        prepared_vectors, layers, magnitude, pre_processed=True
    ):
        # Process in batches
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i : i + batch_size]

            batch_results = logit_extractor.extract_logits_batch(
                batch_samples,
                use_system_prompt=use_system_prompt,
                steering_manager=steering_manager,
            )
            results.extend(batch_results)

    return results
