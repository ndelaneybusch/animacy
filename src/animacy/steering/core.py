"""
Core steering functionality for adding vectors to model activations.
"""

import contextlib
from collections.abc import Generator, Iterable
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer


class SteeringManager:
    """
    Manages the application of steering vectors to model layers.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Initialize the steering manager.

        Args:
            model: The HuggingFace model to steer.
            tokenizer: The tokenizer associated with the model.
        """
        self.model = model
        self.tokenizer = tokenizer
        self._layers = self._find_layers()
        self._current_attention_mask: torch.Tensor | None = None

    def _find_layers(self) -> nn.ModuleList:
        """
        Find the transformer layers in the model.

        Duplicated from ActivationExtractor to ensure robustness to multimodal models.

        Returns:
            ModuleList of transformer layers
        """
        # Try common paths for transformer layers
        layer_paths = [
            # Language model paths (prioritize these for multimodal models)
            (
                "model.language_model.model.layers",
                lambda m: m.model.language_model.model.layers,
            ),  # Multimodal Gemma
            (
                "language_model.model.layers",
                lambda m: m.language_model.model.layers,
            ),  # Some multimodal variants
            (
                "model.layers",
                lambda m: m.model.layers,
            ),  # Llama, Mistral, Qwen, Gemma (text-only)
            ("model.model.layers", lambda m: m.model.model.layers),  # Some variants
            ("transformer.h", lambda m: m.transformer.h),  # GPT-2, GPT-J
            ("model.transformer.h", lambda m: m.model.transformer.h),  # Bloom
            ("gpt_neox.layers", lambda m: m.gpt_neox.layers),  # GPT-NeoX
            ("model.decoder.layers", lambda m: m.model.decoder.layers),  # OPT
        ]

        for path_name, path_func in layer_paths:
            try:
                layers = path_func(self.model)
                if layers is not None and len(layers) > 0:
                    return layers
            except AttributeError:
                continue

        # Fallback: try to find by type
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 5:
                # Skip vision encoder layers
                if len(module) > 0:
                    first_layer_type = type(module[0]).__name__
                    # Skip known vision encoder layer types
                    if (
                        "Vision" in first_layer_type
                        or "Siglip" in first_layer_type
                        or "Clip" in first_layer_type
                    ):
                        continue
                    return module

        raise AttributeError(
            f"Could not find transformer layers for model {type(self.model).__name__}"
        )

    def prepare_vectors(
        self,
        steering_vectors: dict[int, torch.Tensor | Any],
        layers: Iterable[int],
        magnitude: float = 1.0,
    ) -> dict[int, torch.Tensor]:
        """
        Pre-process steering vectors: convert to tensor, normalize, scale, and move to device.

        Args:
            steering_vectors: Dictionary mapping layer indices to vectors (torch.Tensor or numpy array).
            layers: Iterable of layer indices to prepare vectors for.
            magnitude: Scaling factor.

        Returns:
            Dictionary of processed vectors ready for the model.
        """
        processed_vectors = {}
        for layer_idx in layers:
            if layer_idx not in steering_vectors:
                continue

            vector = steering_vectors[layer_idx]

            # Convert from numpy if needed
            if not isinstance(vector, torch.Tensor):
                vector = torch.from_numpy(vector)

            # Move to device/dtype first to ensure operations happen on GPU if possible
            # But normalization is safer/cleaner if we ensure it's a float tensor first
            vector = vector.to(device=self.model.device, dtype=self.model.dtype)

            # Normalize to unit norm
            if vector.dim() > 1:
                norm = torch.norm(vector, p=2, dim=-1, keepdim=True)
                normalized_vector = vector / (norm + 1e-8)
            else:
                norm = torch.norm(vector, p=2)
                normalized_vector = vector / (norm + 1e-8)

            # Scale by magnitude
            scaled_vector = normalized_vector * magnitude

            processed_vectors[layer_idx] = scaled_vector

        return processed_vectors

    @contextlib.contextmanager
    def apply_steering(
        self,
        steering_vectors: dict[int, torch.Tensor],
        layers: Iterable[int],
        magnitude: float = 1.0,
        pre_processed: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> Generator[None, None, None]:
        """
        Apply steering vectors to the specified layers using forward hooks.

        Args:
            steering_vectors: Dictionary mapping layer indices to steering vectors.
            layers: Iterable of layer indices to apply steering to.
            magnitude: Multiplier for the steering vector strength.
            pre_processed: If True, assumes vectors are already normalized,
                           scaled, and on device.
            attention_mask: Optional attention mask to apply steering only to
                           non-padded tokens. Shape: (batch_size, seq_len).
                           If None, applies to all positions.
        """
        handles = []

        if pre_processed:
            processed_vectors = steering_vectors
        else:
            processed_vectors = self.prepare_vectors(
                steering_vectors, layers, magnitude
            )

        def create_hook(layer_idx, vector):
            def hook(module, input, output):
                # output is usually a tuple (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    # Add vector to hidden states
                    # Vector shape: (hidden_dim,) or broadcastable
                    # Hidden states shape: (batch, seq_len, hidden_dim)

                    # Use the attention mask stored in the manager
                    mask_to_use = (
                        attention_mask
                        if attention_mask is not None
                        else self._current_attention_mask
                    )

                    if mask_to_use is not None:
                        # Only apply steering to non-padded positions
                        # Expand mask to match hidden_states shape: (batch, seq_len, 1)
                        mask = mask_to_use.unsqueeze(-1).to(hidden_states.dtype)
                        steered_hidden_states = hidden_states + vector * mask
                    else:
                        steered_hidden_states = hidden_states + vector

                    # Return new tuple with modified hidden states
                    return (steered_hidden_states,) + output[1:]
                else:
                    mask_to_use = (
                        attention_mask
                        if attention_mask is not None
                        else self._current_attention_mask
                    )

                    if mask_to_use is not None:
                        mask = mask_to_use.unsqueeze(-1).to(output.dtype)
                        return output + vector * mask
                    else:
                        return output + vector

            return hook

        try:
            for layer_idx in layers:
                if layer_idx in processed_vectors:
                    layer_module = self._layers[layer_idx]
                    vector = processed_vectors[layer_idx]
                    handle = layer_module.register_forward_hook(
                        create_hook(layer_idx, vector)
                    )
                    handles.append(handle)

            yield

        finally:
            for handle in handles:
                handle.remove()
