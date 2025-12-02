"""
Core steering functionality for adding vectors to model activations.
"""

import contextlib
from collections.abc import Generator, Iterable

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

    @contextlib.contextmanager
    def apply_steering(
        self,
        steering_vectors: dict[int, torch.Tensor],
        layers: Iterable[int],
        magnitude: float = 1.0,
    ) -> Generator[None, None, None]:
        """
        Apply steering vectors to the specified layers using forward hooks.

        Args:
            steering_vectors: Dictionary mapping layer indices to steering vectors.
            layers: Iterable of layer indices to apply steering to.
            magnitude: Multiplier for the steering vector strength.
        """
        handles = []

        # Prepare vectors: normalize and scale
        processed_vectors = {}
        for layer_idx in layers:
            if layer_idx not in steering_vectors:
                # If a layer is requested but no vector provided, we skip it or raise error?
                # The prompt said "Inputs... include a dictionary of steering vectors... and an iterable of layers. If multiple layers are passed, steer all of them."
                # I'll assume we only steer layers that are both in `layers` and `steering_vectors`.
                # Or maybe `layers` defines which keys to look up.
                continue

            vector = steering_vectors[layer_idx]

            # Normalize to unit norm
            # Handle both 1D (hidden_dim) and shaped tensors
            if vector.dim() > 1:
                # Assuming the last dimension is hidden_dim
                norm = torch.norm(vector, p=2, dim=-1, keepdim=True)
                normalized_vector = vector / (norm + 1e-8)
            else:
                norm = torch.norm(vector, p=2)
                normalized_vector = vector / (norm + 1e-8)

            # Scale by magnitude
            scaled_vector = normalized_vector * magnitude

            # Move to model device and dtype
            processed_vectors[layer_idx] = scaled_vector.to(
                device=self.model.device, dtype=self.model.dtype
            )

        def create_hook(layer_idx, vector):
            def hook(module, input, output):
                # output is usually a tuple (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    # Add vector to hidden states
                    # Vector shape: (hidden_dim,) or broadcastable
                    # Hidden states shape: (batch, seq_len, hidden_dim)
                    steered_hidden_states = hidden_states + vector

                    # Return new tuple with modified hidden states
                    return (steered_hidden_states,) + output[1:]
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
