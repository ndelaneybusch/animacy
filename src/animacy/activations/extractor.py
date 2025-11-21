"""
ActivationExtractor - Extract hidden state activations from model layers.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from .token_mapper import ActivationResult


class ActivationExtractor:
    """
    Extract activations from model layers using forward hooks.

    Optimized for parallel extraction from full input texts.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
    ):
        """
        Initialize the activation extractor.

        Args:
            model_name_or_path: HuggingFace model identifier
            device: Device to load model on (e.g., "cuda", "cpu", "auto")
            torch_dtype: Data type for model weights
        """
        self.model_name = model_name_or_path

        # Determine device map
        if device is None:
            device_map = "auto"
        else:
            device_map = device

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype or "auto",
            trust_remote_code=True,
        )
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._layers = self._find_layers()

    def _find_layers(self) -> nn.ModuleList:
        """
        Find the transformer layers in the model.

        Returns:
            ModuleList of transformer layers
        """
        # Try common paths for transformer layers
        layer_paths = [
            ("model.layers", lambda m: m.model.layers),  # Llama, Mistral, Qwen, Gemma
            ("model.model.layers", lambda m: m.model.model.layers),  # Some variants
            ("transformer.h", lambda m: m.transformer.h),  # GPT-2, GPT-J
            ("model.transformer.h", lambda m: m.model.transformer.h),  # Bloom
            ("gpt_neox.layers", lambda m: m.gpt_neox.layers),  # GPT-NeoX
            ("model.decoder.layers", lambda m: m.model.decoder.layers),  # OPT
        ]

        for _, path_func in layer_paths:
            try:
                layers = path_func(self.model)
                if layers is not None and len(layers) > 0:
                    return layers
            except AttributeError:
                continue

        # Fallback: try to find by type
        # This is a bit hacky but works for many models if the above fails
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                # Check if it looks like a layer list (usually has many identical blocks)
                if len(module) > 5:  # Heuristic
                    return module

        raise AttributeError(
            f"Could not find transformer layers for model {self.model_name}"
        )

    def extract(
        self,
        text: str | list[str],
        layers: list[int] | None = None,
        batch_size: int = 8,
    ) -> ActivationResult:
        """
        Extract activations for the given text(s).

        Args:
            text: Input text or list of texts
            layers: List of layer indices to extract from. If None, extracts from all layers.
            batch_size: Batch size for processing multiple texts

        Returns:
            ActivationResult containing activations and token mapping info
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        if layers is None:
            layers = list(range(len(self._layers)))

        # Tokenize
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        input_ids = encodings["input_ids"].to(self.model.device)
        attention_mask = encodings["attention_mask"].to(self.model.device)
        offset_mapping = encodings["offset_mapping"]  # Keep on CPU

        # Prepare hooks
        activations = {layer_idx: [] for layer_idx in layers}
        handles = []

        def create_hook(layer_idx):
            def hook(module, input, output):
                # Handle tuple output (hidden_states, present_key_value_states, ...)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # Detach and move to CPU to save GPU memory
                # Shape: (batch_size, seq_len, hidden_size)
                activations[layer_idx].append(hidden_states.detach().cpu())

            return hook

        # Register hooks
        for layer_idx in layers:
            layer_module = self._layers[layer_idx]
            handle = layer_module.register_forward_hook(create_hook(layer_idx))
            handles.append(handle)

        try:
            # Process in batches if needed (though here we just did one big batch for simplicity
            # based on the tokenizer call above. For very large inputs, we'd loop.)
            # Given the requirement "Optimize for this use case" (parallel from entire input),
            # processing the whole input at once is good unless it OOMs.
            # For now, we assume the input fits in memory or the user batches the calls.

            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask)

        finally:
            for handle in handles:
                handle.remove()

        # Collate activations
        # Each layer has a list of tensors (one per forward pass, here just one)
        # We want a dictionary mapping layer_idx -> Tensor(batch, seq, hidden)
        final_activations = {}
        for layer_idx, act_list in activations.items():
            final_activations[layer_idx] = torch.cat(act_list, dim=0)

        return ActivationResult(
            activations=final_activations,
            input_ids=input_ids.cpu(),
            offset_mapping=offset_mapping,
            tokenizer=self.tokenizer,
            texts=texts,
        )
