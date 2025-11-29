"""
ActivationExtractor - Extract hidden state activations from model layers.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

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
            model_name_or_path, trust_remote_code=True
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
                    print(
                        f"DEBUG: Found layers at path '{path_name}': {len(layers)} layers of type {type(layers[0]).__name__}"
                    )
                    return layers
            except AttributeError:
                continue

        # Fallback: try to find by type
        # This is a bit hacky but works for many models if the above fails
        # Prioritize finding language model layers over vision encoder layers
        print("DEBUG: Using fallback layer detection")
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
                        print(
                            f"DEBUG: Skipping vision encoder layers at '{name}': {len(module)} layers of type {first_layer_type}"
                        )
                        continue
                    print(
                        f"DEBUG: Found layers at '{name}': {len(module)} layers of type {first_layer_type}"
                    )
                    return module

        raise AttributeError(
            f"Could not find transformer layers for model {self.model_name}"
        )

    def extract(
        self,
        prompts: str | list[str] | list[list[dict]],
        layers: list[int] | None = None,
        batch_size: int = 8,
    ) -> ActivationResult:
        """
        Extract activations for the given text(s) or chat messages.

        Args:
            prompts: Input text, list of texts, or list of chat histories (list of dicts).
                     If chat histories are provided, precise message alignment is performed.
            layers: List of layer indices to extract from. If None, extracts from all layers.
            batch_size: Batch size for processing multiple texts

        Returns:
            ActivationResult containing activations and token mapping info
        """
        texts = []
        message_ranges = None

        # Handle different input types
        if isinstance(prompts, str):
            texts = [prompts]
        elif isinstance(prompts, list):
            if not prompts:
                raise ValueError("Prompts list cannot be empty")

            if isinstance(prompts[0], str):
                texts = prompts  # type: ignore
            elif isinstance(prompts[0], list) and isinstance(prompts[0][0], dict):
                # List of chat histories
                texts, message_ranges = self._process_chat_inputs(prompts)  # type: ignore
            else:
                raise ValueError(
                    "Invalid prompt format. Must be str, list[str], or list[list[dict]]"
                )
        else:
            raise ValueError(
                "Invalid prompt format. Must be str, list[str], or list[list[dict]]"
            )

        if layers is None:
            layers = list(range(len(self._layers)))
        else:
            # Validate layer indices
            num_layers = len(self._layers)
            invalid_layers = [l for l in layers if l < 0 or l >= num_layers]
            if invalid_layers:
                raise ValueError(
                    f"Invalid layer indices {invalid_layers}. "
                    f"Model {self.model_name} has {num_layers} layers (valid indices: 0-{num_layers - 1})"
                )

        # Tokenize
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=False,  # We assume chat template handles this or user provided raw text
        )

        input_ids = encodings["input_ids"].to(self.model.device)
        attention_mask = encodings["attention_mask"].to(self.model.device)
        offset_mapping = encodings["offset_mapping"]  # Keep on CPU

        # Prepare hooks
        activations: dict[int, list[torch.Tensor]] = {
            layer_idx: [] for layer_idx in layers
        }
        handles = []
        hook_fired = {layer_idx: False for layer_idx in layers}

        def create_hook(layer_idx):
            def hook(module, input, output):
                hook_fired[layer_idx] = True
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
        print(f"DEBUG: Registering hooks for layers {layers}")
        for layer_idx in layers:
            layer_module = self._layers[layer_idx]
            print(
                f"DEBUG: Registering hook on layer {layer_idx}: {type(layer_module).__name__}"
            )
            handle = layer_module.register_forward_hook(create_hook(layer_idx))
            handles.append(handle)

        try:
            # Process in batches if needed (though here we just did one big batch for simplicity
            # based on the tokenizer call above. For very large inputs, we'd loop.)
            print(f"DEBUG: Running forward pass with input_ids shape {input_ids.shape}")
            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask)
            print(f"DEBUG: Forward pass complete")

        finally:
            for handle in handles:
                handle.remove()

        # Check which hooks fired
        unfired_hooks = [idx for idx, fired in hook_fired.items() if not fired]
        if unfired_hooks:
            print(f"WARNING: Hooks did not fire for layers: {unfired_hooks}")

        # Collate activations
        final_activations = {}
        for layer_idx, act_list in activations.items():
            if not act_list:
                raise RuntimeError(
                    f"No activations captured for layer {layer_idx}. "
                    f"Hook may not have fired. Model: {self.model_name}"
                )
            final_activations[layer_idx] = torch.cat(act_list, dim=0)

        return ActivationResult(
            activations=final_activations,
            input_ids=input_ids.cpu(),
            offset_mapping=offset_mapping,
            tokenizer=self.tokenizer,
            texts=texts,
            message_ranges=message_ranges,
        )

    def _process_chat_inputs(
        self, chat_histories: list[list[dict]]
    ) -> tuple[list[str], list[list[dict]]]:
        """
        Process chat histories into full texts and extract message ranges.

        Args:
            chat_histories: List of conversation histories

        Returns:
            Tuple of (list of full texts, list of message ranges)
        """
        from ..activations.data import _detect_assistant_role_name

        # Detect the expected assistant role name for this tokenizer
        expected_assistant_role = _detect_assistant_role_name(self.tokenizer)

        texts = []
        all_message_ranges = []

        for history in chat_histories:
            # Normalize role names to match what the tokenizer expects
            normalized_history = []
            for message in history:
                role = message["role"]
                # Convert 'assistant' to the tokenizer's expected role name
                if role == "assistant" and expected_assistant_role != "assistant":
                    role = expected_assistant_role
                normalized_history.append({"role": role, "content": message["content"]})

            # Apply chat template to get full text
            full_text = self.tokenizer.apply_chat_template(
                normalized_history, tokenize=False, add_generation_prompt=False
            )
            texts.append(full_text)

            # Find ranges for each message
            # We iterate through messages and find their content in the full text
            # This assumes the content appears verbatim in the output (standard behavior)
            ranges = []
            current_search_idx = 0

            for message in normalized_history:
                content = message["content"]
                role = message["role"]

                if not content:
                    continue

                start_idx = full_text.find(content, current_search_idx)

                if start_idx == -1:
                    # Warning: Content not found. This might happen if the template
                    # transforms the content significantly.
                    print(
                        f"WARNING: Content not found for role '{role}' in full text. "
                        f"Content start: '{content[:20]}...'"
                    )
                    continue

                end_idx = start_idx + len(content)

                ranges.append(
                    {
                        "role": role,
                        "start": start_idx,
                        "end": end_idx,
                        "content": content,  # Optional, for debugging
                    }
                )

                # Update search index to avoid finding the same content again
                # (though unlikely to overlap in a valid chat)
                current_search_idx = end_idx

            all_message_ranges.append(ranges)

        return texts, all_message_ranges
