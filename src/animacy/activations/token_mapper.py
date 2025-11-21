"""
TokenMapper - Helper classes for mapping text to tokens and activations.
"""

import torch
from transformers import PreTrainedTokenizer


class ActivationResult:
    """
    Stores the result of an activation extraction and provides methods to query it.
    """

    def __init__(
        self,
        activations: dict[int, torch.Tensor],
        input_ids: torch.Tensor,
        offset_mapping: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        texts: list[str],
    ):
        """
        Initialize ActivationResult.

        Args:
            activations: Dictionary mapping layer index to activation tensor (batch, seq, hidden)
            input_ids: Input token IDs (batch, seq)
            offset_mapping: Token character offsets (batch, seq, 2)
            tokenizer: Tokenizer used
            texts: Original input texts
        """
        self.activations = activations
        self.input_ids = input_ids
        self.offset_mapping = offset_mapping
        self.tokenizer = tokenizer
        self.texts = texts

    def get_activations_for_text(self, text_index: int = 0) -> dict[int, torch.Tensor]:
        """
        Get all activations for a specific text in the batch.

        Args:
            text_index: Index of the text in the input list

        Returns:
            Dict mapping layer index to activation tensor (seq, hidden)
        """
        result = {}
        for layer, act in self.activations.items():
            result[layer] = act[text_index]
        return result

    def get_token_activation(
        self, token_index: int, layer: int, text_index: int = 0
    ) -> torch.Tensor:
        """
        Get activation for a specific token index.

        Args:
            token_index: Index of the token
            layer: Layer index
            text_index: Index of the text in the batch

        Returns:
            Activation vector (hidden_size,)
        """
        return self.activations[layer][text_index, token_index]

    def get_span_activation(
        self, span: str, layer: int, text_index: int = 0, aggregation: str = "mean"
    ) -> torch.Tensor | None:
        """
        Get activation for a specific text span.

        Finds the first occurrence of the span in the text.

        Args:
            span: Text span to find
            layer: Layer index
            text_index: Index of the text in the batch
            aggregation: How to aggregate multiple tokens ("mean", "last", "first")

        Returns:
            Aggregated activation vector or None if span not found
        """
        full_text = self.texts[text_index]
        start_char = full_text.find(span)

        if start_char == -1:
            return None

        end_char = start_char + len(span)

        # Find tokens that overlap with this character range
        offsets = self.offset_mapping[text_index]
        token_indices = []

        for i, (start, end) in enumerate(offsets):
            # Skip special tokens (usually 0,0)
            if start == end == 0:
                continue

            # Check overlap
            if start < end_char and end > start_char:
                token_indices.append(i)

        if not token_indices:
            return None

        # Extract activations
        acts = self.activations[layer][text_index, token_indices]

        if aggregation == "mean":
            return acts.mean(dim=0)
        elif aggregation == "last":
            return acts[-1]
        elif aggregation == "first":
            return acts[0]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
