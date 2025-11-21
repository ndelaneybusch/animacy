"""
TokenMapper - Helper classes for mapping text to tokens and activations.
"""

import torch
from transformers import PreTrainedTokenizer


class ActivationResult:
    """
    Stores the result of an activation extraction and provides methods to query it.

    Attributes:
        activations: Dictionary mapping layer index to activation tensor (batch, seq, hidden).
        input_ids: Input token IDs (batch, seq).
        offset_mapping: Token character offsets (batch, seq, 2).
        tokenizer: Tokenizer used.
        texts: Original input texts.
        message_ranges: List of list of tuples (start_char, end_char, role) for each text.
                        Only present if input was a list of messages.
    """

    def __init__(
        self,
        activations: dict[int, torch.Tensor],
        input_ids: torch.Tensor,
        offset_mapping: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        texts: list[str],
        message_ranges: list[list[dict]] | None = None,
    ):
        """
        Initialize ActivationResult.

        Args:
            activations: Dictionary mapping layer index to activation tensor (batch, seq, hidden)
            input_ids: Input token IDs (batch, seq)
            offset_mapping: Token character offsets (batch, seq, 2)
            tokenizer: Tokenizer used
            texts: Original input texts
            message_ranges: Optional metadata about message boundaries if input was chat messages.
                            Structure: [[{"start": int, "end": int, "role": str}, ...], ...]
        """
        self.activations = activations
        self.input_ids = input_ids
        self.offset_mapping = offset_mapping
        self.tokenizer = tokenizer
        self.texts = texts
        self.message_ranges = message_ranges
        self.decoded_texts = tokenizer.batch_decode(
            input_ids, skip_special_tokens=False
        )

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

    def get_token_indices_for_char_range(
        self, text_index: int, start_char: int, end_char: int
    ) -> list[int]:
        """
        Get token indices that overlap with a specific character range.

        Uses offset_mapping for efficient lookup.

        Args:
            text_index: Index of the text in the batch
            start_char: Start character index (inclusive)
            end_char: End character index (exclusive)

        Returns:
            List of token indices
        """
        offsets = self.offset_mapping[text_index]  # (seq_len, 2)

        # Find tokens where (token_start < end_char) and (token_end > start_char)
        # Note: offset_mapping usually contains (start, end) for each token

        # We can do this with tensor operations for speed, but a loop is fine for now
        # given sequence lengths aren't massive.

        indices = []
        for i, (tok_start, tok_end) in enumerate(offsets):
            # Skip special tokens which often have (0, 0) offsets
            if tok_start == tok_end:
                continue

            if tok_start < end_char and tok_end > start_char:
                indices.append(i)

        return indices

    def get_token_indices_for_span(
        self, text_index: int, span: str, start_search_index: int = 0
    ) -> list[int]:
        """
        Get token indices for a specific text span.

        Args:
            text_index: Index of the text in the batch
            span: Text span to find
            start_search_index: Character index to start searching from

        Returns:
            List of token indices corresponding to the span
        """
        full_text = self.decoded_texts[text_index]
        start_char = full_text.find(span, start_search_index)

        if start_char == -1:
            return []

        end_char = start_char + len(span)
        return self.get_token_indices_for_char_range(text_index, start_char, end_char)

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
        token_indices = self.get_token_indices_for_span(text_index, span)

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
