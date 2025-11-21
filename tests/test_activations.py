"""
Tests for the activations module.
"""

import pytest
import torch

from animacy.activations import ActivationExtractor

# Use a small model for testing
TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def extractor():
    """Fixture to load the model once for all tests."""
    return ActivationExtractor(TEST_MODEL, device="cpu")


def test_initialization(extractor):
    """Test that the extractor initializes correctly."""
    assert extractor.model is not None
    assert extractor.tokenizer is not None
    assert len(extractor._layers) > 0


def test_extract_activations_shape(extractor):
    """Test extracting activations returns correct shapes."""
    text = "Hello, world!"
    layers = [0, 2]

    result = extractor.extract(text, layers=layers)

    assert len(result.activations) == 2
    assert 0 in result.activations
    assert 2 in result.activations

    # Check tensor shape: (batch=1, seq_len, hidden_size)
    # GPT2 hidden size is 768
    act_0 = result.activations[0]
    assert act_0.dim() == 3
    assert act_0.size(0) == 1
    assert act_0.size(2) == 896

    # Check input_ids shape
    assert result.input_ids.dim() == 2
    assert result.input_ids.size(0) == 1
    assert result.input_ids.size(1) == act_0.size(1)


def test_token_mapping(extractor):
    """Test mapping text spans to activations."""
    text = "The quick brown fox jumps over the lazy dog."
    # GPT2 tokenization: ['The', 'Ġquick', 'Ġbrown', 'Ġfox', 'Ġjumps', 'Ġover', 'Ġthe', 'Ġlazy', 'Ġdog', '.']

    result = extractor.extract(text, layers=[5])

    # Test getting activation for "fox"
    fox_act = result.get_span_activation("fox", layer=5)
    assert fox_act is not None
    assert fox_act.shape == (896,)

    # Test getting activation for "dog"
    dog_act = result.get_span_activation("dog", layer=5)
    assert dog_act is not None
    assert dog_act.shape == (896,)

    # Test non-existent span
    none_act = result.get_span_activation("unicorn", layer=5)
    assert none_act is None


def test_batch_extraction(extractor):
    """Test extracting from a batch of texts."""
    texts = ["Hello world", "Another longer sentence for testing padding"]
    layers = [1]

    result = extractor.extract(texts, layers=layers)

    act = result.activations[1]
    assert act.size(0) == 2  # Batch size 2

    # Check that padding was handled (sequences should be padded to max length in batch)
    # "Hello world" is shorter, so it should be padded
    assert act.size(1) >= 2

    # Check getting activations for specific text index
    act_0 = result.get_activations_for_text(0)
    assert 1 in act_0
    assert act_0[1].shape == (act.size(1), 896)


def test_all_layers(extractor):
    """Test extracting all layers when layers=None."""
    text = "Test"
    result = extractor.extract(text, layers=None)

    num_layers = len(extractor._layers)
    assert len(result.activations) == num_layers


def test_chat_message_extraction(extractor):
    """Test extracting activations from chat messages (primary use case)."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = extractor.extract([messages], layers=[5])

    # Verify message_ranges populated
    assert result.message_ranges is not None
    assert len(result.message_ranges) == 1
    assert len(result.message_ranges[0]) == 3

    # Verify each message range has correct structure
    ranges = result.message_ranges[0]
    assert ranges[0]["role"] == "system"
    assert ranges[1]["role"] == "user"
    assert ranges[2]["role"] == "assistant"

    # Verify all ranges have required fields
    for r in ranges:
        assert "start" in r
        assert "end" in r
        assert "content" in r
        assert r["end"] > r["start"]

    # Verify activations extracted
    assert 5 in result.activations
    assert result.activations[5].shape[0] == 1  # Batch size 1


def test_process_chat_inputs(extractor):
    """Test _process_chat_inputs method directly."""
    chat_histories = [
        [
            {"role": "system", "content": "You are a biologist."},
            {"role": "user", "content": "What is DNA?"},
            {"role": "assistant", "content": "DNA is genetic material."},
        ]
    ]

    texts, message_ranges = extractor._process_chat_inputs(chat_histories)

    # Verify one text produced
    assert len(texts) == 1
    assert len(message_ranges) == 1

    # Verify all message content appears in the text
    full_text = texts[0]
    for msg in chat_histories[0]:
        assert msg["content"] in full_text

    # Verify ranges are in order
    ranges = message_ranges[0]
    assert len(ranges) == 3
    for i in range(len(ranges) - 1):
        assert ranges[i]["end"] <= ranges[i + 1]["start"]


def test_get_token_indices_for_char_range(extractor):
    """Test get_token_indices_for_char_range method."""
    text = "The quick brown fox jumps"
    result = extractor.extract(text, layers=[0])

    # Test getting tokens for "quick" (chars 4-9)
    full_text = result.decoded_texts[0]
    start_idx = full_text.find("quick")
    end_idx = start_idx + len("quick")

    indices = result.get_token_indices_for_char_range(0, start_idx, end_idx)

    assert len(indices) > 0

    # Verify the tokens decode to something containing "quick"
    decoded = extractor.tokenizer.decode(result.input_ids[0, indices])
    assert "quick" in decoded.lower()

    # Test overlapping range
    # Get a range that spans multiple words
    start_idx = full_text.find("brown")
    end_idx = full_text.find("jumps") + len("jumps")
    indices = result.get_token_indices_for_char_range(0, start_idx, end_idx)

    assert len(indices) >= 3  # Should cover multiple tokens


def test_get_token_activation(extractor):
    """Test direct token activation access."""
    text = "Hello world"
    result = extractor.extract(text, layers=[0, 5])

    # Get activation for first token in layer 0
    act_0 = result.get_token_activation(0, layer=0, text_index=0)
    assert act_0.shape == (896,)  # GPT2 hidden size

    # Get activation for first token in layer 5
    act_5 = result.get_token_activation(0, layer=5, text_index=0)
    assert act_5.shape == (896,)

    # Activations from different layers should differ
    assert not torch.equal(act_0, act_5)


def test_aggregation_methods(extractor):
    """Test all aggregation methods for span activations."""
    text = "Hello world test"
    result = extractor.extract(text, layers=[5])

    # Test all three aggregation methods
    mean_act = result.get_span_activation("Hello", layer=5, aggregation="mean")
    first_act = result.get_span_activation("Hello", layer=5, aggregation="first")
    last_act = result.get_span_activation("Hello", layer=5, aggregation="last")

    assert mean_act is not None
    assert first_act is not None
    assert last_act is not None

    # All should have correct shape
    assert mean_act.shape == (896,)
    assert first_act.shape == (896,)
    assert last_act.shape == (896,)

    # For a multi-token span, mean should differ from first/last
    # Note: "Hello" might be a single token, so let's test with a longer span
    mean_long = result.get_span_activation("world test", layer=5, aggregation="mean")
    first_long = result.get_span_activation("world test", layer=5, aggregation="first")
    last_long = result.get_span_activation("world test", layer=5, aggregation="last")

    # If the span has multiple tokens, first and last should differ
    if mean_long is not None:
        assert not torch.equal(first_long, last_long)
