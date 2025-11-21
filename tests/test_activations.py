"""
Tests for the activations module.
"""

import pytest

from animacy.activations import ActivationExtractor

# Use a small model for testing
TEST_MODEL = "gpt2"


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
    assert act_0.size(2) == 768

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
    assert fox_act.shape == (768,)

    # Test getting activation for "dog"
    dog_act = result.get_span_activation("dog", layer=5)
    assert dog_act is not None
    assert dog_act.shape == (768,)

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
    assert act_0[1].shape == (act.size(1), 768)


def test_all_layers(extractor):
    """Test extracting all layers when layers=None."""
    text = "Test"
    result = extractor.extract(text, layers=None)

    num_layers = len(extractor._layers)
    assert len(result.activations) == num_layers
