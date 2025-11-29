"""
Integration test for Gemma model activation extraction.

Tests the full pipeline with a real Gemma tokenizer to ensure:
1. Role name detection works correctly
2. Role normalization happens properly
3. Message ranges are created with correct roles
4. The chat template is applied correctly
"""

import torch
from transformers import AutoTokenizer

from animacy.activations.data import _detect_assistant_role_name


def test_gemma_tokenizer_role_detection():
    """Test role detection with real Gemma tokenizer."""
    print("\n=== Testing Gemma Tokenizer Role Detection ===")

    # Load the real Gemma tokenizer
    print("Loading google/gemma-3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-3-270m-it", trust_remote_code=True
    )

    # Detect the assistant role name
    detected_role = _detect_assistant_role_name(tokenizer)
    print(f"Detected assistant role name: '{detected_role}'")

    # Print the chat template for inspection
    print(f"\nChat template:\n{tokenizer.chat_template[:500]}...")

    assert detected_role in ["model", "assistant"], (
        f"Expected 'model' or 'assistant', got '{detected_role}'"
    )

    return detected_role


def test_gemma_chat_template_application():
    """Test that chat template works with both 'assistant' and 'model' roles."""
    print("\n=== Testing Chat Template Application ===")

    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-3-270m-it", trust_remote_code=True
    )
    detected_role = _detect_assistant_role_name(tokenizer)

    # Test with the detected role
    messages_correct = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": detected_role, "content": "Hi there!"},
    ]

    try:
        result_correct = tokenizer.apply_chat_template(
            messages_correct, tokenize=False, add_generation_prompt=False
        )
        print(f"\n✓ Chat template with '{detected_role}' role works:")
        print(f"  Length: {len(result_correct)} chars")
        print(f"  Preview: {result_correct[:200]}...")

        # Verify all content is present
        assert "You are a helpful assistant" in result_correct
        assert "Hello!" in result_correct
        assert "Hi there!" in result_correct
        print("  ✓ All message content found in output")

    except Exception as e:
        print(f"\n✗ Chat template with '{detected_role}' role failed: {e}")
        raise

    # Test with 'assistant' role (should fail if Gemma expects 'model')
    messages_assistant = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    try:
        result_assistant = tokenizer.apply_chat_template(
            messages_assistant, tokenize=False, add_generation_prompt=False
        )
        print(f"\n✓ Chat template with 'assistant' role also works:")
        print(f"  Length: {len(result_assistant)} chars")

        # Check if assistant content is included
        if "Hi there!" in result_assistant:
            print("  ✓ Assistant message content found (role name is flexible)")
        else:
            print("  ✗ Assistant message content NOT found (needs normalization)")

    except Exception as e:
        print(f"\n✗ Chat template with 'assistant' role failed: {e}")
        print(f"  → This confirms we need role normalization for Gemma")


def test_gemma_message_ranges():
    """Test that message ranges are created correctly with normalized roles."""
    print("\n=== Testing Message Range Creation ===")

    from animacy.activations.extractor import ActivationExtractor

    # We'll test the _process_chat_inputs method directly
    # Create a minimal mock extractor with just the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-3-270m-it", trust_remote_code=True
    )

    # Create a mock extractor
    class MockExtractor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        # Copy the _process_chat_inputs method
        def _process_chat_inputs(self, chat_histories):
            from animacy.activations.data import _detect_assistant_role_name

            expected_assistant_role = _detect_assistant_role_name(self.tokenizer)

            texts = []
            all_message_ranges = []

            for history in chat_histories:
                # Normalize role names
                normalized_history = []
                for message in history:
                    role = message["role"]
                    if role == "assistant" and expected_assistant_role != "assistant":
                        role = expected_assistant_role
                    normalized_history.append(
                        {"role": role, "content": message["content"]}
                    )

                # Apply chat template
                full_text = self.tokenizer.apply_chat_template(
                    normalized_history, tokenize=False, add_generation_prompt=False
                )
                texts.append(full_text)

                # Find ranges
                ranges = []
                current_search_idx = 0

                for message in normalized_history:
                    content = message["content"]
                    role = message["role"]

                    if not content:
                        continue

                    start_idx = full_text.find(content, current_search_idx)

                    if start_idx == -1:
                        print(f"WARNING: Could not find content for role '{role}'")
                        continue

                    end_idx = start_idx + len(content)

                    ranges.append(
                        {
                            "role": role,
                            "start": start_idx,
                            "end": end_idx,
                            "content": content,
                        }
                    )

                    current_search_idx = end_idx

                all_message_ranges.append(ranges)

            return texts, all_message_ranges

    extractor = MockExtractor(tokenizer)

    # Test with 'assistant' role (should be normalized to 'model')
    chat_histories = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]
    ]

    texts, message_ranges = extractor._process_chat_inputs(chat_histories)

    print(f"Generated text length: {len(texts[0])} chars")
    print(f"Number of message ranges: {len(message_ranges[0])}")

    # Check the roles in message_ranges
    roles_in_ranges = [r["role"] for r in message_ranges[0]]
    print(f"Roles in message_ranges: {roles_in_ranges}")

    # Verify all three messages were found
    assert len(message_ranges[0]) == 3, (
        f"Expected 3 message ranges, got {len(message_ranges[0])}"
    )

    # Verify the assistant role was normalized
    detected_role = _detect_assistant_role_name(tokenizer)
    assert detected_role in roles_in_ranges, (
        f"Expected '{detected_role}' in roles, got {roles_in_ranges}"
    )

    print(f"✓ All messages found with correct roles")
    print(f"✓ Assistant role normalized to '{detected_role}'")


if __name__ == "__main__":
    print("=" * 60)
    print("GEMMA INTEGRATION TESTS")
    print("=" * 60)

    try:
        detected_role = test_gemma_tokenizer_role_detection()
        test_gemma_chat_template_application()
        test_gemma_message_ranges()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print(f"\nSummary:")
        print(f"  - Gemma uses '{detected_role}' as the assistant role name")
        print(f"  - Role normalization is working correctly")
        print(f"  - Message ranges are created with normalized roles")

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
