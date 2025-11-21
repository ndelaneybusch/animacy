import torch
from transformers import AutoTokenizer

from animacy.activations.data import extract_activation_summaries
from animacy.activations.token_mapper import ActivationResult


def test_extraction():
    # 1. Setup Mock Data
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for testing tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Could not load tokenizer for {model_name}: {e}")
        return

    role_name = "biologist"
    system_prompt = f"You are a {role_name}."
    user_prompt = "What is life?"
    response = "Life is complex."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response},
    ]

    # Create input_ids using the real chat template
    # We use tokenize=False first to get the full text for range calculation
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Now tokenize
    encodings = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = encodings["input_ids"]
    offset_mapping = encodings["offset_mapping"]

    # Calculate message ranges manually for the test
    # This mimics what ActivationExtractor._process_chat_inputs does
    message_ranges = []
    current_idx = 0
    for msg in messages:
        content = msg["content"]
        start = full_text.find(content, current_idx)
        end = start + len(content)
        message_ranges.append(
            {"role": msg["role"], "start": start, "end": end, "content": content}
        )
        current_idx = end

    # Create dummy activations
    # Shape: (1, seq_len, hidden_size)
    seq_len = input_ids.shape[1]
    hidden_size = 16
    layer = 5
    activations = {layer: torch.randn(1, seq_len, hidden_size)}

    texts = [full_text]

    result = ActivationResult(
        activations=activations,
        input_ids=input_ids,
        offset_mapping=offset_mapping,
        tokenizer=tokenizer,
        texts=texts,
        message_ranges=[message_ranges],  # List of lists
    )

    print(f"Full Text: {texts[0]}")

    # 2. Run Extraction
    summary = extract_activation_summaries(
        activation_result=result, role_name=role_name, layer=layer, text_index=0
    )

    # 3. Verify Results
    print("\nExtraction Results:")
    print(
        f"avg_system_prompt shape: {summary.avg_system_prompt.shape if summary.avg_system_prompt is not None else 'None'}"
    )
    print(
        f"avg_user_prompt shape: {summary.avg_user_prompt.shape if summary.avg_user_prompt is not None else 'None'}"
    )
    print(
        f"avg_response shape: {summary.avg_response.shape if summary.avg_response is not None else 'None'}"
    )
    print(
        f"at_role shape: {summary.at_role.shape if summary.at_role is not None else 'None'}"
    )
    print(
        f"at_role_period shape: {summary.at_role_period.shape if summary.at_role_period is not None else 'None'}"
    )

    # Basic Assertions
    assert summary.avg_system_prompt is not None
    assert summary.avg_user_prompt is not None
    assert summary.avg_response is not None
    assert summary.at_role is not None
    assert (
        summary.at_role_period is not None
    )  # Should find the period in "You are a biologist."

    print("\nVerification Successful!")


def test_token_alignment():
    print("\nRunning Token Alignment Test...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Could not load tokenizer: {e}")
        return

    # Construct a predictable sequence
    # "A B C" -> tokens for A, B, C
    # We want to verify that if we ask for "B", we get the activation for B's token.

    # Using a simple sentence where tokens are likely clear
    # "A B C D"
    # Tokens (likely): "A", "B", "C", "D"

    text = "A B C D"
    encodings = tokenizer(
        text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False
    )
    input_ids = encodings["input_ids"]
    offset_mapping = encodings["offset_mapping"]

    seq_len = input_ids.shape[1]
    hidden_size = 4
    layer = 0

    # Create activations where the value equals the token index
    # This makes it easy to verify we got the right token
    # shape: (1, seq_len, hidden_size)
    # act[0, i, :] = i
    activations_tensor = torch.zeros(1, seq_len, hidden_size)
    for i in range(seq_len):
        activations_tensor[0, i, :] = i

    activations = {layer: activations_tensor}

    result = ActivationResult(
        activations=activations,
        input_ids=input_ids,
        offset_mapping=offset_mapping,
        tokenizer=tokenizer,
        texts=[text],
    )

    # Test 1: Get tokens for "B"
    # "A B C D"
    #   ^
    indices = result.get_token_indices_for_span(0, "B")
    print(f"Indices for 'B': {indices}")

    # Verify the tokens actually decode to something containing "quick"
    decoded_tokens = [tokenizer.decode([input_ids[0][i]]) for i in indices]
    print(f"Decoded tokens for 'quick': {decoded_tokens}")

    # Verify activations
    # We expect the activation values to match the indices
    span_act = result.get_span_activation("B", layer, text_index=0, aggregation="first")

    print(f"Activation for 'B' (first token): {span_act}")

    # The value should be the index of the first token in 'indices'
    expected_val = indices[0]
    assert torch.allclose(span_act, torch.tensor(float(expected_val))), (
        f"Expected activation {expected_val}, got {span_act}"
    )

    # Test 2: Get tokens for "C"
    indices_c = result.get_token_indices_for_span(0, "C")
    print(f"Indices for 'C': {indices_c}")

    # Verify overlap logic
    # If we ask for a range that partially overlaps a token, does it include it?
    # Our logic: if tok_start < end_char and tok_end > start_char

    # "B" is indices[1]
    # "B" -> chars 1-2 (approx)
    # Ask for chars 1-2 ("h")
    # Should return index 1
    indices_c = result.get_token_indices_for_span(0, "C")
    print(f"Indices for 'C': {indices_c}")
    assert 2 in indices_c

    print("Token Alignment Test Successful!")


def test_extraction_no_system_prompt():
    """Test extraction when there's no system prompt."""
    print("\nRunning No System Prompt Test...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Could not load tokenizer for {model_name}: {e}")
        return

    role_name = "biologist"
    user_prompt = "What is life?"
    response = "Life is complex."

    # No system message
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response},
    ]

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    encodings = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = encodings["input_ids"]
    offset_mapping = encodings["offset_mapping"]

    # Calculate message ranges
    message_ranges = []
    current_idx = 0
    for msg in messages:
        content = msg["content"]
        start = full_text.find(content, current_idx)
        end = start + len(content)
        message_ranges.append(
            {"role": msg["role"], "start": start, "end": end, "content": content}
        )
        current_idx = end

    # Create dummy activations
    seq_len = input_ids.shape[1]
    hidden_size = 16
    layer = 5
    activations = {layer: torch.randn(1, seq_len, hidden_size)}

    texts = [full_text]

    result = ActivationResult(
        activations=activations,
        input_ids=input_ids,
        offset_mapping=offset_mapping,
        tokenizer=tokenizer,
        texts=texts,
        message_ranges=[message_ranges],
    )

    # Run Extraction
    summary = extract_activation_summaries(
        activation_result=result, role_name=role_name, layer=layer, text_index=0
    )

    # Verify Results
    print("\nExtraction Results (No System Prompt):")
    print(f"avg_system_prompt: {summary.avg_system_prompt}")
    print(f"avg_user_prompt: {summary.avg_user_prompt is not None}")
    print(f"avg_response: {summary.avg_response is not None}")
    print(f"at_role: {summary.at_role}")
    print(f"at_role_period: {summary.at_role_period}")

    # System-related fields should be None
    assert summary.avg_system_prompt is None
    assert summary.at_role is None
    assert summary.at_role_period is None
    assert summary.at_end_system_prompt is None

    # User and response should still work
    assert summary.avg_user_prompt is not None
    assert summary.avg_response is not None

    print("No System Prompt Test Successful!")


def test_extraction_role_not_found():
    """Test extraction when role_name doesn't appear in system prompt."""
    print("\nRunning Role Not Found Test...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Could not load tokenizer for {model_name}: {e}")
        return

    # System prompt says "teacher" but we're looking for "biologist"
    role_name = "biologist"
    system_prompt = "You are a helpful teacher."
    user_prompt = "What is life?"
    response = "Life is complex."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response},
    ]

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    encodings = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = encodings["input_ids"]
    offset_mapping = encodings["offset_mapping"]

    # Calculate message ranges
    message_ranges = []
    current_idx = 0
    for msg in messages:
        content = msg["content"]
        start = full_text.find(content, current_idx)
        end = start + len(content)
        message_ranges.append(
            {"role": msg["role"], "start": start, "end": end, "content": content}
        )
        current_idx = end

    # Create dummy activations
    seq_len = input_ids.shape[1]
    hidden_size = 16
    layer = 5
    activations = {layer: torch.randn(1, seq_len, hidden_size)}

    texts = [full_text]

    result = ActivationResult(
        activations=activations,
        input_ids=input_ids,
        offset_mapping=offset_mapping,
        tokenizer=tokenizer,
        texts=texts,
        message_ranges=[message_ranges],
    )

    # Run Extraction
    summary = extract_activation_summaries(
        activation_result=result, role_name=role_name, layer=layer, text_index=0
    )

    # Verify Results
    print("\nExtraction Results (Role Not Found):")
    print(f"avg_system_prompt: {summary.avg_system_prompt is not None}")
    print(f"at_role: {summary.at_role}")
    print(f"at_role_period: {summary.at_role_period}")

    # System prompt should still be extracted
    assert summary.avg_system_prompt is not None

    # Role-specific activations should be None
    assert summary.at_role is None
    assert summary.at_role_period is None

    # Other fields should still work
    assert summary.avg_user_prompt is not None
    assert summary.avg_response is not None

    print("Role Not Found Test Successful!")
