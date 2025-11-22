from transformers import AutoModelForCausalLM, AutoTokenizer

from animacy.analysis import LogitExtractor
from animacy.prompts import construct_chat_history


def test_default_role_logits():
    # Mock model and tokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for testing
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    extractor = LogitExtractor(model, tokenizer)

    # Test case: role_name is None
    role_name = None
    task_name = "meaning_of_life"  # Needs to be a valid key in TASK_PROMPTS if accessed directly, but here we mock or ensure it exists
    # We need to ensure TASK_PROMPTS has this key or mock it.
    # Since we import TASK_PROMPTS in logits.py, we can't easily mock it without patching.
    # Let's assume "meaning_of_life" is in TASK_PROMPTS based on previous context or add a fallback.
    # Actually, let's just check if it runs without error and returns reasonable structure.

    # We need to make sure TASK_PROMPTS has the key.
    from animacy.prompts.tasks import TASK_PROMPTS

    if "meaning_of_life" not in TASK_PROMPTS:
        TASK_PROMPTS["meaning_of_life"] = "What is the meaning of life?"

    sample_idx = 0
    response_text = "42"

    logits = extractor.extract_logits(
        role_name=role_name,
        task_name=task_name,
        sample_idx=sample_idx,
        response_text=response_text,
        use_system_prompt=True,  # Should be ignored
    )

    assert logits.role_name is None
    assert logits.role_logits is None
    assert logits.role_period_logit is None


def test_construct_chat_history_default_role():
    config = {
        "SYSTEM_PROMPT": "You are a {role_name}.",
        "TASK_PROMPTS": {"test_task": "Do something."},
    }
    item = {"role_name": None, "task_name": "test_task", "response": "OK."}

    messages = construct_chat_history(item, config, use_system_prompt=True)

    # Should not have system message
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
