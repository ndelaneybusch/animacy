from typing import Any

from .roles import get_article


def construct_chat_history(
    item: dict[str, Any], config: dict[str, Any], use_system_prompt: bool = True
) -> list[dict[str, str]]:
    """
    Construct the chat history for a given item.

    Args:
        item: Dictionary containing role_name, task_name, response.
        config: Configuration dictionary containing prompts.
        use_system_prompt: Whether to include the system prompt.

    Returns:
        List of message dictionaries for the chat template.
    """
    role_name = item["role_name"]
    task_name = item["task_name"]
    response = item["response"]

    system_prompt_template = config["SYSTEM_PROMPT"]
    system_prompt = ""

    if role_name is None:
        use_system_prompt = False
    else:
        # Handle "a" vs "an"
        # We assume the template might contain " a {role_name}" and we need to fix it
        # if the role requires "an".
        if " a {role_name}" in system_prompt_template:
            article = get_article(role_name)
            if article == "an":
                system_prompt_template = system_prompt_template.replace(
                    " a {role_name}", " an {role_name}"
                )

        system_prompt = system_prompt_template.format(role_name=role_name)

    # Handle task prompts being in a subsection
    task_prompts = config.get("TASK_PROMPTS", {})
    user_prompt = task_prompts.get(task_name, "")

    if not user_prompt:
        # Fallback or error handling if task name not found
        print(
            f"Warning: Task '{task_name}' not found in config. Using empty user prompt."
        )

    messages = []
    if use_system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend(
        [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ]
    )
    return messages
