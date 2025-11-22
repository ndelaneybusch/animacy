import numpy as np
from pydantic import BaseModel, Field, field_serializer

from .token_mapper import ActivationResult


class ActivationSummaries(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    avg_system_prompt: np.ndarray | None = Field(
        description="Average activation of the system prompt, excluding special tokens denoting the start and end of the system prompt"
    )
    avg_user_prompt: np.ndarray = Field(
        description="Average activation of the user prompt, excluding special tokens denoting the start and end of the user prompt"
    )
    avg_response: np.ndarray = Field(
        description="Average activation of the response, excluding special tokens denoting the start and end of the response"
    )
    avg_response_first_10_tokens: np.ndarray = Field(
        description="Average activation of the response's first 10 tokens."
    )
    at_role: np.ndarray | None = Field(
        description="Activation of the role_name in the system prompt (all tokens of the role_name)."
    )
    at_role_period: np.ndarray | None = Field(
        description="Activation of the first period following the role inside the system prompt"
    )
    at_end_system_prompt: np.ndarray | None = Field(
        description="Activation of the special token denoting the end of the system prompt"
    )
    at_end_user_prompt: np.ndarray | None = Field(
        description="Activation of the special token denoting the end of the user prompt"
    )
    at_start_agent_response: np.ndarray | None = Field(
        description="Activation of the special token denoting the start of the agent response"
    )

    @field_serializer(
        "avg_system_prompt",
        "avg_user_prompt",
        "avg_response",
        "avg_response_first_10_tokens",
        "at_role",
        "at_role_period",
        "at_end_system_prompt",
        "at_end_user_prompt",
        "at_start_agent_response",
    )
    def serialize_numpy(self, value: np.ndarray | None, _info) -> list[float] | None:
        if value is None:
            return None
        return value.tolist()


def extract_activation_summaries(
    activation_result: ActivationResult,
    role_name: str | None,
    layer: int,
    text_index: int = 0,
) -> ActivationSummaries:
    """
    Extract activation summaries for a specific role and task.

    Args:
        activation_result: The result containing activations and tokenizer.
        role_name: The name of the role to extract activations for.
        layer: The layer to extract activations from.
        text_index: The index of the text in the batch.

    Returns:
        ActivationSummaries object populated with extracted activations.
    """
    if not activation_result.message_ranges:
        raise ValueError(
            "ActivationResult must contain message_ranges for extraction. "
            "Ensure you passed a list of chat messages to ActivationExtractor.extract()."
        )

    ranges = activation_result.message_ranges[text_index]
    input_ids = activation_result.input_ids[text_index]
    tokenizer = activation_result.tokenizer
    full_text = activation_result.decoded_texts[text_index]

    # Helper to find message by role
    def find_message_range(role):
        for r in ranges:
            if r["role"] == role:
                return r
        return None

    sys_range = find_message_range("system")
    user_range = find_message_range("user")
    asst_range = find_message_range("assistant")

    # Helper to calculate mean activation for a char range
    def get_mean_activation_for_range(start, end):
        indices = activation_result.get_token_indices_for_char_range(
            text_index, start, end
        )
        if not indices:
            return None

        # Filter special tokens
        valid_indices = [
            idx for idx in indices if input_ids[idx] not in tokenizer.all_special_ids
        ]

        if not valid_indices:
            return None

        acts = activation_result.activations[layer][text_index, valid_indices]
        return acts.mean(dim=0).cpu().numpy()

    # 1. Calculate Averages
    avg_system_prompt = (
        get_mean_activation_for_range(sys_range["start"], sys_range["end"])
        if sys_range
        else None
    )
    avg_user_prompt = (
        get_mean_activation_for_range(user_range["start"], user_range["end"])
        if user_range
        else None
    )
    avg_response = (
        get_mean_activation_for_range(asst_range["start"], asst_range["end"])
        if asst_range
        else None
    )

    # First 10 tokens of response
    avg_response_first_10 = None
    if asst_range:
        indices = activation_result.get_token_indices_for_char_range(
            text_index, asst_range["start"], asst_range["end"]
        )
        valid_indices = [
            idx for idx in indices if input_ids[idx] not in tokenizer.all_special_ids
        ]
        if valid_indices:
            first_10 = valid_indices[:10]
            acts = activation_result.activations[layer][text_index, first_10]
            avg_response_first_10 = acts.mean(dim=0).cpu().numpy()

    # 2. Specific Role Activations
    at_role = None
    at_role_period = None

    if sys_range and role_name:
        # Find role name within system prompt
        sys_text = full_text[sys_range["start"] : sys_range["end"]]
        role_start_local = sys_text.find(role_name)

        if role_start_local != -1:
            role_start_global = sys_range["start"] + role_start_local
            role_end_global = role_start_global + len(role_name)

            at_role = get_mean_activation_for_range(role_start_global, role_end_global)

            # Find period after role
            period_idx_local = sys_text.find(".", role_start_local + len(role_name))
            if period_idx_local != -1:
                period_start_global = sys_range["start"] + period_idx_local
                period_end_global = period_start_global + 1

                # Get token for period
                period_indices = activation_result.get_token_indices_for_char_range(
                    text_index, period_start_global, period_end_global
                )
                if period_indices:
                    at_role_period = (
                        activation_result.activations[layer][
                            text_index, period_indices[0]
                        ]
                        .cpu()
                        .numpy()
                    )

    # 3. Special Tokens (Start/End)
    # We look for tokens immediately adjacent to the content ranges
    at_end_system = None
    at_end_user = None
    at_start_response = None

    def get_token_at_index(idx):
        if 0 <= idx < len(input_ids):
            return activation_result.activations[layer][text_index, idx].cpu().numpy()
        return None

    if sys_range:
        # The token *after* the last content token is likely the end delimiter
        # But we need to be careful. get_token_indices_for_char_range returns indices covering the content.
        # The token *after* the last one in that list might be the delimiter.
        indices = activation_result.get_token_indices_for_char_range(
            text_index, sys_range["start"], sys_range["end"]
        )
        if indices:
            last_content_token = indices[-1]
            at_end_system = get_token_at_index(last_content_token + 1)

    if user_range:
        indices = activation_result.get_token_indices_for_char_range(
            text_index, user_range["start"], user_range["end"]
        )
        if indices:
            last_content_token = indices[-1]
            at_end_user = get_token_at_index(last_content_token + 1)

    if asst_range:
        # The token *before* the first content token is likely the start delimiter
        indices = activation_result.get_token_indices_for_char_range(
            text_index, asst_range["start"], asst_range["end"]
        )
        if indices:
            first_content_token = indices[0]
            at_start_response = get_token_at_index(first_content_token - 1)

    return ActivationSummaries(
        avg_system_prompt=avg_system_prompt,
        avg_user_prompt=avg_user_prompt,
        avg_response=avg_response,
        avg_response_first_10_tokens=avg_response_first_10,
        at_role=at_role,
        at_role_period=at_role_period,
        at_end_system_prompt=at_end_system,
        at_end_user_prompt=at_end_user,
        at_start_agent_response=at_start_response,
    )
