import torch
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from animacy.prompts.roles import BASE_STEM, get_article
from animacy.prompts.tasks import TASK_PROMPTS


class ResponseLogits(BaseModel):
    """
    Container for tokenwise log-probabilities of interest from a model response.
    """

    role_name: str | None
    task_name: str
    sample_idx: int
    average_log_probs: float
    role_log_probs: float | None = None
    role_period_log_prob: float | None = None
    first_100_response_log_probs: list[float]
    first_100_response_text_len: int


class LogitExtractor:
    """
    Extracts specific logits from model outputs for animacy experiments.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.period_token_ids = self._find_period_tokens()

    def _find_period_tokens(self) -> set[int]:
        """
        Find all token IDs in the vocabulary that contain a period.
        This allows direct token ID matching without decoding.
        """
        period_tokens = set()
        vocab = self.tokenizer.get_vocab()
        for token_str, token_id in vocab.items():
            if "." in token_str:
                period_tokens.add(token_id)
        return period_tokens

    @staticmethod
    def _find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
        """
        Find the starting index of a subsequence within a sequence.
        Returns -1 if not found.
        """
        n = len(subsequence)
        for i in range(len(sequence) - n + 1):
            if sequence[i : i + n] == subsequence:
                return i
        return -1

    def extract_logits_batch(
        self,
        samples: list[dict],
        use_system_prompt: bool = True,
        steering_manager=None,
    ) -> list[ResponseLogits]:
        """
        Calculate log-probabilities for a batch of samples.

        Args:
            samples: List of dictionaries containing sample data. Each dict must have:
                     - role_name (str | None)
                     - task_name (str)
                     - sample_idx (int)
                     - response (str)
                     - system_prompt (str, optional)
                     - task_prompt (str, optional)
            use_system_prompt: Whether to include the role-assigning system prompt.

        Returns:
            List of ResponseLogits objects.
        """
        if not samples:
            return []

        # 1. Prepare inputs
        batch_prompts = []
        batch_metadata = []

        for sample in samples:
            role_name = sample.get("role_name")
            task_name = sample["task_name"]
            response_text = sample["response"]
            custom_system_prompt = sample.get("system_prompt")
            custom_task_prompt = sample.get("task_prompt")

            # Reconstruct prompts
            if role_name is None:
                use_sys = False
                system_prompt = ""
            else:
                use_sys = use_system_prompt
                if custom_system_prompt is not None:
                    system_prompt = custom_system_prompt
                else:
                    article = get_article(role_name)
                    system_prompt = f"{BASE_STEM} {article} {role_name}."

            if custom_task_prompt is not None:
                task_prompt = custom_task_prompt
            else:
                task_prompt = TASK_PROMPTS[task_name]

            # Construct messages
            messages = []
            if use_sys:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": task_prompt})
            messages.append({"role": "assistant", "content": response_text})

            batch_prompts.append(messages)
            batch_metadata.append(
                {
                    "role_name": role_name,
                    "task_name": task_name,
                    "sample_idx": sample["sample_idx"],
                    "system_prompt": system_prompt,
                    "task_prompt": task_prompt,
                    "use_system_prompt": use_sys,
                }
            )

        # 2. Tokenize batch
        # Use right padding for scoring to ensure position IDs align naturally
        # with serial execution. We enforce this regardless of tokenizer settings.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Save original padding side and enforce right padding
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"

        try:
            encodings = self.tokenizer.apply_chat_template(
                batch_prompts,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                add_generation_prompt=False,
            )
        finally:
            # Always restore original padding side
            self.tokenizer.padding_side = original_padding_side

        input_ids = encodings["input_ids"].to(self.model.device)
        attention_mask = encodings["attention_mask"].to(self.model.device)

        # 3. Run model
        with torch.no_grad():
            # If steering manager is provided, set the attention mask
            if steering_manager is not None:
                steering_manager._current_attention_mask = attention_mask

            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Clear the attention mask after use
            if steering_manager is not None:
                steering_manager._current_attention_mask = None

        # 4. Calculate log-probs
        log_probs = torch.log_softmax(logits, dim=-1)

        # Shift
        shift_log_probs = log_probs[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Gather target log probs
        target_log_probs = shift_log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(
            2
        )

        # 5. Process each sample
        results = []
        for i, meta in enumerate(batch_metadata):
            # With right padding, valid tokens start at 0 and end where mask becomes 0
            valid_len = attention_mask[i].sum().item()

            # Extract the valid sequence for this sample
            sample_input_ids = input_ids[i, :valid_len]
            sample_target_log_probs = target_log_probs[
                i, : valid_len - 1
            ]  # target_log_probs is 1 shorter

            # Now we need to find the prompt boundaries within this valid sequence.
            system_prompt = meta["system_prompt"]
            task_prompt = meta["task_prompt"]
            use_sys = meta["use_system_prompt"]
            role_name = meta["role_name"]
            task_name = meta["task_name"]

            # 1. System Prompt Boundary
            system_end_idx = -1
            user_end_idx = -1

            if use_sys:
                ids_sys = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt}],
                    add_generation_prompt=False,
                )
                system_end_idx = len(ids_sys) - 1

                ids_sys_user = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task_prompt},
                    ],
                    add_generation_prompt=True,
                )
                user_end_idx = len(ids_sys_user) - 1
            else:
                ids_user = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": task_prompt}],
                    add_generation_prompt=True,
                )
                user_end_idx = len(ids_user) - 1

            response_start_idx = user_end_idx + 1

            # Check bounds
            if response_start_idx >= len(sample_input_ids):
                # Should not happen if response is not empty
                avg_log_prob = 0.0
                first_100 = []
                first_100_text_len = 0
            else:
                # Response log probs
                # sample_target_log_probs[k] corresponds to sample_input_ids[k+1]
                # So we want sample_target_log_probs[response_start_idx-1:]
                response_log_probs = sample_target_log_probs[response_start_idx - 1 :]
                avg_log_prob = response_log_probs.mean().item()

                # First 100
                first_100 = response_log_probs[:100].tolist()

                first_100_token_ids = sample_input_ids[
                    response_start_idx : response_start_idx + 100
                ]
                first_100_text = self.tokenizer.decode(
                    first_100_token_ids, skip_special_tokens=True
                )
                first_100_text_len = len(first_100_text)

            # Role metrics
            role_period_log_prob = None
            role_log_probs_val = None

            if use_sys and role_name:
                # Find role in system prompt
                sys_text_ids = self.tokenizer(system_prompt, add_special_tokens=False)[
                    "input_ids"
                ]
                role_ids = self.tokenizer(role_name, add_special_tokens=False)[
                    "input_ids"
                ]

                full_ids_list = sample_input_ids.tolist()

                # Search within system prompt range
                sys_search_end = min(system_end_idx + 1, len(full_ids_list))
                sys_start_idx = self._find_subsequence(
                    full_ids_list[:sys_search_end], sys_text_ids
                )

                if sys_start_idx != -1:
                    role_in_sys_idx = self._find_subsequence(sys_text_ids, role_ids)

                    if role_in_sys_idx != -1:
                        role_start_idx = sys_start_idx + role_in_sys_idx
                        role_end_idx = role_start_idx + len(role_ids)

                        # Log probs
                        role_log_probs_tokens = sample_target_log_probs[
                            role_start_idx - 1 : role_end_idx - 1
                        ]
                        role_log_probs_val = role_log_probs_tokens.sum().item()

                        # Period
                        search_start = role_end_idx
                        for k in range(search_start, sys_start_idx + len(sys_text_ids)):
                            if (
                                k < len(sample_input_ids)
                                and sample_input_ids[k].item() in self.period_token_ids
                            ):
                                role_period_log_prob = sample_target_log_probs[
                                    k - 1
                                ].item()
                                break
                else:
                    # Fallback
                    for k in range(system_end_idx, -1, -1):
                        if (
                            k < len(sample_input_ids)
                            and sample_input_ids[k].item() in self.period_token_ids
                        ):
                            if role_period_log_prob is None:
                                role_period_log_prob = sample_target_log_probs[
                                    k - 1
                                ].item()
                            break

            results.append(
                ResponseLogits(
                    role_name=role_name,
                    task_name=task_name,
                    sample_idx=meta["sample_idx"],
                    average_log_probs=avg_log_prob,
                    role_log_probs=role_log_probs_val,
                    role_period_log_prob=role_period_log_prob,
                    first_100_response_log_probs=first_100,
                    first_100_response_text_len=first_100_text_len,
                )
            )

        return results
