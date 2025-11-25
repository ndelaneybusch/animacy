"""
Inference engine for generating responses from language models.

This module provides a base class for inference engines and concrete
implementations for different model types (Transformers, vLLM, Anthropic).
Each has the same interface and can be used interchangeably. The factory
function `create_inference_engine` is provided to create the appropriate
inference engine for a given model.
"""

import os
from abc import ABC, abstractmethod
from typing import Any

from ..models import (
    AnthropicModelConfig,
    ModelConfig,
    TransformersModelConfig,
    VLLMModelConfig,
)
from .tasks import Task


class InferenceEngine(ABC):
    """
    Abstract base class for inference engines.

    This class defines the interface for all inference engines that generate
    responses from language models given Task objects.
    """

    def __init__(self, model_config: ModelConfig):
        """
        Initialize the inference engine with a model configuration.

        Args:
            model_config: Model configuration object specifying model parameters.
        """
        self.model_config = model_config

    @abstractmethod
    def generate_response(self, task: Task) -> str:
        """
        Generate a response for the given task.

        Args:
            task: Task object containing role name, task name, system prompt,
                  and task prompt.

        Returns:
            Generated text response from the model.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up resources used by the inference engine.

        This method should be called when the inference engine is no longer needed
        to free up memory and other resources.
        """
        pass

    def __enter__(self) -> "InferenceEngine":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()


class TransformersInferenceEngine(InferenceEngine):
    """
    Inference engine using HuggingFace Transformers library.

    This class handles model loading, tokenization, and text generation using
    the Transformers library with support for various quantization options.
    """

    def __init__(self, model_config: TransformersModelConfig):
        """
        Initialize the Transformers inference engine.

        Args:
            model_config: TransformersModel configuration object.
        """
        super().__init__(model_config)
        self.model_config: TransformersModelConfig = model_config
        self._pipeline: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer using Transformers pipeline."""
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "transformers library not found. "
                "Install with: pip install transformers torch"
            ) from e

        model_kwargs: dict[str, Any] = {
            "device_map": self.model_config.device,
            "torch_dtype": self.model_config.torch_dtype,
        }

        if self.model_config.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif self.model_config.load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        self._pipeline = pipeline(
            "text-generation",
            model=self.model_config.model_name,
            model_kwargs=model_kwargs,
        )

    def generate_response(self, task: Task) -> str:
        """
        Generate a response using Transformers pipeline.

        Args:
            task: Task object with system and task prompts.

        Returns:
            Generated text response.
        """
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call _load_model first.")

        # Format messages for chat models
        messages = []
        if task.system_prompt is not None:
            messages.append({"role": "system", "content": task.system_prompt})
        messages.append({"role": "user", "content": task.task_prompt})

        # Generate response
        outputs = self._pipeline(
            messages,
            max_new_tokens=self.model_config.max_tokens,
            temperature=self.model_config.temperature,
            return_full_text=False,
        )

        # Extract generated text
        if outputs and len(outputs) > 0:
            return outputs[0]["generated_text"]
        return ""

    def cleanup(self) -> None:
        """Free up model resources."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

        # Clear CUDA cache if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class VLLMInferenceEngine(InferenceEngine):
    """
    Inference engine using vLLM for high-throughput inference.

    This class provides optimized inference using the vLLM engine with support
    for tensor parallelism and efficient GPU memory utilization.
    """

    def __init__(self, model_config: VLLMModelConfig):
        """
        Initialize the vLLM inference engine.

        Args:
            model_config: VLLMModel configuration object.
        """
        super().__init__(model_config)
        self.model_config: VLLMModelConfig = model_config
        self._llm: Any = None
        self._sampling_params: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the model using vLLM."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vllm library not found. Install with: pip install vllm"
            ) from e

        llm_kwargs = {
            "model": self.model_config.model_name,
            "tensor_parallel_size": self.model_config.tensor_parallel_size,
            "gpu_memory_utilization": self.model_config.gpu_memory_utilization,
            "dtype": self.model_config.dtype,
            "trust_remote_code": self.model_config.trust_remote_code,
            "enforce_eager": self.model_config.enforce_eager,
        }

        # Only add max_model_len if specified
        if self.model_config.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.model_config.max_model_len

        self._llm = LLM(**llm_kwargs)

        self._sampling_params = SamplingParams(
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens,
        )

    def generate_response(self, task: Task) -> str:
        """
        Generate a response using vLLM.

        Args:
            task: Task object with system and task prompts.

        Returns:
            Generated text response.
        """
        if self._llm is None or self._sampling_params is None:
            raise RuntimeError("Model not loaded. Call _load_model first.")

        # Format the prompt combining system and task prompts
        # vLLM expects a single string prompt or can use chat templates
        try:
            # Try to use chat template if available
            tokenizer = self._llm.get_tokenizer()
            messages = []
            if task.system_prompt is not None:
                messages.append({"role": "system", "content": task.system_prompt})
            messages.append({"role": "user", "content": task.task_prompt})
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (AttributeError, Exception):
            # Fallback to simple concatenation
            if task.system_prompt is not None:
                prompt = f"{task.system_prompt}\n\n{task.task_prompt}"
            else:
                prompt = task.task_prompt

        # Generate response
        outputs = self._llm.generate([prompt], self._sampling_params)

        # Extract generated text
        if outputs and len(outputs) > 0:
            return outputs[0].outputs[0].text
        return ""

    def cleanup(self) -> None:
        """Free up model resources."""
        if self._llm is not None:
            del self._llm
            self._llm = None

        # vLLM handles its own cleanup, but we can help with CUDA cache
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class AnthropicInferenceEngine(InferenceEngine):
    """
    Inference engine using Anthropic API.

    This class handles communication with Anthropic's Claude models via their API,
    supporting both streaming and non-streaming responses.
    """

    def __init__(self, model_config: AnthropicModelConfig):
        """
        Initialize the Anthropic inference engine.

        Args:
            model_config: AnthropicModel configuration object.
        """
        super().__init__(model_config)
        self.model_config: AnthropicModelConfig = model_config
        self._client: Any = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Anthropic API client."""
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic library not found. Install with: pip install anthropic"
            ) from e

        # Use provided API key or fall back to environment variable
        api_key = self.model.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment "
                "variable or provide api_key in AnthropicModel configuration."
            )

        self._client = Anthropic(api_key=api_key)

    def generate_response(self, task: Task) -> str:
        """
        Generate a response using Anthropic API.

        Args:
            task: Task object with system and task prompts.

        Returns:
            Generated text response.
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Call _initialize_client first.")

        # Build request parameters
        request_params: dict[str, Any] = {
            "model": self.model_config.model_name,
            "max_tokens": self.model_config.max_tokens,
            "temperature": self.model_config.temperature,
            "messages": [{"role": "user", "content": task.task_prompt}],
        }

        # Only add system prompt if it's not None
        if task.system_prompt is not None:
            request_params["system"] = task.system_prompt

        # Add optional parameters if specified
        if self.model_config.top_p is not None:
            request_params["top_p"] = self.model_config.top_p
        if self.model_config.top_k is not None:
            request_params["top_k"] = self.model_config.top_k

        # Generate response
        response = self._client.messages.create(**request_params)

        # Extract text from response
        if response.content and len(response.content) > 0:
            # Anthropic returns a list of content blocks
            return response.content[0].text
        return ""

    def cleanup(self) -> None:
        """Clean up API client resources."""
        # Anthropic client doesn't require explicit cleanup
        self._client = None


def create_inference_engine(model_config: ModelConfig) -> InferenceEngine:
    """
    Factory function to create the appropriate inference engine for a given model.

    Args:
        model_config: Model configuration object (TransformersModelConfig, VLLMModelConfig, or AnthropicModelConfig).

    Returns:
        Appropriate InferenceEngine instance based on model type.

    Raises:
        ValueError: If model type is not recognized.
    """
    if isinstance(model_config, TransformersModelConfig):
        return TransformersInferenceEngine(model_config)
    elif isinstance(model_config, VLLMModelConfig):
        return VLLMInferenceEngine(model_config)
    elif isinstance(model_config, AnthropicModelConfig):
        return AnthropicInferenceEngine(model_config)
    else:
        raise ValueError(
            f"Unknown model type: {type(model_config).__name__}. "
            "Expected TransformersModelConfig, VLLMModelConfig, or AnthropicModelConfig."
        )
