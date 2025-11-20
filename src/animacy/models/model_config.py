from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class ModelConfig(BaseModel, ABC):
    """
    Abstract base class for model configurations.

    This class defines the interface for model configurations used across
    different inference backends (Transformers, vLLM, Anthropic).
    """

    model_name: str = Field(..., description="Name/identifier of the model")
    max_tokens: int = Field(
        default=1024, description="Maximum tokens to generate in response"
    )
    temperature: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Sampling temperature"
    )

    @abstractmethod
    def get_backend_type(self) -> str:
        """
        Get the backend type for this model.

        Returns:
            String identifier for the backend (e.g., 'transformers', 'vllm', 'anthropic')
        """
        pass


class TransformersModelConfig(ModelConfig):
    """
    Configuration for models using HuggingFace Transformers library.

    Attributes:
        model_name: HuggingFace model identifier (e.g., 'meta-llama/Llama-2-7b-chat-hf')
        device: Device to run model on ('cuda', 'cpu', 'auto')
        torch_dtype: Torch dtype for model weights (e.g., 'float16', 'bfloat16', 'auto')
        load_in_8bit: Whether to load model in 8-bit quantization
        load_in_4bit: Whether to load model in 4-bit quantization
    """

    device: str = Field(default="auto", description="Device for model execution")
    torch_dtype: str = Field(
        default="auto", description="Torch dtype for model weights"
    )
    load_in_8bit: bool = Field(
        default=False, description="Load model with 8-bit quantization"
    )
    load_in_4bit: bool = Field(
        default=False, description="Load model with 4-bit quantization"
    )
    trust_remote_code: bool = Field(
        default=False, description="Trust remote code when loading model"
    )

    def get_backend_type(self) -> str:
        """Return the backend type identifier."""
        return "transformers"


class VLLMModelConfig(ModelConfig):
    """
    Configuration for models using vLLM inference engine.

    Attributes:
        model_name: Model identifier compatible with vLLM
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        dtype: Data type for model weights ('auto', 'half', 'float16', 'bfloat16', 'float', 'float32')
        max_model_len: Maximum sequence length (None = use model's default)
        trust_remote_code: Whether to trust remote code when loading model
    """

    tensor_parallel_size: int = Field(
        default=1, ge=1, description="Number of GPUs for tensor parallelism"
    )
    gpu_memory_utilization: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Fraction of GPU memory to use"
    )
    dtype: str = Field(default="auto", description="Data type for model weights")
    max_model_len: int | None = Field(
        default=None,
        description="Maximum sequence length for model (None = use model's default)",
    )
    trust_remote_code: bool = Field(
        default=False, description="Trust remote code when loading model"
    )

    def get_backend_type(self) -> str:
        """Return the backend type identifier."""
        return "vllm"


class AnthropicModelConfig(ModelConfig):
    """
    Configuration for models using Anthropic API.

    Attributes:
        model_name: Anthropic model identifier (e.g., 'claude-3-5-sonnet-20241022')
        api_key: Anthropic API key (if None, will use ANTHROPIC_API_KEY environment variable)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
    """

    api_key: str | None = Field(
        default=None, description="Anthropic API key (optional, uses env var if None)"
    )
    top_p: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )
    top_k: int | None = Field(
        default=None, ge=0, description="Top-k sampling parameter"
    )

    def get_backend_type(self) -> str:
        """Return the backend type identifier."""
        return "anthropic"

    model_config = {"arbitrary_types_allowed": True}
