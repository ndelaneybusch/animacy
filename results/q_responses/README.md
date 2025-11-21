# Q Responses Experiment

This directory contains scripts for running animacy experiments that generate question-based responses from language models.

## Overview

The `run_experiment.py` script loads a dataset of words, creates role-based prompts, and collects model responses. It supports two inference backends:

- **Transformers**: HuggingFace Transformers library (supports quantization, flexible device placement)
- **vLLM**: High-performance inference engine (supports tensor parallelism, optimized GPU utilization)

## Usage

### Basic Usage (Transformers)

Run with default Transformers backend:

```bash
python scripts/run_experiment.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --num_samples 5
```

Models of interest:
meta-llama/Llama-3.3-70B-Instruct
google/gemma-3-27b-it
Qwen/Qwen3-30B-A3B-Instruct-2507

### Transformers with Quantization

Use 4-bit quantization to reduce memory usage:

```bash
python scripts/run_experiment.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --load_in_4bit \
  --num_samples 5
```

Use 8-bit quantization:

```bash
python scripts/run_experiment.py \
  --model_name meta-llama/Llama-2-13b-chat-hf \
  --load_in_8bit \
  --torch_dtype float16 \
  --num_samples 5
```

### vLLM Backend

Run with vLLM for faster inference:

```bash
python scripts/run_experiment.py \
  --backend vllm \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --num_samples 5 \
  --max_model_len 50000
```

### vLLM with Tensor Parallelism

Use multiple GPUs with tensor parallelism:

```bash
python scripts/run_experiment.py \
  --backend vllm \
  --model_name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --temperature 0.7 \
  --max_tokens 1024 \
  --gpu_memory_utilization 0.9 \
  --dtype auto \
  --trust_remote_code \
  --num_samples 5 \
  --max_model_len 50000
```

### Custom Data and Output Paths

Specify custom input and output paths:

```bash
python scripts/run_experiment.py \
  --backend vllm \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --csv_path data/my_words.csv \
  --output_dir results/my_experiment \
  --num_samples 10
```

## Arguments

### Backend Selection

- `--backend`: Choose inference backend (`transformers` or `vllm`, default: `transformers`)

### Common Arguments

- `--model_name`: HuggingFace model identifier (required)
- `--temperature`: Sampling temperature (default: 1.0)
- `--max_tokens`: Maximum tokens to generate (default: 1024)
- `--num_samples`: Number of response samples per task (default: 5)
- `--csv_path`: Path to input CSV (default: `data/selected_words.csv`)
- `--output_dir`: Output directory for results (default: `results/q_responses/data`)
- `--trust_remote_code`: Trust remote code when loading model

### Transformers-Specific Arguments

- `--device`: Device to run on (`cuda`, `cpu`, `auto`, default: `auto`)
- `--torch_dtype`: Torch dtype (`float16`, `bfloat16`, `auto`, default: `auto`)
- `--load_in_8bit`: Enable 8-bit quantization
- `--load_in_4bit`: Enable 4-bit quantization

### vLLM-Specific Arguments

- `--tensor_parallel_size`: Number of GPUs for tensor parallelism (default: 1)
- `--gpu_memory_utilization`: Fraction of GPU memory to use (0.0-1.0, default: 0.9)
- `--dtype`: Data type for model weights (`auto`, `half`, `float16`, `bfloat16`, `float`, `float32`, default: `auto`)

## Output Format

Results are saved as JSON files, one per role, in the output directory. Each file contains an array of response objects:

```json
[
  {
    "role_name": "role_identifier",
    "task_name": "task_identifier",
    "sample_idx": 1,
    "response": "generated response text..."
  }
]
```

## Performance Comparison

**Transformers**:
- Pros: Flexible quantization options, wider model support, easier debugging
- Cons: Slower inference, higher memory usage per token
- Best for: Single GPU, smaller models, development/testing

**vLLM**:
- Pros: Significantly faster inference, optimized memory usage, tensor parallelism
- Cons: Less flexible quantization, requires more setup
- Best for: Production workloads, large models, multi-GPU setups
