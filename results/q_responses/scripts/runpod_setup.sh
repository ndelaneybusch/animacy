#!/bin/bash
set -e  # Exit on error

echo "=== RunPod Setup and Experiment Runner ==="

# ============================================================================
# CONFIGURATION - Set your tokens here
# ============================================================================
export GITHUB_TOKEN="YOUR_GITHUB_PAT_HERE"

# Use RunPod secret for HuggingFace token
export HF_TOKEN="${RUNPOD_SECRET_HF_TOKEN}"

# ============================================================================
# Install Dependencies
# ============================================================================
echo "Installing system dependencies..."
apt-get update && apt-get install -y git curl
# ============================================================================
# Clone and Install Project
# ============================================================================
echo "Cloning repository..."

git clone https://github.com/ndelaneybusch/animacy.git
cd animacy

echo "Creating virtual environment..."
uv venv
source .venv/bin/activate

echo "Installing Python dependencies with uv..."
# Install vllm (this will also install torch, transformers, etc.)
echo "Installing vllm..."
uv pip install vllm tqdm pandas numpy pydantic hf_transfer openai google-genai accelerate awscli


echo "Installing animacy package in editable mode..."
uv pip install -e .


cd /workspace

# ============================================================================
# Run Experiment
# ============================================================================
echo "Running experiment with Qwen/Qwen3-30B-A3B-Instruct-2507..."

python results/q_responses/scripts/run_experiment.py \
    --backend vllm \
    --model_name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --temperature 0.7 \
    --max_tokens 1024 \
    --gpu_memory_utilization 0.9 \
    --dtype auto \
    --trust_remote_code \
    --num_samples 5


python results/q_responses/scripts/run_experiment.py \
    --backend vllm \
    --model_name google/gemma-3-27b-it \
    --temperature 0.7 \
    --max_tokens 1024 \
    --gpu_memory_utilization 0.95 \
    --dtype auto \
    --trust_remote_code \
    --num_samples 5 \
    --csv_path data/selected_words.csv \
    --output_dir results/data/gemma-3-27b-it \
    --max_model_len 4096

python results/q_responses/scripts/run_experiment.py \
    --backend vllm \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --temperature 0.7 \
    --max_tokens 1024 \
    --gpu_memory_utilization 0.95 \
    --dtype auto \
    --trust_remote_code \
    --num_samples 5 \
    --csv_path data/selected_words.csv \
    --output_dir results/data/Llama-3.3-70B-Instruct \
    --max_model_len 4096

echo "=== Experiment Complete ==="
echo "Results saved to results/q_responses/data/"
echo ""
echo "To download results to your local machine, run this command locally:"
echo "scp -P <RUNPOD_PORT> -r root@<RUNPOD_IP>:/workspace/animacy/results/q_responses/data/* ~/repos/animacy/results/q_responses/data/"
echo ""
echo "Replace <RUNPOD_IP> with your pod's public IP address and <RUNPOD_PORT> with the SSH port"
echo "Example: scp -P 19210 -r root@213.181.105.236:/workspace/animacy/results/q_responses/data/* ~/repos/animacy/results/q_responses/data/"
echo ""
echo "--- OPTION 2: Fast Transfer with Rsync (Recommended for Linux/Mac) ---"
echo "rsync -avz -e 'ssh -p <RUNPOD_PORT>' --progress root@<RUNPOD_IP>:/workspace/animacy/results/q_responses/data/ ~/repos/animacy/results/q_responses/data/"
echo ""
echo "--- OPTION 3: Ultra-Fast Transfer via RunPod Object Storage (S3) ---"
echo "1. Configure AWS CLI on RunPod (run these commands in the pod):"
echo "   export AWS_ACCESS_KEY_ID='<YOUR_RUNPOD_ACCESS_KEY>'"
echo "   export AWS_SECRET_ACCESS_KEY='<YOUR_RUNPOD_SECRET_KEY>'"
echo ""
echo "2. Download from S3 to your local machine:"
echo "   aws s3 sync s3://r7y28kemyg/animacy/results/activations/data/gemma-3-27b-it/without_sys/ results/activations/data/gemma-3-27b-it/without_sys/ --endpoint-url https://s3api-us-ca-2.runpod.io --region us-ca-2"
