#!/bin/bash
set -e

# Define base directory
BASE_DIR="$HOME/animacy"
SCRIPT_PATH="$BASE_DIR/results/steering/scripts/run_steering_experiment.py"

S3="/workspace/animacy"

echo "Starting Steering Experiments..."
echo "================================"

# 1. Qwen Role Responses
echo "[1/4] Running Qwen Role Responses experiment..."

echo "Running avg_response:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/q_responses/data/Qwen3-30B-A3B-Instruct-2507/" \
    --output_file "$S3/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_responses/avg_response.csv" \
    --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --role_vectors_file "$BASE_DIR/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_vectors_avg_response.pkl" \
    --magnitudes 0.3 0.6 1 1.3 1.6 2.3 2.6 3 \
    --no_system_prompt \
    --batch_size 26

echo "Running avg_response_sys_diff:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/q_responses/data/Qwen3-30B-A3B-Instruct-2507/" \
    --output_file "$S3/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_responses/avg_response_sys_diff.csv" \
    --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --role_vectors_file "$BASE_DIR/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_vectors_avg_response_sys_diff.pkl" \
    --magnitudes 0.3 0.6 1 1.3 1.6 2.3 2.6 3 \
    --no_system_prompt \
    --batch_size 26

echo "Running avg_response_first_10:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/q_responses/data/Qwen3-30B-A3B-Instruct-2507/" \
    --output_file "$S3/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_responses/avg_response_first_10.csv" \
    --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --role_vectors_file "$BASE_DIR/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_vectors_avg_response_first_10_tokens.pkl" \
    --magnitudes 0.3 0.6 1 1.3 1.6 2.3 2.6 3 \
    --no_system_prompt \
    --batch_size 26

echo "Running avg_response_first_10_sys_diff:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/q_responses/data/Qwen3-30B-A3B-Instruct-2507/" \
    --output_file "$S3/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_responses/avg_response_first_10_sys_diff.csv" \
    --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --role_vectors_file "$BASE_DIR/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_vectors_avg_response_first_10_tokens_sys_diff.pkl" \
    --magnitudes 0.3 0.6 1 1.3 1.6 2.3 2.6 3 \
    --no_system_prompt \
    --batch_size 26

echo "Running at_role_response:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/q_responses/data/Qwen3-30B-A3B-Instruct-2507/" \
    --output_file "$S3/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_responses/at_role_response.csv" \
    --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --role_vectors_file "$BASE_DIR/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_vectors_at_role.pkl" \
    --magnitudes 0.3 0.6 1 1.3 1.6 2.3 2.6 3 \
    --no_system_prompt \
    --batch_size 26

echo "Running at_role_period_response:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/q_responses/data/Qwen3-30B-A3B-Instruct-2507/" \
    --output_file "$S3/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_responses/at_role_period_response.csv" \
    --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --role_vectors_file "$BASE_DIR/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_vectors_at_role_period.pkl" \
    --magnitudes 0.3 0.6 1 1.3 1.6 2.3 2.6 3 \
    --no_system_prompt \
    --batch_size 26

# 2. Qwen Word Guess
echo "[2/4] Running Qwen Word Guess experiment..."
echo "Running avg_response:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/steering/data/word_guess/" \
    --output_file "$S3/results/steering/data/Qwen3-30B-A3B-Instruct-2507/word_guess/avg_response.csv" \
    --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --role_vectors_file "$BASE_DIR/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_vectors_avg_response.pkl" \
    --magnitudes 0 0.3 0.6 1 1.3 1.6 2.3 2.6 3 \
    --no_system_prompt \
    --batch_size 8

echo "Running avg_response_sys_diff:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/steering/data/word_guess/" \
    --output_file "$S3/results/steering/data/Qwen3-30B-A3B-Instruct-2507/word_guess/avg_response_sys_diff.csv" \
    --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --role_vectors_file "$BASE_DIR/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_vectors_avg_response_sys_diff.pkl" \
    --magnitudes 0 0.3 0.6 1 1.3 1.6 2.3 2.6 3 \
    --no_system_prompt \
    --batch_size 8

echo "Running avg_response_first_10_tokens:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/steering/data/word_guess/" \
    --output_file "$S3/results/steering/data/Qwen3-30B-A3B-Instruct-2507/word_guess/avg_response_first_10.csv" \
    --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --role_vectors_file "$BASE_DIR/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_vectors_avg_response_first_10_tokens.pkl" \
    --magnitudes 0 0.3 0.6 1 1.3 1.6 2.3 2.6 3 \
    --no_system_prompt \
    --batch_size 8

echo "Running avg_response_first_10_tokens_sys_diff:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/steering/data/word_guess/" \
    --output_file "$S3/results/steering/data/Qwen3-30B-A3B-Instruct-2507/word_guess/avg_response_first_10_sys_diff.csv" \
    --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --role_vectors_file "$BASE_DIR/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_vectors_avg_response_first_10_tokens_sys_diff.pkl" \
    --magnitudes 0 0.3 0.6 1 1.3 1.6 2.3 2.6 3 \
    --no_system_prompt \
    --batch_size 8

echo "Running at_role_response:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/steering/data/word_guess/" \
    --output_file "$S3/results/steering/data/Qwen3-30B-A3B-Instruct-2507/word_guess/at_role_response.csv" \
    --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --role_vectors_file "$BASE_DIR/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_vectors_at_role.pkl" \
    --magnitudes 0 0.3 0.6 1 1.3 1.6 2.3 2.6 3 \
    --no_system_prompt \
    --batch_size 8

echo "Running at_role_period_response:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/steering/data/word_guess/" \
    --output_file "$S3/results/steering/data/Qwen3-30B-A3B-Instruct-2507/word_guess/at_role_period_response.csv" \
    --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --role_vectors_file "$BASE_DIR/results/steering/data/Qwen3-30B-A3B-Instruct-2507/role_vectors_at_role_period.pkl" \
    --magnitudes 0 0.3 0.6 1 1.3 1.6 2.3 2.6 3 \
    --no_system_prompt \
    --batch_size 8

# 3. Gemma Role Responses
echo "[3/4] Running Gemma Role Responses experiment..."
echo "Running avg_response:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/q_responses/data/gemma-3-27b-it/" \
    --output_file "$S3/results/steering/data/gemma-3-27b-it/role_responses/avg_response.csv" \
    --model_name "google/gemma-3-27b-it" \
    --role_vectors_file "$BASE_DIR/results/steering/data/gemma-3-27b-it/role_vectors_avg_response.pkl" \
    --magnitudes 500 1000 2000 3000 4000 5000 6000 7000 8000 \
    --no_system_prompt \
    --batch_size 26

echo "Running avg_response_sys_diff:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/q_responses/data/gemma-3-27b-it/" \
    --output_file "$S3/results/steering/data/gemma-3-27b-it/role_responses/avg_response_sys_diff.csv" \
    --model_name "google/gemma-3-27b-it" \
    --role_vectors_file "$BASE_DIR/results/steering/data/gemma-3-27b-it/role_vectors_avg_response_sys_diff.pkl" \
    --magnitudes 500 1000 2000 3000 4000 5000 6000 7000 8000 \
    --no_system_prompt \
    --batch_size 26

echo "Running avg_response_first_10_tokens:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/q_responses/data/gemma-3-27b-it/" \
    --output_file "$S3/results/steering/data/gemma-3-27b-it/role_responses/avg_response_first_10.csv" \
    --model_name "google/gemma-3-27b-it" \
    --role_vectors_file "$BASE_DIR/results/steering/data/gemma-3-27b-it/role_vectors_avg_response_first_10_tokens.pkl" \
    --magnitudes 500 1000 2000 3000 4000 5000 6000 7000 8000 \
    --no_system_prompt \
    --batch_size 26

echo "Running avg_response_first_10_tokens_sys_diff:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/q_responses/data/gemma-3-27b-it/" \
    --output_file "$S3/results/steering/data/gemma-3-27b-it/role_responses/avg_response_first_10_sys_diff.csv" \
    --model_name "google/gemma-3-27b-it" \
    --role_vectors_file "$BASE_DIR/results/steering/data/gemma-3-27b-it/role_vectors_avg_response_first_10_tokens_sys_diff.pkl" \
    --magnitudes 500 1000 2000 3000 4000 5000 6000 7000 8000 \
    --no_system_prompt \
    --batch_size 26

echo "Running at_role_response:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/q_responses/data/gemma-3-27b-it/" \
    --output_file "$S3/results/steering/data/gemma-3-27b-it/role_responses/at_role_response.csv" \
    --model_name "google/gemma-3-27b-it" \
    --role_vectors_file "$BASE_DIR/results/steering/data/gemma-3-27b-it/role_vectors_at_role.pkl" \
    --magnitudes 500 1000 2000 3000 4000 5000 6000 7000 8000 \
    --no_system_prompt \
    --batch_size 26

echo "Running at_role_period_response:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/q_responses/data/gemma-3-27b-it/" \
    --output_file "$S3/results/steering/data/gemma-3-27b-it/role_responses/at_role_period_response.csv" \
    --model_name "google/gemma-3-27b-it" \
    --role_vectors_file "$BASE_DIR/results/steering/data/gemma-3-27b-it/role_vectors_at_role_period.pkl" \
    --magnitudes 500 1000 2000 3000 4000 5000 6000 7000 8000 \
    --no_system_prompt \
    --batch_size 26

# 4. Gemma Word Guess
echo "[4/4] Running Gemma Word Guess experiment..."
echo "Running avg_response:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/steering/data/word_guess/" \
    --output_file "$S3/results/steering/data/gemma-3-27b-it/word_guess_steered/avg_response.csv" \
    --model_name "google/gemma-3-27b-it" \
    --role_vectors_file "$BASE_DIR/results/steering/data/gemma-3-27b-it/role_vectors_avg_response.pkl" \
    --magnitudes 0 500 1000 2000 3000 4000 5000 6000 7000 8000 \
    --no_system_prompt \
    --batch_size 8

echo "Running avg_response_sys_diff:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/steering/data/word_guess/" \
    --output_file "$S3/results/steering/data/gemma-3-27b-it/word_guess_steered/avg_response_sys_diff.csv" \
    --model_name "google/gemma-3-27b-it" \
    --role_vectors_file "$BASE_DIR/results/steering/data/gemma-3-27b-it/role_vectors_avg_response_sys_diff.pkl" \
    --magnitudes 0 500 1000 2000 3000 4000 5000 6000 7000 8000 \
    --no_system_prompt \
    --batch_size 8

echo "Running avg_response_first_10_tokens:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/steering/data/word_guess/" \
    --output_file "$S3/results/steering/data/gemma-3-27b-it/word_guess_steered/avg_response_first_10.csv" \
    --model_name "google/gemma-3-27b-it" \
    --role_vectors_file "$BASE_DIR/results/steering/data/gemma-3-27b-it/role_vectors_avg_response_first_10_tokens.pkl" \
    --magnitudes 0 500 1000 2000 3000 4000 5000 6000 7000 8000 \
    --no_system_prompt \
    --batch_size 8

echo "Running avg_response_first_10_tokens_sys_diff:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/steering/data/word_guess/" \
    --output_file "$S3/results/steering/data/gemma-3-27b-it/word_guess_steered/avg_response_first_10_sys_diff.csv" \
    --model_name "google/gemma-3-27b-it" \
    --role_vectors_file "$BASE_DIR/results/steering/data/gemma-3-27b-it/role_vectors_avg_response_first_10_tokens_sys_diff.pkl" \
    --magnitudes 0 500 1000 2000 3000 4000 5000 6000 7000 8000 \
    --no_system_prompt \
    --batch_size 8

  echo "Running at_role_response:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/steering/data/word_guess/" \
    --output_file "$S3/results/steering/data/gemma-3-27b-it/word_guess_steered/at_role_response.csv" \
    --model_name "google/gemma-3-27b-it" \
    --role_vectors_file "$BASE_DIR/results/steering/data/gemma-3-27b-it/role_vectors_at_role.pkl" \
    --magnitudes 0 500 1000 2000 3000 4000 5000 6000 7000 8000 \
    --no_system_prompt \
    --batch_size 8

echo "Running at_role_period_response:"
python "$SCRIPT_PATH" \
    --input_dir "$BASE_DIR/results/steering/data/word_guess/" \
    --output_file "$S3/results/steering/data/gemma-3-27b-it/word_guess_steered/at_role_period_response.csv" \
    --model_name "google/gemma-3-27b-it" \
    --role_vectors_file "$BASE_DIR/results/steering/data/gemma-3-27b-it/role_vectors_at_role_period.pkl" \
    --magnitudes 0 500 1000 2000 3000 4000 5000 6000 7000 8000 \
    --no_system_prompt \
    --batch_size 8

echo "================================"
echo "All experiments completed successfully!"
