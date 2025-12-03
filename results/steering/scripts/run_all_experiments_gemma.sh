#!/bin/bash
set -e

# Define base directory
BASE_DIR="$HOME/animacy"
SCRIPT_PATH="$BASE_DIR/results/steering/scripts/run_steering_experiment.py"

S3="/workspace/animacy"

echo "Starting Steering Experiments..."
echo "================================"

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
    --role_vectors_file "$BASE_DIR/results/steering/data/gemma-3-27b-it/role_vectors_avg_response_first_10_sys_diff.pkl" \
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
