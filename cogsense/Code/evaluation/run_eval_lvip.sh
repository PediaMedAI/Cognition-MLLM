#!/bin/bash

# Evaluate CogSense-8B on CogSense-Bench with a single GPU
# Usage: bash evaluation/run_eval_lvip.sh

cd "$(dirname "$0")"

# Hugging Face model id or a local checkpoint directory
MODEL_PATH="PediaMedAI/CogSense-8B"
# CogSense-Bench jsonl; image paths in it are relative to its directory
BENCH_PATH="../data/CogSense-Bench/CogSense-Bench.jsonl"
OUTPUT_DIR="../outputs/eval/cogsense-8b"

# Use LLM extraction (GPT-4o-mini, needs OPENAI_API_KEY) instead of rule-based extraction
# Set to "true" to enable, "false" to use rule-based extraction
USE_LLM_EXTRACTION="true"

# Random seed for shuffling data (set to same value for reproducibility)
RANDOM_SEED=42

# Set to "true" to disable shuffling (use sequential order)
NO_SHUFFLE="false"

# Set to "true" for a quick test run on 5 samples
TEST_MODE="false"

# Specify which GPU to use
GPU_ID=0

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Starting Evaluation"
echo "Model: ${MODEL_PATH}"
echo "Benchmark: ${BENCH_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# Build command arguments
if [ "${USE_LLM_EXTRACTION}" = "true" ]; then
    LLM_FLAG="--use_llm_extraction"
    echo "Answer Extraction: LLM (GPT-4o-mini)"
else
    LLM_FLAG=""
    echo "Answer Extraction: Rule-based"
fi

if [ "${NO_SHUFFLE}" = "true" ]; then
    SHUFFLE_FLAG="--no_shuffle"
    echo "Data Order: Sequential (no shuffle)"
else
    SHUFFLE_FLAG="--seed ${RANDOM_SEED}"
    echo "Data Order: Random shuffle (seed=${RANDOM_SEED})"
fi

if [ "${TEST_MODE}" = "true" ]; then
    TEST_FLAG="--test_mode"
    echo "Test Mode: Enabled (5 samples)"
else
    TEST_FLAG=""
    echo "Test Mode: Disabled (full benchmark)"
fi
echo "=========================================="

# Run evaluation
CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python eval_lvip.py \
    --model_path ${MODEL_PATH} \
    --bench_path ${BENCH_PATH} \
    --output_dir ${OUTPUT_DIR} \
    ${LLM_FLAG} \
    ${SHUFFLE_FLAG} \
    ${TEST_FLAG} > ${OUTPUT_DIR}/eval_lvip.log 2>&1 & PID=$!

echo "Evaluation running in background with PID: $PID"

# Monitor progress
tail -f ${OUTPUT_DIR}/eval_lvip.log & TAIL_PID=$!

wait $PID
kill $TAIL_PID

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
