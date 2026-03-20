#!/bin/bash

# This script runs the hit rate calculation for all pipelines and datasets.

# Usage: ./run_calculate_hit_rate.sh [gpu_id] [output_file]
# Example: ./run_calculate_hit_rate.sh 1 my_results.json (to use GPU 1 and save to my_results.json)
# Example: ./run_calculate_hit_rate.sh (to use default GPU 0 and default output file)


# --- Configuration ---
DATA_DIR="/data/rag_data/rag_output" # Set your data directory here
MODEL_PATH="/data/Llama-3-8B" # Set your model path here

GPU_MODEL="h100" # or "rtx4090", "a6000"
BUDGET_TYPE="22b"
NPROBE=256
TOPK=3
N_SAMPLES=1024 # Use -1 for all samples, or set a specific number e.g., 512
RETRIEVAL_GPU_ID=${1:-0} # Optional: Specify GPU ID as the first argument, defaults to 0
OUTPUT_FILE=${2:-"hit_rate_results_all.json"} # Optional: Specify output file as the second argument
VM_SIZE=12.0
# ---------------------

echo "=================================================="
echo "Starting Cluster Prefetch Hit Rate Calculation"
echo "This will run for all pipelines on dataset: nq"
echo "Data Dir: ${DATA_DIR}"
echo "Model Path: ${MODEL_PATH}"
echo "GPU Model: ${GPU_MODEL}"
echo "GPU ID: ${RETRIEVAL_GPU_ID}"
echo "Output File: ${OUTPUT_FILE}"
echo "Nprobe: ${NPROBE}"
echo "Samples per dataset: ${N_SAMPLES}"
echo "=================================================="

python3 calculate_hit_rate.py \
    --data-dir "${DATA_DIR}" \
    --model-path "${MODEL_PATH}" \
    --tokenizer-model-path "${MODEL_PATH}" \
    --emb-model facebook/contriever-msmarco \
    --gpu-model "${GPU_MODEL}" \
    --topk "${TOPK}" \
    --nprobe "${NPROBE}" \
    --num-samples "${N_SAMPLES}" \
    --index-type ragacc \
    --retrieval-gpu-id "${RETRIEVAL_GPU_ID}" \
    --vm-size "${VM_SIZE}" \
    --budget-type "${BUDGET_TYPE}" \
    --output "${OUTPUT_FILE}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=================================================="
    echo "Calculation finished successfully."
    echo "Results are saved in: ${OUTPUT_FILE}"
    echo "=================================================="
else
    echo "=================================================="
    echo "Error: Script failed with exit code $EXIT_CODE"
    echo "=================================================="
fi
