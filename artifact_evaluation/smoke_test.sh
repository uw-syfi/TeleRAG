#!/bin/bash

# Smoke test for TeleRAG artifact evaluation.
# Runs a minimal experiment (16 samples, 1 run, 1 pipeline, 1 dataset)
# to verify the setup is working before running full experiments.
#
# Usage: ./artifact_evaluation/smoke_test.sh [gpu_id]
# Expected runtime: ~5 minutes on a single GPU

set -e

GPU_ID=${1:-0}
INDEX_TYPE=ragacc
PREFETCH_METHOD=gradual
NPROBE=256
TOPK=3
VM_SIZE=12
N_SAMPLES=16
N_RUNS=1
MODEL=llama3/Meta-Llama-3-8B-Instruct-hf
LOG_DIR=evaluation/smoke_test

echo "=================================================="
echo "TeleRAG Smoke Test"
echo "GPU ID: ${GPU_ID}"
echo "Samples: ${N_SAMPLES}, Runs: ${N_RUNS}"
echo "=================================================="

echo ""
echo "[1/2] Running hit rate calculation (ragacc index, NQ, 16 samples)..."
python3 calculate_hit_rate.py \
    --data-dir /data/rag_data/rag_output \
    --model-path /hf_models/${MODEL} \
    --tokenizer-model-path /hf_models/${MODEL} \
    --emb-model facebook/contriever-msmarco \
    --gpu-model h100 \
    --topk ${TOPK} \
    --nprobe ${NPROBE} \
    --num-samples ${N_SAMPLES} \
    --index-type ragacc \
    --retrieval-gpu-id ${GPU_ID} \
    --vm-size ${VM_SIZE} \
    --budget-type small \
    --output evaluation/smoke_test_hit_rate.json

echo ""
echo "[2/2] Running batch evaluation (ragacc, NQ, linear pipeline, batch=1, 16 samples)..."
mkdir -p ${LOG_DIR}
python3 eval_ragacc_batch.py \
    --emb-model facebook/contriever-msmarco \
    --data-dir /data/rag_data/rag_output \
    --model-path /hf_models/${MODEL} \
    --tokenizer-model-path /hf_models/${MODEL} \
    --log-dir ${LOG_DIR} \
    --mem-fraction-static 0.4 \
    --topk ${TOPK} \
    --nprobe ${NPROBE} \
    --index-type ${INDEX_TYPE} \
    --vm-size ${VM_SIZE} \
    --prefetch-strategy ${PREFETCH_METHOD} \
    --num-samples ${N_SAMPLES} \
    --num-runs ${N_RUNS} \
    --batch-strategy naive \
    --mini-batch-strategy greedy \
    --batch-size 16 \
    --mini-batch-size 1 \
    --multi-gpu \
    --num-gpu 1 \
    --gpu-id ${GPU_ID} \
    --profile

echo ""
echo "=================================================="
echo "Smoke test completed successfully."
echo "Hit rate results: evaluation/smoke_test_hit_rate.json"
echo "Batch results:    ${LOG_DIR}/"
echo "=================================================="
