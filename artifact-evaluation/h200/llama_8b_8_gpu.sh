#!/bin/bash

INDEX_TYPE=$1

PREFETCH_METHOD=gradual

NPROBE=256
TOPK=3
VM_SIZE=24
N_SAMPLES=512
N_RUNS=5
MODEL=Meta-Llama-3-8B-Instruct 

echo "Index type: ${INDEX_TYPE}"
echo "Nprobe: ${NPROBE}"
echo "Prefetch method: ${PREFETCH_METHOD}"

for NUM_GPU in 8 4 2 1; do
    echo "Num GPU: ${NUM_GPU}"
    python3 eval_ragacc_batch.py \
        --emb-model facebook/contriever-msmarco \
        --data-dir /data/rag_data/rag_output \
        --model-path /hf_models/${MODEL} \
        --tokenizer-model-path /hf_models/${MODEL} \
        --log-dir evaluation/h200/llama_8b_8_gpu \
        --mem-fraction-static 0.35 \
        --topk ${TOPK} \
        --nprobe ${NPROBE} \
        --index-type ${INDEX_TYPE} \
        --vm-size ${VM_SIZE} \
        --prefetch-strategy ${PREFETCH_METHOD} \
        --num-samples ${N_SAMPLES} \
        --batch-strategy naive \
        --mini-batch-strategy greedy \
        --batch-size 128 \
        --mini-batch-size 4 \
        --multi-gpu \
        --num-gpu ${NUM_GPU} \
        --profile
done
