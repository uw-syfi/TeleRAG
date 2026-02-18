#!/bin/bash

if [ "$1" == "" ]; then
    echo "Usage: $0 <index_type>"
    exit 1
fi

INDEX_TYPE=$1

PREFETCH_METHOD=gradual
NPROBE=256
TOPK=3
VM_SIZE=24
N_SAMPLES=1024
N_RUNS=5
MODEL=llama3/Meta-Llama-3-8B-Instruct-hf

echo "Index type: ${INDEX_TYPE}"
echo "Nprobe: ${NPROBE}"
echo "Prefetch method: ${PREFETCH_METHOD}"

python3 eval_ragacc_batch.py \
    --emb-model facebook/contriever-msmarco \
    --data-dir /data/rag_data/rag_output \
    --model-path /hf_models/${MODEL} \
    --tokenizer-model-path /hf_models/${MODEL} \
    --log-dir evaluation/h100/llama_8b \
    --mem-fraction-static 0.4 \
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
    --num-gpu 1 \
    --profile
