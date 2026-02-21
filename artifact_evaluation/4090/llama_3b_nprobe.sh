#!/bin/bash

if [ "$1" == "" ]; then
    echo "Usage: $0 <index_type>"
    exit 1
fi

INDEX_TYPE=$1

PREFETCH_METHOD=gradual
NPROBE=256
TOPK=3
VM_SIZE=10
N_SAMPLES=1024
N_RUNS=5
MODEL=llama3/Meta-Llama-3.2-3B-Instruct

echo "Index type: ${INDEX_TYPE}"
echo "Nprobe: ${NPROBE}"
echo "Prefetch method: ${PREFETCH_METHOD}"

python3 eval_ragacc_nprobe.py \
    --emb-model facebook/contriever-msmarco \
    --data-dir /data/rag_data/rag_output \
    --model-path /hf_models/${MODEL} \
    --tokenizer-model-path /hf_models/${MODEL} \
    --log-dir evaluation/4090/llama_3b_nprobe \
    --mem-fraction-static 0.4 \
    --gpu-model rtx4090 \
    --topk ${TOPK} \
    --nprobe ${NPROBE} \
    --index-type ${INDEX_TYPE} \
    --vm-size ${VM_SIZE} \
    --prefetch-strategy ${PREFETCH_METHOD} \
    --num-samples ${N_SAMPLES} \
    --batch-strategy naive \
    --mini-batch-strategy naive \
    --batch-size 1 \
    --mini-batch-size 1 \
    --multi-gpu \
    --num-gpu 1 \
    --profile
