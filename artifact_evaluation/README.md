# artifact_evaluation

Shell scripts that configure and launch experiments for each hardware setup. Each script takes an index type (`faiss` or `ragacc`) as its first argument unless otherwise noted.

## File Overview

### Smoke Test

| Script | Description |
|--------|-------------|
| `smoke_test.sh` | Quick setup verification. Runs hit rate + batch evaluation with Llama-3-8B, ragacc index, 16 samples, 1 run on a single GPU (~5 min). Takes an optional GPU ID argument. |

### RTX 4090 (Single-GPU Latency)

| Script | Model | Samples | Runs | Description |
|--------|-------|---------|------|-------------|
| `4090/llama_3b.sh` | Llama-3.2-3B | 1024 | 5 | Single-sample evaluation on NQ, HotpotQA, and TriviaQA with all 6 pipelines. |
| `4090/llama_8b.sh` | Llama-3-8B | 1024 | 5 | Single-sample evaluation on NQ, HotpotQA, and TriviaQA with all 6 pipelines. |

### H100 (Single-GPU Throughput)

| Script | Model | Samples | Runs | Description |
|--------|-------|---------|------|-------------|
| `h100/llama_8b_batch.sh` | Llama-3-8B | 1024 | 5 | Batch evaluation on NQ with batch size 128, mini-batch size 4, greedy mini-batch scheduling. |
| `h100/mistral_22b_batch.sh` | Mistral-Small-22B | 1024 | 5 | Batch evaluation on NQ with batch size 128, mini-batch size 4, greedy mini-batch scheduling. |

### H200 (Multi-GPU Scaling)

| Script | Model | GPUs | Samples | Runs | Description |
|--------|-------|------|---------|------|-------------|
| `h200/llama_8b_8_gpu.sh` | Llama-3-8B | 1/2/4/8 | 512 | 5 | Full multi-GPU evaluation on NQ, HotpotQA, and TriviaQA with greedy scheduling and varying cache fractions. |
| `h200/llama_8b_4_gpu_no_schedule.sh` | Llama-3-8B | 4 | 512 | 5 | Ablation: 4-GPU with naive batching and no cache-aware scheduling. |
| `h200/llama_8b_4_gpu_prefetch_only.sh` | Llama-3-8B | 4 | 512 | 5 | Ablation: 4-GPU with greedy batching and prefetch-only (no cache scheduling). |

### Hit Rate

These scripts take an optional GPU ID as the first argument and output path as the second.

| Script | Model | GPU Config | Samples | Description |
|--------|-------|------------|---------|-------------|
| `hit_rate/run_calculate_hit_rate_4090_3b.sh` | Llama-3-8B | RTX 4090 (vm=10 GB) | 1024 | Prefetch hit rate with small budget type. |
| `hit_rate/run_calculate_hit_rate_h100_8b.sh` | Llama-3-8B | H100 (vm=12 GB) | 1024 | Prefetch hit rate with small budget type. |
| `hit_rate/run_calculate_hit_rate_h100_22b.sh` | Mistral-Small-22B | H100 (vm=12 GB) | 1024 | Prefetch hit rate with 22B budget type. |
