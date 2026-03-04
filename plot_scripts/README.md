# plot_scripts

Scripts that generate paper figures from CSV evaluation results.

## File Overview

| Script | Paper Figure | Description | Key Arguments |
|--------|-------------|-------------|---------------|
| `plot_rtx4090.py` | Fig. 9(a,b) | End-to-end latency speedup across NQ, HotpotQA, and TriviaQA for six RAG pipelines, comparing FAISS vs. TeleRAG on RTX 4090. | `--faiss_nq`, `--ragacc_nq`, `--faiss_hotpot`, `--ragacc_hotpot`, `--faiss_trivia`, `--ragacc_trivia`, `--output` |
| `plot_batch.py` | Fig. 10(a,b) | Throughput and speedup across batch sizes (1, 2, 4, 8) for six RAG pipelines, comparing FAISS baseline vs. TeleRAG. | `--baseline`, `--ragacc`, `--output` |
| `plot_result_breakdown.py` | Fig. 12 | Stacked bar chart breaking down per-request time (LLM, retrieval, misc) across batch sizes, comparing FAISS vs. TeleRAG. | `--faiss`, `--ragacc`, `--output` |
| `plot_multi_gpu.py` | Fig. 11(a--c) | Throughput scaling from 1 to 8 H200 GPUs for six RAG pipelines with cache fraction 0.5. | `--csv`, `--output` |
| `plot_h200_throughput_cache.py` | Fig. 13 | Throughput comparison across 1--8 GPUs with and without cache (cache fraction 0.0 vs. 0.5) for six pipelines. | `--csv`, `--output` |
| `plot_h200_schedule_overhead.py` | Fig. 14 | Stacked bar chart of scheduling overhead: naive retrieval vs. prefetch-only vs. prefetch+cache across six pipelines on 4 H200 GPUs. | `--no_schedule`, `--prefetch_only`, `--with_cache`, `--output` |
| `plot_retrieval_speedup.py` | — | Retrieval-only speedup across nprobe values (128, 256, 512) for six RAG pipelines, comparing FAISS vs. TeleRAG. | `--faiss`, `--ragacc`, `--name`, `--output` |
