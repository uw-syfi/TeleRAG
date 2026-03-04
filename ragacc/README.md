# ragacc

RAG Acceleration — the core TeleRAG library.

## File Overview

### Core Components

| File | Description |
|------|-------------|
| `ragacc.py` | Main `RAGAcc` orchestrator. Coordinates retrieval and LLM services, handles embedding generation, and manages prefetching strategies via ZMQ inter-process communication. |
| `index.py` | `RAGAccIndex` class implementing cluster-based prefetching for retrieval. Supports both GPU-accelerated and CPU search, cache management, and multi-GPU simulation. |
| `pipeline.py` | RAG pipeline evaluation logic. Handles request batching, scheduling, prefetching strategies (gradual, runtime, once, all), and performance benchmarking across pipeline types (linear, parallel, iterative, iterretgen, flare, selfrag). |
| `llm_serving.py` | `RAGAccLLM` class that loads and manages LLM models using SGLang for batch generation and simulated LLM inference. |
| `schedule.py` | Batch scheduling algorithms: greedy batching by cluster overlap (1/2-optimal), naive sequential batching, and greedy grouping mini-batch using cosine similarity. |

### Services (Multi-GPU)

| File | Description |
|------|-------------|
| `services.py` | Service framework with base `Service` class, `RetrievalService`, `LLMService`, and `RagService` implementations. Uses ZMQ-based request/reply messaging and a `ServiceManager` for service discovery and lifecycle. |
| `rag_service.py` | Entry point that runs `RagService` as a standalone subprocess for orchestrating RAG pipeline requests. |
| `retrieval_service.py` | Entry point that runs `RetrievalService` as a standalone subprocess for handling retrieval and prefetch requests. |
| `llm_service.py` | Entry point that runs `LLMService` as a standalone subprocess for handling LLM generation requests. |

### Configuration

| File | Description |
|------|-------------|
| `arguments.py` | CLI argument parsers for batch evaluation, retrieval, LLM, and RAGAcc services. Defines options for GPU selection, prefetch budgets, batch sizes, and scheduling strategies. |
| `index_args.py` | `IndexArgs` dataclass for configuring the retrieval index: type (faiss/ragacc), nprobe, topk, GPU selection, prefetch buffer size, and embedding dimensions. Also wraps SGLang `ServerArgs`. |
| `pipeline_budgets.py` | Prefetch budget lookup tables tuned per dataset (NQ, HotpotQA, TriviaQA), GPU model (H100, RTX 4090), and model size (small, large, 22B). |
| `const.py` | Global constants: `MAX_ITER`, `CLUSTER_LIMIT`, `CACHE_FRACTION_DEFAULT`, hotness parameters, and data path templates. |

### Utilities

| File | Description |
|------|-------------|
| `zmq_utils.py` | Async wrapper (`async_send_recv`) for ZMQ-based IPC with pickle serialization. |
| `numa.py` | NUMA utilities for binding processes to specific NUMA nodes and mapping GPU IDs to NUMA node IDs. |
| `sglang_utils.py` | Utilities for loading LLM models via SGLang and preparing synthetic inputs for benchmarking. |
| `faiss_utils.py` | Utility to extract inverted list content (document IDs and codes) from FAISS index structures. |
| `prompt_templates.py` | Warm-up prompt and output templates used for LLM initialization and testing. |
| `test_utils.py` | Testing and benchmarking utilities for validating index correctness, measuring search performance, and comparing against FAISS baselines. |

### Other

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization. Re-exports key classes (`RAGAccIndex`, `Pipeline`, `RAGAcc`, `RAGAccLLM`, `IndexArgs`) and scheduling functions. Patches a `transformers` library safety check. |
| `depre.py` | Deprecated helper functions for CPU-based cluster search (legacy, no longer in active use). |
