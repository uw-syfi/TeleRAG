# Artifact Evaluation Documentation

TeleRAG is an efficient RAG inference system that reduces retrieval latency and improves end-to-end throughput via *lookahead retrieval*, a prefetching mechanism that predicts required data and transfers them from CPU to GPU in parallel with LLM generation. This artifact reproduces Figures 9--14 of the paper.

### Repository Overview

```
TeleRAG/
├── ragacc/                      # Core library (RAG Acceleration)
│   ├── index.py                 #   Retrieval index with cluster-based prefetching
│   ├── ragacc.py                #   Main orchestrator for retrieval & LLM services
│   ├── pipeline.py              #   RAG evaluation pipeline (linear, parallel, iterative, etc.)
│   ├── schedule.py              #   Batch scheduling algorithms (greedy, naive)
│   ├── llm_serving.py           #   LLM wrapper using SGLang
│   ├── services.py              #   ZMQ-based inter-process service infrastructure
│   └── pipeline_budgets.py      #   Prefetch budgets per GPU/dataset/pipeline
├── artifact_evaluation/         # Shell scripts for each experiment configuration
│   ├── 4090/                    #   RTX 4090 single-GPU experiments
│   ├── h100/                    #   H100 single-GPU batch experiments
│   ├── h200/                    #   H200 multi-GPU experiments
│   ├── hit_rate/                #   Prefetch hit rate calculation
│   └── smoke_test.sh            #   Quick setup verification
├── eval_ragacc_*.py             # Top-level evaluation entry points
├── calculate_hit_rate.py        # Prefetch hit rate computation
├── plot_scripts/                # Scripts to generate paper figures from CSV results
├── 3rdparty/sglang/             # SGLang (git submodule)
├── Makefile                     # Orchestrates all experiments and plots
└── Dockerfile                   # Reproducible environment (CUDA 12.8, Python 3)
```

### Hardware Requirements

- **Single-GPU experiments (4090, H100)**: 1x NVIDIA GPU (RTX 4090 or H100 80GB), 64+ GB system RAM.
- **Multi-GPU experiments (H200)**: Up to 8x NVIDIA H200 GPUs, 900+ GB system RAM (1200 GB recommended).
- PCIe Gen4/Gen5 connectivity between CPU and GPU.
- Storage: at least 400 GB recommended.

### Software Requirements

- Docker with NVIDIA Container Toolkit (`--gpus all` support).
- HuggingFace account with access to gated Llama 3 and Mistral models (see [Prerequisites](#prerequisites)).

### Validation

Reviewers can validate the artifact by:

1. Running `make smoke-test` to verify the setup works (~5 minutes, single GPU).
2. Running full experiments via `make h100`, `make 4090`, or `make h200`.
3. Generating plots via `make h100-plots`, `make 4090-plots`, or `make h200-plots` and comparing with the paper figures.

See [Expected Results](#description-of-the-plots-and-expected-results) for specific validation criteria.

## Introduction

This is the documentation for the artifact evaluation of TeleRAG. Our library is called `ragacc` (RAG Acceleration), while the baseline is faiss (https://github.com/facebookresearch/faiss).

## Installation and Setup

### Prerequisites

First, clone this GitHub repository:

```bash
git clone https://github.com/uw-syfi/TeleRAG.git
```

Then, download the models and datasets from HuggingFace. If the HuggingFace CLI tool is not installed and authorized in your environment, run the following commands to set it up. See [here](https://huggingface.co/docs/huggingface_hub/guides/cli) for more details.

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login
```

Download the dataset and models as follows. TeleRAG-Dataset consumes around 200 GB. Note that the Llama and Mistral models are gated on HuggingFace and require accepting a license agreement before download.

```bash
# Customize this variable to change the data storage location.
# The data should be stored outside this repository to avoid very large Docker images.
export TELERAG_DATA_DIR=/data 

hf download lauyeeyu/TeleRAG-Dataset --repo-type=dataset --local-dir ${TELERAG_DATA_DIR}/TeleRAG-Dataset

hf download meta-llama/Llama-3.2-3B-Instruct --local-dir ${TELERAG_DATA_DIR}/models/llama3/Meta-Llama-3.2-3B-Instruct
hf download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ${TELERAG_DATA_DIR}/models/llama3/Meta-Llama-3-8B-Instruct-hf
hf download mistralai/Mistral-Small-Instruct-2409 --local-dir ${TELERAG_DATA_DIR}/models/Mistral-Small-22B
```

### Install with Docker

To build the Docker image, run this at the root directory of the repository:

```bash
docker build -t telerag .
```

This will build a Docker image named `telerag`.

### Run with Docker

To start a container, run this at the root directory of the repository:

```bash
docker run --gpus all --cap-add=SYS_NICE -d -it --name telerag-ae -v "$(pwd)":/app -v ${TELERAG_DATA_DIR}/models:/hf_models -v ${TELERAG_DATA_DIR}/TeleRAG-Dataset:/data/ telerag:latest
```

> [!NOTE]
> If you are going to run the 8 GPU experiments, please set the
> memory limit of the container to at least 900 GB (better to be even larger).
> Here is the recommended command:
>
> ```bash
> docker run --gpus all --cap-add=SYS_NICE -d -it --name telerag-ae -m 1200g --oom-kill-disable -v "$(pwd)":/app -v ${TELERAG_DATA_DIR}/models:/hf_models -v ${TELERAG_DATA_DIR}/TeleRAG-Dataset:/data/ telerag:latest
> ```

This will start a container named `telerag-ae`.

### Start Evaluation

We recommend running the experiments in a tmux session so that you can detach and let them continue in the background.

Start a tmux session:

```bash
tmux
```

Then enter the container:

```bash
docker exec -it telerag-ae /bin/bash
```

Use `Ctrl+b` then `d` to detach from the tmux session. To reattach:

```bash
tmux a
```

### Smoke Test

Before running the full experiments, verify the setup is working with a quick smoke test. This runs a minimal experiment (16 samples, 1 run) on a single GPU:

```bash
make smoke-test
```

This should complete in ~5 minutes. If it finishes without errors, the environment is ready for full evaluation.

### Run Experiments

The whole project uses GNU Make to manage all scripts and dependencies. There are two kinds of main targets:

- Experiment targets: `4090`, `h100`, and `h200` run evaluations on the corresponding GPUs. Results are saved in the `evaluation/` directory.
- Plotting targets: `4090-plots`, `h100-plots`, and `h200-plots` generate figures from the evaluation results. Plots are saved in the `figure/` directory. See [Description of the Plots and Expected Results](#description-of-the-plots-and-expected-results) for details.

> [!CAUTION]
> Do not run experiments in parallel (e.g., `make ... -j` or `make ... & make ...`), as this will cause PCIe bandwidth contention and performance degradation.

The following table summarizes what each target runs:

| Target | GPU | What it runs | Figures |
|--------|-----|-------------|---------|
| `make 4090` | 1x RTX 4090 | Single-sample evaluation with Llama-3.2-3B and Llama-3-8B using both FAISS and TeleRAG on NQ, HotpotQA, and TriviaQA (6 pipelines each). | — |
| `make 4090-plots` | — | Generates plots from `make 4090` results. | Fig. 9(a), 9(b) |
| `make h100` | 1x H100 | Hit rate calculation (3B, 8B, 22B), batch evaluation with Llama-3-8B (batch sizes 1/2/4/8) and Mistral-Small-22B using both FAISS and TeleRAG on NQ (6 pipelines each). | — |
| `make h100-plots` | — | Generates plots from `make h100` results. | Fig. 10(a), 10(b), 12 |
| `make h200` | 4--8x H200 | Multi-GPU evaluation with Llama-3-8B on NQ, HotpotQA, and TriviaQA (8 GPUs), plus ablation studies on 4 GPUs (no scheduling, prefetch only). | — |
| `make h200-plots` | — | Generates plots from `make h200` results. | Fig. 11(a--c), 13, 14 |

For example, to run all experiments on H100:

```bash
make h100
```

To run all experiments on H100 and then generate the plots:

```bash
make h100-plots
```

#### Potential Problems in Evaluation

Most problems are related to Docker process management. **If you encounter any issues, first try restarting the container with `docker stop telerag-ae` and `docker start telerag-ae`.** Known issues include:

- `AttributeError: 'RAGAcc' object has no attribute 'llm_service_addr'`: This is likely because some subprocesses failed to start. Check **outside the container** whether any processes starting with `python3 -m ragacc...` are still occupying the GPU. If so, restart the container.
- `RuntimeError: No CUDA GPUs are available`: This is likely due to a CUDA context issue. Run `nvidia-smi` to check whether CUDA is available. If not, restart the container.

#### Potential Problems in Plotting

If the plots differ from the originals, it is likely due to interference from other processes. For example, we have observed that the pre-retrieval LLM time for ragacc can sometimes be more than twice the FAISS time under such conditions.

## Description of the Plots and Expected Results

### 4090 Plots

- `figure/rtx4090_3b.pdf`: Figure 9(a), end-to-end latency speedup with Llama-3.2-3B.
- `figure/rtx4090_8b.pdf`: Figure 9(b), end-to-end latency speedup with Llama-3-8B.

### H100 Plots

- `figure/h100_batch_per_pipeline.pdf`: Figure 10(a), end-to-end throughput on Llama-3-8B.
- `figure/h100_batch_per_pipeline_22b.pdf`: Figure 10(b), end-to-end throughput on Mistral-Small-22B.
- `figure/h100_8b_breakdown.pdf`: Figure 12, latency breakdown for Llama-3-8B on NQ with an H100 GPU at different batch sizes (nprobe = 256).

### H200 Plots

- `figure/h200_multi_gpu_nq_8b.pdf`: Figure 11(a), multi-GPU throughput on NQ with Llama-3-8B.
- `figure/h200_multi_gpu_hotpotqa_8b.pdf`: Figure 11(b), multi-GPU throughput on HotpotQA with Llama-3-8B.
- `figure/h200_multi_gpu_triviaqa_8b.pdf`: Figure 11(c), multi-GPU throughput on TriviaQA with Llama-3-8B.
- `figure/h200_throughput_nq_8b.pdf`: Figure 13, throughput of TeleRAG on the NQ dataset with different numbers of H200 GPUs (with and without cache).
- `figure/rag_schedule_overhead.pdf`: Figure 14, comparison of end-to-end latency for prefetching and cache-aware schedulers on 4 H200 GPUs.

### Hit Rate

- `evaluation/hit_rate/hit_rate_4090_3b.json`: Prefetch hit rate for Llama-3.2-3B on RTX 4090.
- `evaluation/hit_rate/hit_rate_h100_8b.json`: Prefetch hit rate for Llama-3-8B on H100.
- `evaluation/hit_rate/hit_rate_h100_22b.json`: Prefetch hit rate for Mistral-Small-22B on H100.

> [!NOTE]
> Performance numbers may vary across runs due to system load, thermal throttling, and PCIe contention. Ensure no other GPU-intensive processes are running during evaluation. If results differ significantly from expected values, re-run the experiment after restarting the Docker container.
