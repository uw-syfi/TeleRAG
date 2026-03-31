# TeleRAG: Efficient Retrieval-Augmented Generation Inference with Lookahead Retrieval

![TeleRAG-design](assets/design.svg)

[![arXiv](https://img.shields.io/badge/arXiv-2502.20969-b31b1b.svg)](https://arxiv.org/abs/2502.20969)
[![DOI](https://zenodo.org/badge/1159746702.svg)](https://doi.org/10.5281/zenodo.19361856)

[TeleRAG][telerag-paper] is an efficient inference system that reduces latency and improves throughput with minimal GPU memory requirements. The core innovation of TeleRAG is *lookahead retrieval*, a prefetching mechanism that predicts required data and transfers them from CPU to GPU in parallel with LLM generation. In addition, TeleRAG adopts a prefetching scheduler and a cache-aware scheduler to support efficient multi-GPU inference with minimal overhead.

[telerag-paper]: https://arxiv.org/abs/2502.20969

This repo includes:

- TeleRAG's implementation: the `ragacc` library, which stands for RAG Acceleration.
- Scripts to run experiments and generate plots.

## Artifact Evaluation

Please refer to [docs/artifact-evaluation.md](docs/artifact-evaluation.md) for details.

## Evaluation Scripts

### Entry Points

| Script | Description |
|--------|-------------|
| `eval_ragacc_single.py` | Single-sample evaluation across all 6 pipelines, 3 datasets, and nprobe=256. Used for single-GPU latency measurements (RTX 4090). |
| `eval_ragacc_batch.py` | Batch evaluation across all 6 pipelines with batch sizes 1/2/4/8 on NQ. Used for single-GPU throughput measurements (H100). |
| `eval_ragacc_nprobe.py` | Nprobe sensitivity analysis across nprobe values 128/256/512 on NQ with batch size 1. |
| `eval_ragacc_4_gpu_nq.py` | 4-GPU multi-GPU evaluation on NQ with varying batch sizes and cache fractions. |
| `eval_ragacc_8_gpu.py` | 8-GPU multi-GPU evaluation on NQ, HotpotQA, and TriviaQA with varying batch sizes, cache fractions, and GPU counts (1--8). |
| `calculate_hit_rate.py` | Computes prefetch cluster hit rates by comparing predicted prefetch clusters against actual retrieval clusters for each pipeline. |

### Make Targets

All experiments are orchestrated via `make`. Do not run targets in parallel (running two `make` commands at the same time) due to PCIe bandwidth contention.

| Target | GPU | Description |
|--------|-----|-------------|
| `make smoke-test` | 1x any | Quick setup verification (16 samples, 1 run, ~5 min). |
| `make 4090` | 1x RTX 4090 | Single-sample latency with Llama-3.2-3B and Llama-3-8B using FAISS and TeleRAG on NQ, HotpotQA, and TriviaQA. |
| `make 4090-plots` | — | Generates Figures 9(a,b) from `make 4090` results. |
| `make h100` | 1x H100 | Hit rate calculation + batch throughput with Llama-3-8B and Mistral-Small-22B using FAISS and TeleRAG on NQ. |
| `make h100-plots` | — | Generates Figures 10(a,b) and 12, and Table 3 from `make h100` results. |
| `make h200` | 4--8x H200 | Multi-GPU throughput with Llama-3-8B on NQ, HotpotQA, and TriviaQA, plus scheduling ablations on 4 GPUs. |
| `make h200-plots` | — | Generates Figures 11(a--c), 13, and 14 from `make h200` results. |
| `make hit_rate` | 1x any | Prefetch hit rate calculation for 3B, 8B, and 22B models. |

Results are saved in `evaluation/`, plots in `figure/`.

## Citation

```bibtex
@article{lin2025telerag,
  title={{TeleRAG}: Efficient retrieval-augmented generation inference with lookahead retrieval},
  author={Lin, Chien-Yu and Kamahori, Keisuke and Liu, Yiyu and Shi, Xiaoxiang and Kashyap, Madhav and Gu, Yile and Shao, Rulin and Ye, Zihao and Zhu, Kan and Kadekodi, Rohan and others},
  journal={arXiv preprint arXiv:2502.20969},
  year={2025}
}
```

## License

This project is licensed under the terms of the Apache 2.0 license. See [LICENSE](LICENSE) for more details.
