# Artifact Evaluation Documentation

## Installation and Setup

### Prerequisites

TODO: docs on how to set up wiki_dpr and rag_data.

### Install with Docker

To build the docker image, run this at the root directory of the repository:

```bash
docker build -t telerag .
```

This will build a docker image named `telerag`.

### Run with Docker

To start a container, run this: (replace `/path/to/rag_data` and `/path/to/wiki_dpr` with the paths to your rag_data and wiki_dpr directories)

```
docker run --gpus all --cap-add=SYS_NICE -d -it   --name telerag-ae   -v "$(pwd)":/app -v /data:/hf_models -v /path/to/rag_data:/data/rag_data -v /path/to/wiki_dpr:/data/wiki_dpr telerag:latest
```

This will start a container named `telerag-ae`.

### Start Evaluation

It is recommended to run the experiments in a tmux session, so that you can detach from the session and the experiments will continue running in the background.

To start the evaluation, run this:

```bash
tmux
```

You may use `Ctrl+b` and `d` to detach from the tmux session. If you want to reattach to the session, run this:

```bash
tmux attach
```

### Run Experiments

#### Potential Problems

- AttributeError: 'RAGAcc' object has no attribute 'llm_service_addr'
- RuntimeError: No CUDA GPUs are available

