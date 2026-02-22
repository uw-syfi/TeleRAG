# Artifact Evaluation Documentation

## Installation and Setup

### Prerequisites

#### Models and Datasets

Please make sure the models are placed in a directory, for example `/data/hf_models`.

You should download the following models/datasets at the following paths:

- `llama3/Meta-Llama-3-8B-Instruct-hf` at `llama3/Meta-Llama-3-8B-Instruct-hf`.
- `llama3/Meta-Llama-3.2-3B-Instruct` at `llama3/Meta-Llama-3.2-3B-Instruct`.
- `mistralai/Mistral-Small-22B` at `Mistral-Small-22B`.

You should also download the dataset `lauyeeyu/TeleRAG-Dataset`.

You may use the following script at anywhere (preferably in a directory outside the repo) to download the models and the dataset:

```bash
huggingface-cli download lauyeeyu/TeleRAG-Dataset --local-dir TeleRAG-Dataset
huggingface-cli download llama3/Meta-Llama-3-8B-Instruct-hf --local-dir models/llama3/Meta-Llama-3-8B-Instruct-hf
huggingface-cli download llama3/Meta-Llama-3.2-3B-Instruct --local-dir models/llama3/Meta-Llama-3.2-3B-Instruct
huggingface-cli download mistralai/Mistral-Small-22B --local-dir models/Mistral-Small-22B
```


### Install with Docker

To build the docker image, run this at the root directory of the repository:

```bash
docker build -t telerag .
```

This will build a docker image named `telerag`.

### Run with Docker

To start a container, run this at the root directory of the repository: (replace `/path/to/hf_models` and `/path/to/TeleRAG-Dataset` with the paths to your hf_models and TeleRAG-Dataset directories)

```
docker run --gpus all --cap-add=SYS_NICE -d -it --name telerag-ae -v "$(pwd)":/app -v /path/to/hf_models:/hf_models -v /path/to/TeleRAG-Dataset:/data/ telerag:latest
```

This will start a container named `telerag-ae`.

### Start Evaluation

It is recommended to run the experiments in a tmux session, so that you can detach from the session and the experiments will continue running in the background.

To start the evaluation, run this:

```bash
tmux
```

Then use enter the container:

```bash
docker exec -it telerag-ae /bin/bash
```

You may use `Ctrl+b` and `d` to detach from the tmux session. If you want to reattach to the session, run this:

```bash
tmux attach
```

### Run Experiments

The whole project uses GNU Make to manage all scripts and dependencies. There are two kinds of main targets:

- Experiement targets: `4090`, `h100`, and `h200` for the evaluations on the corresponding GPUs.
- Plotting targets: `4090-plots`, `h100-plots`, and `h200-plots` for plotting the results of the corresponding GPUs.

For example, if you want to run all the experiments on h100, you can run this:

```bash
make h100
```

If you want to run all the experiments on h100 and plot the results, you can run this:

```bash
make h100-plots
```

#### Potential Problems

- `AttributeError: 'RAGAcc' object has no attribute 'llm_service_addr'`: This is likely because some subprocesses failed to start. You should check **outside the container** if any subprocesses starts with `python3 -m ragacc...` is still occupying the GPU. If so, you should kill them by restarting the container with `docker stop telerag-ae` and `docker start telerag-ae`.
- `RuntimeError: No CUDA GPUs are available`: This is likely because of some issue with the CUDA context. You can try to run `nvidia-smi` to check if the CUDA is available. If not, you can try to restart the container with `docker stop telerag-ae` and `docker start telerag-ae`.

