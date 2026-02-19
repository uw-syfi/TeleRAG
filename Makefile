# Makefile for RAGAcc evaluation

# --- Configuration ---
GPU_ID ?= 0
OUTPUT_DIR ?= evaluation/hit_rate

EVALUATION_FILES = $(OUTPUT_DIR)/hit_rate_4090_3b.json \
                   $(OUTPUT_DIR)/hit_rate_h100_22b.json \
                   $(OUTPUT_DIR)/hit_rate_h100_8b.json

# --- Hit Rate Evaluation ---
.PHONY: all hit_rate hit_rate_4090_3b hit_rate_h100_22b hit_rate_h100_8b

h100: hit_rate llama_8b_batch mistral_22b_batch

hit_rate: hit_rate_4090_3b hit_rate_h100_22b hit_rate_h100_8b
hit_rate_4090_3b: $(OUTPUT_DIR)/hit_rate_4090_3b.json
hit_rate_h100_22b: $(OUTPUT_DIR)/hit_rate_h100_22b.json
hit_rate_h100_8b: $(OUTPUT_DIR)/hit_rate_h100_8b.json

$(OUTPUT_DIR)/hit_rate_4090_3b.json: calculate_hit_rate.py ./artifact-evaluation/hit_rate/run_calculate_hit_rate_4090_3b.sh
	@mkdir -p $(@D)
	./artifact-evaluation/hit_rate/run_calculate_hit_rate_4090_3b.sh $(GPU_ID) $@

$(OUTPUT_DIR)/hit_rate_h100_22b.json: calculate_hit_rate.py ./artifact-evaluation/hit_rate/run_calculate_hit_rate_h100_22b.sh
	@mkdir -p $(@D)
	./artifact-evaluation/hit_rate/run_calculate_hit_rate_h100_22b.sh $(GPU_ID) $@


$(OUTPUT_DIR)/hit_rate_h100_8b.json: calculate_hit_rate.py ./artifact-evaluation/hit_rate/run_calculate_hit_rate_h100_8b.sh
	@mkdir -p $(@D)
	./artifact-evaluation/hit_rate/run_calculate_hit_rate_h100_8b.sh $(GPU_ID) $@

# --- Llama 8B Evaluation ---
.PHONY: llama_8b_batch llama_8b_batch_faiss llama_8b_batch_ragacc

llama_8b_batch: llama_8b_batch_faiss llama_8b_batch_ragacc
llama_8b_batch_faiss: evaluation/h100/llama_8b_batch/nq_faiss_topk_3_ndata_1024.csv
llama_8b_batch_ragacc: evaluation/h100/llama_8b_batch/nq_ragacc_mini_greedy_topk_3_ndata_1024.csv

evaluation/h100/llama_8b_batch/nq_faiss_topk_3_ndata_1024.csv: artifact-evaluation/h100/llama_8b_batch.sh
	@mkdir -p $(@D)
	./artifact-evaluation/h100/llama_8b_batch.sh faiss

evaluation/h100/llama_8b_batch/nq_ragacc_mini_greedy_topk_3_ndata_1024.csv: artifact-evaluation/h100/llama_8b_batch.sh
	@mkdir -p $(@D)
	./artifact-evaluation/h100/llama_8b_batch.sh ragacc

# --- Mistral 22B Evaluation ---
.PHONY: mistral_22b_batch mistral_22b_batch_faiss mistral_22b_batch_ragacc

mistral_22b_batch: mistral_22b_batch_faiss mistral_22b_batch_ragacc
mistral_22b_batch_faiss: evaluation/h100/mistral_22b_batch/nq_faiss_topk_3_ndata_1024.csv
mistral_22b_batch_ragacc: evaluation/h100/mistral_22b_batch/nq_ragacc_mini_greedy_topk_3_ndata_1024.csv

evaluation/h100/mistral_22b_batch/nq_faiss_topk_3_ndata_1024.csv: artifact-evaluation/h100/mistral_22b_batch.sh
	@mkdir -p $(@D)
	./artifact-evaluation/h100/mistral_22b_batch.sh faiss

evaluation/h100/mistral_22b_batch/nq_ragacc_mini_greedy_topk_3_ndata_1024.csv: artifact-evaluation/h100/mistral_22b_batch.sh
	@mkdir -p $(@D)
	./artifact-evaluation/h100/mistral_22b_batch.sh ragacc

# --- H200 Evaluation ---
.PHONY: h200 h200_llama_8b_8_gpu h200_llama_8b_4_gpu_no_cache h200_llama_8b_4_gpu_no_schedule

h200: h200_llama_8b_8_gpu h200_llama_8b_4_gpu_no_schedule

h200_llama_8b_8_gpu: evaluation/h200/llama_8b_8_gpu/nq_ragacc_mini_greedy_topk_3_ndata_512.csv
h200_llama_8b_4_gpu_no_schedule: evaluation/h200/llama_8b_4_gpu_no_schedule/nq_ragacc_topk_3_ndata_512.csv

evaluation/h200/llama_8b_8_gpu/nq_ragacc_mini_greedy_topk_3_ndata_512.csv: artifact-evaluation/h200/llama_8b_8_gpu.sh
	@mkdir -p $(@D)
	./artifact-evaluation/h200/llama_8b_8_gpu.sh ragacc

evaluation/h200/llama_8b_4_gpu_no_schedule/nq_ragacc_topk_3_ndata_512.csv: artifact-evaluation/h200/llama_8b_4_gpu_no_schedule.sh
	@mkdir -p $(@D)
	./artifact-evaluation/h200/llama_8b_4_gpu_no_schedule.sh ragacc
