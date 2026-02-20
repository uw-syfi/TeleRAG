# Makefile for RAGAcc evaluation

# --- Configuration ---
GPU_ID ?= 0

# --- H100 Evaluation ---
.PHONY: h100
h100: hit_rate llama_8b_batch mistral_22b_batch llama_8b_nprobe

# --- Hit Rate Evaluation ---
.PHONY: all hit_rate hit_rate_4090_3b hit_rate_h100_22b hit_rate_h100_8b

hit_rate: hit_rate_4090_3b hit_rate_h100_22b hit_rate_h100_8b
hit_rate_4090_3b: evaluation/hit_rate/hit_rate_4090_3b.json
hit_rate_h100_22b: evaluation/hit_rate/hit_rate_h100_22b.json
hit_rate_h100_8b: evaluation/hit_rate/hit_rate_h100_8b.json

evaluation/hit_rate/hit_rate_4090_3b.json: calculate_hit_rate.py ./artifact-evaluation/hit_rate/run_calculate_hit_rate_4090_3b.sh
	@mkdir -p $(@D)
	./artifact-evaluation/hit_rate/run_calculate_hit_rate_4090_3b.sh $(GPU_ID) $@

evaluation/hit_rate/hit_rate_h100_22b.json: calculate_hit_rate.py ./artifact-evaluation/hit_rate/run_calculate_hit_rate_h100_22b.sh
	@mkdir -p $(@D)
	./artifact-evaluation/hit_rate/run_calculate_hit_rate_h100_22b.sh $(GPU_ID) $@


evaluation/hit_rate/hit_rate_h100_8b.json: calculate_hit_rate.py ./artifact-evaluation/hit_rate/run_calculate_hit_rate_h100_8b.sh
	@mkdir -p $(@D)
	./artifact-evaluation/hit_rate/run_calculate_hit_rate_h100_8b.sh $(GPU_ID) $@

# --- Llama 8B Evaluation ---
.PHONY: llama_8b_batch llama_8b_batch_faiss llama_8b_batch_ragacc llama_8b_nprobe llama_8b_nprobe_faiss llama_8b_nprobe_ragacc

llama_8b_batch: llama_8b_batch_faiss llama_8b_batch_ragacc
llama_8b_nprobe: llama_8b_nprobe_faiss llama_8b_nprobe_ragacc
llama_8b_nprobe_faiss: evaluation/h100/llama_8b_nprobe/nq_faiss_topk_3_ndata_1024.csv
llama_8b_nprobe_ragacc: evaluation/h100/llama_8b_nprobe/nq_ragacc_topk_3_ndata_1024.csv
llama_8b_batch_faiss: evaluation/h100/llama_8b_batch/nq_faiss_topk_3_ndata_1024.csv
llama_8b_batch_ragacc: evaluation/h100/llama_8b_batch/nq_ragacc_mini_greedy_topk_3_ndata_1024.csv

evaluation/h100/llama_8b_batch/nq_faiss_topk_3_ndata_1024.csv: artifact-evaluation/h100/llama_8b_batch.sh
	@mkdir -p $(@D)
	./artifact-evaluation/h100/llama_8b_batch.sh faiss

evaluation/h100/llama_8b_batch/nq_ragacc_mini_greedy_topk_3_ndata_1024.csv: artifact-evaluation/h100/llama_8b_batch.sh
	@mkdir -p $(@D)
	./artifact-evaluation/h100/llama_8b_batch.sh ragacc

evaluation/h100/llama_8b_nprobe/nq_faiss_topk_3_ndata_1024.csv: artifact-evaluation/h100/llama_8b_nprobe.sh
	@mkdir -p $(@D)
	./artifact-evaluation/h100/llama_8b_nprobe.sh faiss

evaluation/h100/llama_8b_nprobe/nq_ragacc_topk_3_ndata_1024.csv: artifact-evaluation/h100/llama_8b_nprobe.sh
	@mkdir -p $(@D)
	./artifact-evaluation/h100/llama_8b_nprobe.sh ragacc

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
.PHONY: h200 h200_llama_8b_8_gpu h200_llama_8b_4_gpu_no_cache h200_llama_8b_4_gpu_no_schedule h200_llama_8b_4_gpu_prefetch_only

h200: h200_llama_8b_8_gpu h200_llama_8b_4_gpu_no_schedule h200_llama_8b_4_gpu_prefetch_only

h200_llama_8b_8_gpu: evaluation/h200/llama_8b_8_gpu/nq_ragacc_mini_greedy_topk_3_ndata_512.csv \
                     evaluation/h200/llama_8b_8_gpu/hotpotqa_ragacc_mini_greedy_topk_3_ndata_512.csv \
                     evaluation/h200/llama_8b_8_gpu/triviaqa_ragacc_mini_greedy_topk_3_ndata_512.csv
h200_llama_8b_4_gpu_no_schedule: evaluation/h200/llama_8b_4_gpu_no_schedule/nq_ragacc_topk_3_ndata_512.csv
h200_llama_8b_4_gpu_prefetch_only: evaluation/h200/llama_8b_4_gpu_prefetch_only/nq_ragacc_mini_greedy_topk_3_ndata_512.csv

evaluation/h200/llama_8b_8_gpu/nq_ragacc_mini_greedy_topk_3_ndata_512.csv \
evaluation/h200/llama_8b_8_gpu/hotpotqa_ragacc_mini_greedy_topk_3_ndata_512.csv \
evaluation/h200/llama_8b_8_gpu/triviaqa_ragacc_mini_greedy_topk_3_ndata_512.csv: artifact-evaluation/h200/llama_8b_8_gpu.sh
	@mkdir -p $(@D)
	./artifact-evaluation/h200/llama_8b_8_gpu.sh ragacc

evaluation/h200/llama_8b_4_gpu_no_schedule/nq_ragacc_topk_3_ndata_512.csv: artifact-evaluation/h200/llama_8b_4_gpu_no_schedule.sh
	@mkdir -p $(@D)
	./artifact-evaluation/h200/llama_8b_4_gpu_no_schedule.sh ragacc

evaluation/h200/llama_8b_4_gpu_prefetch_only/nq_ragacc_mini_greedy_topk_3_ndata_512.csv: artifact-evaluation/h200/llama_8b_4_gpu_prefetch_only.sh
	@mkdir -p $(@D)
	./artifact-evaluation/h200/llama_8b_4_gpu_prefetch_only.sh ragacc

# --- 4090 Evaluation ---
.PHONY: 4090 4090_llama_3b 4090_llama_3b_faiss 4090_llama_3b_ragacc 4090_llama_8b 4090_llama_8b_faiss 4090_llama_8b_ragacc 4090_llama_3b_nprobe 4090_llama_3b_nprobe_faiss 4090_llama_3b_nprobe_ragacc

4090: 4090_llama_3b 4090_llama_3b_nprobe 4090_llama_8b

4090_llama_3b: 4090_llama_3b_faiss 4090_llama_3b_ragacc
4090_llama_3b_faiss: evaluation/4090/llama_3b/nq_faiss_topk_3_ndata_1024.csv \
                     evaluation/4090/llama_3b/hotpotqa_faiss_topk_3_ndata_1024.csv \
                     evaluation/4090/llama_3b/triviaqa_faiss_topk_3_ndata_1024.csv
4090_llama_3b_ragacc: evaluation/4090/llama_3b/nq_ragacc_topk_3_ndata_1024.csv \
                      evaluation/4090/llama_3b/hotpotqa_ragacc_topk_3_ndata_1024.csv \
                      evaluation/4090/llama_3b/triviaqa_ragacc_topk_3_ndata_1024.csv

evaluation/4090/llama_3b/nq_faiss_topk_3_ndata_1024.csv \
evaluation/4090/llama_3b/hotpotqa_faiss_topk_3_ndata_1024.csv \
evaluation/4090/llama_3b/triviaqa_faiss_topk_3_ndata_1024.csv: artifact-evaluation/4090/llama_3b.sh
	@mkdir -p $(@D)
	./artifact-evaluation/4090/llama_3b.sh faiss

evaluation/4090/llama_3b/nq_ragacc_topk_3_ndata_1024.csv \
evaluation/4090/llama_3b/hotpotqa_ragacc_topk_3_ndata_1024.csv \
evaluation/4090/llama_3b/triviaqa_ragacc_topk_3_ndata_1024.csv: artifact-evaluation/4090/llama_3b.sh
	@mkdir -p $(@D)
	./artifact-evaluation/4090/llama_3b.sh ragacc

4090_llama_8b: 4090_llama_8b_faiss 4090_llama_8b_ragacc
4090_llama_8b_faiss: evaluation/4090/llama_8b/nq_faiss_topk_3_ndata_1024.csv \
                     evaluation/4090/llama_8b/hotpotqa_faiss_topk_3_ndata_1024.csv \
                     evaluation/4090/llama_8b/triviaqa_faiss_topk_3_ndata_1024.csv
4090_llama_8b_ragacc: evaluation/4090/llama_8b/nq_ragacc_topk_3_ndata_1024.csv \
                      evaluation/4090/llama_8b/hotpotqa_ragacc_topk_3_ndata_1024.csv \
                      evaluation/4090/llama_8b/triviaqa_ragacc_topk_3_ndata_1024.csv

evaluation/4090/llama_8b/nq_faiss_topk_3_ndata_1024.csv \
evaluation/4090/llama_8b/hotpotqa_faiss_topk_3_ndata_1024.csv \
evaluation/4090/llama_8b/triviaqa_faiss_topk_3_ndata_1024.csv: artifact-evaluation/4090/llama_8b.sh
	@mkdir -p $(@D)
	./artifact-evaluation/4090/llama_8b.sh faiss

evaluation/4090/llama_8b/nq_ragacc_topk_3_ndata_1024.csv \
evaluation/4090/llama_8b/hotpotqa_ragacc_topk_3_ndata_1024.csv \
evaluation/4090/llama_8b/triviaqa_ragacc_topk_3_ndata_1024.csv: artifact-evaluation/4090/llama_8b.sh
	@mkdir -p $(@D)
	./artifact-evaluation/4090/llama_8b.sh ragacc

4090_llama_3b_nprobe: 4090_llama_3b_nprobe_faiss 4090_llama_3b_nprobe_ragacc
4090_llama_3b_nprobe_faiss: evaluation/4090/llama_3b_nprobe/nq_faiss_topk_3_ndata_1024.csv
4090_llama_3b_nprobe_ragacc: evaluation/4090/llama_3b_nprobe/nq_ragacc_topk_3_ndata_1024.csv

evaluation/4090/llama_3b_nprobe/nq_faiss_topk_3_ndata_1024.csv: artifact-evaluation/4090/llama_3b_nprobe.sh
	@mkdir -p $(@D)
	./artifact-evaluation/4090/llama_3b_nprobe.sh faiss

evaluation/4090/llama_3b_nprobe/nq_ragacc_topk_3_ndata_1024.csv: artifact-evaluation/4090/llama_3b_nprobe.sh
	@mkdir -p $(@D)
	./artifact-evaluation/4090/llama_3b_nprobe.sh ragacc
