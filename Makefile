# Makefile for RAGAcc evaluation

# --- Configuration ---
GPU_ID ?= 0
OUTPUT_DIR ?= evaluation/hit_rate

EVALUATION_FILES = $(OUTPUT_DIR)/hit_rate_4090_3b.json \
                   $(OUTPUT_DIR)/hit_rate_h100_22b.json \
                   $(OUTPUT_DIR)/hit_rate_h100_8b.json

# --- Hit Rate Evaluation ---
.PHONY: all hit_rate_4090_3b hit_rate_h100_22b hit_rate_h100_8b

all: $(EVALUATION_FILES)

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
