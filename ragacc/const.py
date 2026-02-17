MAX_ITER = 3
CLUSTER_LIMIT = 512
INIT_HOTNESS = 1024
CACHE_FRACTION_DEFAULT = 0.5
HOTNESS_DECAY = 2
HOTNESS_INCREASE = 1024
WARM_UP_ITER = 3
SMALL_NUMBER = 1e-6


data_path_template = "{data_dir}/{pipeline}/gpt-3.5-turbo_{dataset}_topk_{topk}_nprobe_256_sample_1024.jsonl"
data_path_template_selfrag = "{data_dir}/{pipeline}/selfrag-llama2-13B_{dataset}_topk_{topk}_nprobe_256_sample_1024.jsonl"
data_path_template_flare = "{data_dir}/{pipeline}/gpt-3.5-turbo_{dataset}_topk_{topk}_nprobe_256_minprob_0.8_sample_1024.jsonl"

