import argparse

from ragacc import Pipeline, add_args_for_batch
from ragacc.services import start_and_register_all_services, service_manager

rag_configs = [
    {
        'numa_node': 0,
        'service_port': 29001,
        'retrieval_port': 29102,
        'llm_port': 29003,
        'retrieval_gpu_id': 0,
        'llm_gpu_id': 0,
        'nccl_port': 29004,
    },
    {
        'numa_node': 0,
        'service_port': 29011,
        'retrieval_port': 29012,
        'llm_port': 29013,
        'retrieval_gpu_id': 1,
        'llm_gpu_id': 1,
        'nccl_port': 29014,
    },
    {
        'numa_node': 0,
        'service_port': 29021,
        'retrieval_port': 29022,
        'llm_port': 29023,
        'retrieval_gpu_id': 2,
        'llm_gpu_id': 2,
        'nccl_port': 29024,
    },
    {
        'numa_node': 0,
        'service_port': 29031,
        'retrieval_port': 29032,
        'llm_port': 29033,
        'retrieval_gpu_id': 3,
        'llm_gpu_id': 3,
        'nccl_port': 29034,
    },
    {
        'numa_node': 1,
        'service_port': 29041,
        'retrieval_port': 29042,
        'llm_port': 29043,
        'retrieval_gpu_id': 4,
        'llm_gpu_id': 4,
        'nccl_port': 29044,
    },
    {
        'numa_node': 1,
        'service_port': 29051,
        'retrieval_port': 29052,
        'llm_port': 29053,
        'retrieval_gpu_id': 5,
        'llm_gpu_id': 5,
        'nccl_port': 29054,
    },
    {
        'numa_node': 1,
        'service_port': 29061,
        'retrieval_port': 29062,
        'llm_port': 29063,
        'retrieval_gpu_id': 6,
        'llm_gpu_id': 6,
        'nccl_port': 29064,
    },
    {
        'numa_node': 1,
        'service_port': 29071,
        'retrieval_port': 29072,
        'llm_port': 29073,
        'retrieval_gpu_id': 7,
        'llm_gpu_id': 7,
        'nccl_port': 29074,
    }
]

def choose_gpu(num_gpu: int):
    interval = len(rag_configs) // num_gpu
    chosen = []
    for i in range(num_gpu):
        chosen.append(rag_configs[i * interval])
    return chosen

class RagAccEvaluator:
    def __init__(self, args):
        self.pipeline = Pipeline()
        start_and_register_all_services(choose_gpu(args.num_gpu), args)

    def evaluate(self, args, budget=None, batch_size=1, mini_batch=1):
        self.pipeline.evaluate(
            args, budget, args.enable_speculative_branch,
            batch_size, mini_batch, args.num_gpu,
        )

    @staticmethod
    def shutdown():
        service_manager.shutdown_all_matching_services('rag')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments for RAGAcc.')
    add_args_for_batch(parser)

    args = parser.parse_args()
    evaluator = RagAccEvaluator(args)

    pipeline_list = ["linear", "parallel", "iterative", "iterretgen", "flare", "selfrag"]
    dataset_list = ["hotpotqa", "triviaqa", "nq"]
    nprobe_list = [256]
    cache_fraction_list = [0.5]

    for dataset in dataset_list:
        for pipeline in pipeline_list:
            for i, nprobe in enumerate(nprobe_list):
                if i > 0:
                    args.disable_bench_llm = True
                else:
                    args.disable_bench_llm = False

                args.pipeline_type = pipeline
                args.datasets = dataset
                args.nprobe = nprobe

                for cache_fraction in cache_fraction_list:
                    args.cache_fraction = cache_fraction
                    evaluator.evaluate(
                        args, args.prefetch_budget, args.batch_size,
                        args.mini_batch_size,
                    )

    evaluator.shutdown()
