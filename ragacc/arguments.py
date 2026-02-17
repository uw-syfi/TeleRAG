from .index_args import IndexArgs

def add_args_for_batch(parser):
    IndexArgs.add_cli_args(parser)
    parser.add_argument(
        '--tokenizer-model-path',
        default="/data/llama3/Meta-Llama-3-8B-Instruct-hf",
        help='Root directory for the tokenizer model',
    )
    parser.add_argument(
        '--data-dir',
        default="/data/rag_data/rag_output",
        help='Root directory for the evaluation data',
    )
    parser.add_argument(
        '--log-dir',
        default="eval_results",
        help='File to output the evaluation results',
    )
    parser.add_argument(
        '--emb-model',
        default="facebook/contriever-msmarco",
        choices=["facebook/contriever-msmarco", "google-bert/bert-base-uncased"],
        help='select the emb model used by the index',
    )
    parser.add_argument(
        '--gpu-model',
        default="h100",
        choices=["h100", "rtx4090", "a6000"],
    )
    parser.add_argument(
        '--num-samples',
        default=-1,
        type=int,
        help='Number of sample to evaluate. Default: -1 (all samples)',
    )
    parser.add_argument(
        '--num-runs',
        default=3,
        type=int,
        help='Number of runs to benchmark. Default: 3',
    )
    parser.add_argument(
        '--gpu-only-search',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--prefetch-budget',
        default=None,
        type=float,
        help='Prefetch budget for the evaluation. Will override the default budget.',
    )
    parser.add_argument(
        '--budget-type',
        default="small",
        type=str,
        choices=["small", "large", "22b"],
        help='Prefetch budget for the evaluation. Will override the default budget.',
    )
    parser.add_argument(
        '--prefetch-strategy',
        type=str,
        choices=["all", "runtime", "once", "gradual"],
        default="gradual",
        help='Which prefetch strategy to use. Default: all',
    )
    parser.add_argument(
        '--enable-speculative-branch',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--disable-bench-llm',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--disable-bench-retrieval',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--profile-cache',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--disable-log',
        action='store_true',
        default=False
    )
    # parser.add_argument(
    #     '--trivial-batch-strategy',
    #     default=False,
    #     action='store_true',
    #     help='Use naive batch strategy for the evaluation. Default: False',
    # )
    parser.add_argument(
        '--batch-size',
        default=-1,
        type=int,
        help='Global batch size to process at one time. If set to -1, it will be number of total samples.',
    )
    parser.add_argument(
        '--mini-batch-size',
        default=1,
        type=int,
        help='Determine the batch size to process each time for each GPU.',
    )
    parser.add_argument(
        '--batch-strategy',
        type=str,
        choices=["naive", "greedy"],
        default="naive",
        help='How to batch queries from the global pool. Done in off-line. Default: naive',
    )
    parser.add_argument(
        '--mini-batch-strategy',
        type=str,
        choices=["naive", "greedy"],
        default="naive",
        help='How to batch queries from the global pool. Process at runtime. Default: naive',
    )
    parser.add_argument(
        '--num-gpu',
        default=1,
        type=int,
        help='Number of GPUs. Default: 1',
    )
    parser.add_argument(
        '--multi-gpu',
        default=False,
        action='store_true',
        help='Use multiple GPUs for the evaluation. Default: False',
    )
    parser.add_argument(
        '--sim-multi-gpu',
        default=False,
        action='store_true',
        help='Use simulation mode for multi GPU. Default: True',
    )
    parser.add_argument(
        '--disable-cache',
        default=False,
        action='store_true',
        help='Disable cache for the evaluation. Default: False',
    )
    parser.add_argument(
        '--no-cache-schedule',
        default=False,
        action='store_true',
        help='Disable cache-aware scheduling for the evaluation. Default: False',
    )

def add_args_for_retrieval(parser):
    add_args_for_batch(parser)
    parser.add_argument(
        '--retrieval-port',
        type=int,
        help='Port for the retrieval service to listen on.',
    )

def add_args_for_llm(parser):
    add_args_for_batch(parser)
    parser.add_argument(
        '--llm-port',
        type=int,
        help='Port for the retrieval service to listen on.',
    )
    parser.add_argument(
        '--nccl-port',
        type=int,
        help='NCCL port for launching multiple sglang backends (srt) on different gpus.',
    )

def add_args_for_ragacc(parser):
    add_args_for_batch(parser)
    parser.add_argument(
        '--service-port',
        type=int,
        help='Port for the RAG service to listen on.',
    )
    parser.add_argument(
        '--retrieval-port',
        type=int,
        help='Port for the retrieval service to listen on.',
    )
    parser.add_argument(
        '--llm-port',
        type=int,
        help='Port for the LLM service to listen on.',
    )
    parser.add_argument(
        '--retrieval-gpu-id',
        type=int,
        help='GPU ID for the retrieval service',
    )
    parser.add_argument(
        '--llm-gpu-id',
        type=int,
        help='GPU ID for the LLM service',
    )
    parser.add_argument(
        '--nccl-port',
        type=int,
        help='NCCL port for launching multiple sglang backends (srt) on different gpus.',
    )
    parser.add_argument(
        '--numa-node',
        type=int,
        default=-1,
        help='NUMA node for the services. Default: -1 (no NUMA binding)',
    )