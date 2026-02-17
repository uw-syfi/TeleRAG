import argparse
import dataclasses

from sglang.srt.server_args import ServerArgs

@dataclasses.dataclass
class IndexArgs:
    # Index setup 
    index_type: str = "ragacc"
    index_key: str = "IVF4096,FlatIP"
    index_load_dir: str = "/data/wiki_dpr/contriever-msmarco/index/IVF4096,FlatIP"
    save_invlist_data: bool = False
    index_save_dir: str = "/data/wiki_dpr/contriever-msmarco/index/IVF4096,FlatIP"
    nprobe: int = 32
    topk: int = 10
    gpu_id: int = 0

    # Optimization
    disable_prefetch: bool = False
    disable_llm: bool = False
    disable_retrieval: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        # parser = argparse.ArgumentParser(description='Process arguments for RAGFlow.')

        parser.add_argument(
            '--index-type', 
            choices=['faiss', 'ragacc'],
            default='faiss',
        )
        parser.add_argument(
            '--from-faiss', 
            action='store_true',
            default=False,
            help="Whether to load and convert the index from Faiss."
        )
        parser.add_argument(
            '--use-faiss-gpu', 
            action='store_true',
            default=False,
            help="Whether to load faiss index to GPU."
        )
        parser.add_argument(
            '--index-key', 
            type=str,
            default="IVF4096,FlatIP",
            help='The config of Index (in Faiss style).'
        )
        parser.add_argument(
            '--index-load-dir', 
            type=str,
            default="/data/wiki_dpr/contriever-msmarco/index/IVF4096,FlatIP",
            help='Path to load the invlist data.'
        )
        parser.add_argument(
            '--save-invlist-data', 
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--index-save-dir', 
            type=str,
            default="/data/wiki_dpr/contriever-msmarco/index/IVF4096,FlatIP",
            help='Directory to save the invlist data.'
        )
        parser.add_argument(
            '--nprobe', 
            type=int,
            default=32,
        )
        parser.add_argument(
            '--topk', 
            type=int,
            default=3,
        )
        parser.add_argument(
            '--gpu-id', 
            type=int,
            default=0,
        )
        parser.add_argument(
            '--disable-prefetch', 
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--disable-gpu-sort', 
            action='store_true',
            default=False
        )
        # parser.add_argument(
        #     '--gpu-only-search', 
        #     action='store_true',
        #     default=False
        # )
        parser.add_argument(
            '--disable-llm', 
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--disable-retrieval', 
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--max-cpu-threads', 
            type=int,
            default=128,
            help='Max number of threads used for CPU search.'
        )
        parser.add_argument(
            '--vm-size', 
            type=float,
            default=12.0,
            help='Prefetch buffer size in GB.'
        )
        parser.add_argument(
            '--embed-dim', 
            type=int,
            default=768,
            help='Embedding dim for index.'
        )
        parser.add_argument(
            '--cpu-only',
            action='store_true',
            default=False,
            help='Whether to run on CPU only.'
        )

        ServerArgs.add_cli_args(parser)

    @classmethod
    def get_sglang_args(cls, args: argparse.Namespace):
        sglang_args = ServerArgs.from_cli_args(args)
        return sglang_args

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # args.tp_size = args.tensor_parallel_size
        # args.dp_size = args.data_parallel_size
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})