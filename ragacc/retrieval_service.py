import argparse

from .arguments import add_args_for_retrieval
from .services import RetrievalService

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process arguments for Retrieval Service.'
    )
    add_args_for_retrieval(parser)
    args, _ = parser.parse_known_args()
    _ = RetrievalService(
        args, port=args.retrieval_port, byte_mode=False, stand_alone=True,
    )
