import argparse

from .arguments import add_args_for_ragacc
from .services import RagService, namespace_to_args_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process arguments for RAG Service.'
    )
    add_args_for_ragacc(parser)
    args, _ = parser.parse_known_args()
    _ = RagService(
        args, port=args.service_port, byte_mode=False, stand_alone=True,
    )
