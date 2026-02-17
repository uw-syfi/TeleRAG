import argparse

from .arguments import add_args_for_llm
from .services import LLMService

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process arguments for Retrieval Service.'
    )
    add_args_for_llm(parser)
    args, _ = parser.parse_known_args()
    _ = LLMService(
        args, port=args.llm_port, byte_mode=False, stand_alone=True,
    )
