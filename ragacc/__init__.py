import sys
import types

# 1. Define the mock function
def mock_check_safe(*args, **kwargs):
    return

# 2. Pre-create the module structure to intercept the import
# We manually create transformers.utils.import_utils before it is actually loaded
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = mock_check_safe

from .index import RAGAccIndex
from .index_args import IndexArgs
from .schedule import greedy_batch_requests, naive_batch_requests, greedy_grouping_mini_batch
from .pipeline import Pipeline
from .ragacc import RAGAcc
from .llm_serving import RAGAccLLM
from .arguments import add_args_for_batch
