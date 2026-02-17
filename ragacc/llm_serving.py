from .sglang_utils import (load_model, prepare_llm_inputs,
                           prepare_synthetic_llm_inputs_batch, extend, decode)
from .index_args import IndexArgs

class RAGAccLLM:
    def __init__(self, args, faiss_index=None):
        self.disable_llm = args.disable_llm
        if not self.disable_llm:
            print("Loading LLM model")
            llm_args = IndexArgs.get_sglang_args(args)
            self.model_runner, self.tokenizer = load_model(
                llm_args, 0, args.nccl_port
            )

    def llm_sim_generate_batch(self, batch_size, input_lens, output_lens):
        assert not self.disable_llm, "LLM is disabled."

        reqs = prepare_synthetic_llm_inputs_batch(
            batch_size, input_lens, output_lens
        )

        # Clear the pools.
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool.clear()

        max_output_len = max(output_lens)

        outputs = []
        next_token_ids, output, batch = extend(reqs, self.model_runner)
        outputs.append(output)
        for i in range(max_output_len):
            next_token_ids, output = decode(next_token_ids, batch, self.model_runner)
            outputs.append(output)
        return outputs

