import numpy as np
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.utils import suppress_other_loggers
from sglang.srt.hf_transformers_utils import get_tokenizer

def load_model(server_args, tp_rank, nccl_port=28888):
    assert server_args.tp_size <= 1, "Only support one GPU as now"
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    model_config = ModelConfig(
        server_args.model_path,
        server_args.trust_remote_code,
        context_length=server_args.context_length,
    )
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        # nccl_port=28888,
        nccl_port=nccl_port,
        server_args=server_args,
    )
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    # if server_args.tp_size > 1:
    #     dist.barrier()
    return model_runner, tokenizer

def prepare_llm_inputs(tokenizer, prompts, output_len):
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=output_len,
    )

    reqs = []
    for i in range(len(prompts)):
        req = Req(rid=i, origin_input_text=prompts[i], origin_input_ids=input_ids[i])
        req.prefix_indices = []
        req.sampling_params = sampling_params
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        reqs.append(req)

    return input_ids, reqs

def prepare_synthetic_llm_inputs(batch_size, input_len, output_len):
    input_ids = np.ones((batch_size, input_len), dtype=np.int32)
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(rid=i, origin_input_text="", origin_input_ids=list(input_ids[i]))
        req.prefix_indices = []
        req.sampling_params = sampling_params
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        reqs.append(req)

    return reqs

def prepare_synthetic_llm_inputs_batch(
        batch_size: int, 
        input_lens: list[int], 
        output_lens: list[int],
        skip_output: bool = False,
    ):
    assert len(input_lens) == batch_size
    assert len(output_lens) == batch_size

    # TODO: Check here. In the real case, input_lens in a batch could be the same because of padding from tokenizer.
    # (CY) Checked at 11.19.2024. Using different input_lens for each input in a batch leads to better performance.
    input_ids = [np.ones(input_len, dtype=np.int32) for input_len in input_lens]

    # max_input_len = max(input_lens)
    # input_ids = [np.ones(max_input_len, dtype=np.int32) for _ in range(batch_size)]

    sampling_params = []
    if not skip_output:
        for i in range(batch_size):
            sampling_params.append(SamplingParams(
                temperature=0,
                max_new_tokens=output_lens[i],
            ))

    reqs = []
    for i in range(batch_size):
        req = Req(rid=i, origin_input_text="", origin_input_ids=list(input_ids[i]))
        req.prefix_indices = []
        if not skip_output:
            req.sampling_params = sampling_params[i]
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        reqs.append(req)

    return reqs

@torch.inference_mode()
# @nvtx.annotate(message="sglang.extend", color="orange")
def extend(reqs, model_runner):
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool,
        tree_cache=None,
    )
    batch.prepare_for_extend(model_runner.model_config.vocab_size)
    logits_output = model_runner.forward(batch)
    next_token_ids = model_runner.sample(logits_output, batch).tolist()
    return next_token_ids, logits_output.next_token_logits, batch

@torch.inference_mode()
# @nvtx.annotate(message="sglang.decode", color="yellow")
def decode(input_token_ids, batch, model_runner):
    batch.prepare_for_decode(input_token_ids)
    logits_output = model_runner.forward(batch)
    next_token_ids = model_runner.sample(logits_output, batch).tolist()
    return next_token_ids, logits_output.next_token_logits
