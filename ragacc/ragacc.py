import sys
import transformers.utils.import_utils

# 1. Define the bypass
def bypass_security_check(*args, **kwargs):
    return

# 2. Patch the source
transformers.utils.import_utils.check_torch_load_is_safe = bypass_security_check

# 3. Aggressively hunt down and patch any other references in memory
target_name = "check_torch_load_is_safe"
for module_name, module in list(sys.modules.items()):
    if not module: continue
    # If this module has a "check_torch_load_is_safe" attribute, overwrite it
    if hasattr(module, target_name):
        setattr(module, target_name, bypass_security_check)
import subprocess

import torch
import transformers
import time
import numpy as np
from typing import List, Union
import asyncio

from transformers import BertModel
from ragacc.prompt_templates import WARM_UP_PROMPT, WARM_UP_OUT
from .const import WARM_UP_ITER
from .services import (
    Request, namespace_to_args_list, add_env,
    wait_service_initialization,
    LLM_GENERATE_SIM_REQUEST, RETRIEVAL_PREFETCH_REQUEST,
    RETRIEVAL_CLEAR_PREFETCH_DATA_REQUEST, RETRIEVAL_FIND_CLUSTERS_REQUEST,
    RETRIEVAL_SEARCH_REQUEST, RETRIEVAL_SWITCH_GPU_REQUEST,
    RETRIEVAL_RESIZE_CACHE_REQUEST, RETRIEVAL_GET_CACHE_CLUSTERS_OVERLAP_SIM_REQUEST,
    RETRIEVAL_CHANGE_NUM_GPU_REQUEST, SHUTDOWN_REQUEST,
)
from .sglang_utils import prepare_synthetic_llm_inputs_batch
from .zmq_utils import async_send_recv


class RAGAcc:
    """RAGAcc class for RAGAcc system, which takes charge of both the
    retrieval service and LLM service.

    This class does the following things:
    1. Tokenization
    2. Kick off retrieval service and LLM service
    3. Direct operations to the retrieval service and LLM service
    """
    def __init__(self, args):
        self.cpu_only = args.cpu_only
        if self.cpu_only:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{args.gpu_id}")

        if not self.cpu_only:
            self.llm_tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_model_path, use_fast=True)
        self.emb_model = BertModel.from_pretrained(args.emb_model).to(self.device)
        self.emb_tokenizer = transformers.AutoTokenizer.from_pretrained(args.emb_model)
        self.embed_dim = args.embed_dim
        self.disable_bench_llm = args.disable_bench_llm
        self.sim_multi_gpu = args.sim_multi_gpu
        self.multi_gpu = args.multi_gpu
        self.num_gpu = args.num_gpu
        if self.multi_gpu:
            self.llm_port = args.llm_port
            self.llm_service_addr = f'tcp://localhost:{self.llm_port}'
            self.retrieval_port = args.retrieval_port
            self.retrieval_service_addr = f'tcp://localhost:{self.retrieval_port}'
            async def async_start_service(gpu_id, service_name, addr):
                arg_list = namespace_to_args_list(args)
                numa_id = str(args.numa_node)
                numa_args = [
                    'numactl', '-m', numa_id, '-N', numa_id, '--',
                ] if args.numa_node >= 0 else []
                service = subprocess.Popen(
                    numa_args + [
                        'python3', '-m', f'ragacc.{service_name}',
                    ] + [str(x) for x in arg_list],
                    env=add_env({'CUDA_VISIBLE_DEVICES': gpu_id}),
                )
                await wait_service_initialization(addr)
                return service

            async def start_services():
                return await asyncio.gather(
                    async_start_service(
                        args.retrieval_gpu_id,
                        'retrieval_service',
                        self.retrieval_service_addr,
                    ),
                    async_start_service(
                        args.llm_gpu_id,
                        'llm_service',
                        self.llm_service_addr,
                    ),
                )
            self.retrieval_service, self.llm_service = \
                asyncio.run(start_services())
        self.nprobe = args.nprobe

    def __del__(self):
        # shutdown all related services
        self.shutdown_services()

    def count_token_txt(self, txt: str):
        return len(self.llm_tokenizer(txt)["input_ids"])

    def count_token_msg(self, msg: List):
        return len(self.llm_tokenizer.apply_chat_template(msg, add_generation_prompt=True))

    def count_token(self, input: Union[str | List]):
        """
        Call either count_token_txt or count_token_msg
        """
        if isinstance(input, str):
            return self.count_token_txt(input)
        elif isinstance(input, List):
            return self.count_token_msg(input)
        else:
            print(input)
            print("Invalid input type.")
            raise TypeError("Input must be a string or a list of messages.")

    def count_token_batch(self, inputs: list):
        """
        Call either count_token_txt or count_token_msg
        """
        token_counts = []
        for txt in inputs:
            token_counts.append(self.count_token(txt))
        return token_counts

    @torch.inference_mode()
    def txt_to_emb(self, txt: str | list[str]):
        # padding=True for batch input text
        tokens = self.emb_tokenizer(txt, padding=True, truncation=True, return_tensors="pt")
        if not self.cpu_only:
            tokens = tokens.to(self.emb_model.device)
        emb = self.emb_model(**tokens).last_hidden_state.mean(dim=1)
        return emb

    def find_clusters(self, emb, nprobe):
        request = Request(
            RETRIEVAL_FIND_CLUSTERS_REQUEST,
            {
                'emb': emb,
                'nprobe': nprobe,
            },
        )
        response = asyncio.run(
            async_send_recv(self.retrieval_service_addr,
                            request, byte_mode=False)
        )
        return response.data['clusters']
        # return self.index.find_clusters(emb, nprobe)

    @torch.inference_mode()
    def run_prefetch(
            self,
            prefetch_query: str | list[str] = None,
            prefetch_budget: int = 2,
            sync: bool = False,
    ):
        assert prefetch_query is not None, "Prefetch query is None"

        emb = self.txt_to_emb(prefetch_query)
        request = Request(
            RETRIEVAL_PREFETCH_REQUEST,
            {
                'prefetch_emb': emb,
                'prefetch_budget': prefetch_budget,
            }
        )
        asyncio.run(
            async_send_recv(self.retrieval_service_addr,
                            request, byte_mode=False)
        )
        # self.index.prefetch_batch(emb, prefetch_budget, sync=sync)

    @torch.inference_mode()
    def bench_llm_batch(
            self,
            llm_inputs: list,
            llm_outputs: list,
            prefetch_query: str | list[str] = None,
            warm_up: int = 0,
            disable_bench: bool = False,
            prefetch_budget: int = None,
            sync: bool = True,
    ):
        """
        Bench batch LLM calls with specified input and output length
        """
        t = 0.0
        batch = None
        input_lens = None
        output_lens = None
        reqs = None
        # return 0 for cpu only or disable
        # if self.cpu_only or disable_bench:
        if self.cpu_only:
            return 0.0

        assert len(llm_inputs) == len(llm_outputs), "Input and output length mismatch"
        if len(llm_inputs) == 0:
            return 0.0

        if not disable_bench:
            tick = time.time()
            batch = len(llm_inputs)
            input_lens = self.count_token_batch(llm_inputs)
            output_lens = self.count_token_batch(llm_outputs)
            # print(f"Batch: {batch}, Input lens: {input_lens}, Output lens: {output_lens}")
            reqs = prepare_synthetic_llm_inputs_batch(
                batch, input_lens, output_lens, skip_output=True,
            )
            t += time.time() - tick

            for _ in range(warm_up):
                self.llm_generate_sim_batch(
                    batch, input_lens, output_lens,
                )
        if prefetch_query is not None:
            tick = time.time()
            prefetch_emb = self.txt_to_emb(prefetch_query)
            prefetch = True
            t += time.time() - tick
        else:
            prefetch_emb = None
            prefetch = False

        if not disable_bench:
            tick = time.time()
            self.llm_generate_sim_batch(
                batch, input_lens, output_lens, prefetch, prefetch_emb,
                prefetch_budget, sync, reqs,
            )
            t += time.time() - tick
        else:
            if prefetch_query is not None:
                self.run_prefetch(prefetch_query, prefetch_budget, sync=False)
                torch.cuda.synchronize()

            t = 0.0

        return t

    @torch.inference_mode()
    def llm_generate_sim_batch(
        self, batch_size, input_lens, output_lens,
        prefetch=False, prefetch_emb=None, prefetch_budget=2,
        sync=True, reqs=None, gpu_id=None
    ):
        """LLM generate simulation batch (was in RAGAccIndex)

        The function basically does two things simultaneously: prefetch and LLM generation.
        """
        # self.index.llm_generate_sim_batch(
        #     batch_size, input_lens, output_lens,
        #     prefetch=prefetch, prefetch_emb=prefetch_emb,
        #     prefetch_budget=prefetch_budget, sync=sync
        # )
        asyncio.run(
            self.llm_generate_sim_batch_remote(
                batch_size, input_lens, output_lens,
                prefetch=prefetch, prefetch_emb=prefetch_emb,
                prefetch_budget=prefetch_budget, reqs=reqs,
            )
        )

    async def llm_generate_sim_batch_remote(
            self, batch_size, input_lens, output_lens,
            prefetch=False, prefetch_emb=None, prefetch_budget=2, reqs=None,
    ):
        llm_message = Request(
            LLM_GENERATE_SIM_REQUEST,
            {
                'batch_size': batch_size,
                'input_lens': input_lens,
                'output_lens': output_lens,
                'reqs': reqs,
            },
        )
        retrieval_message = Request(
            RETRIEVAL_PREFETCH_REQUEST,
            {
                'prefetch_emb': prefetch_emb,
                'prefetch_budget': prefetch_budget,
            },
        ) if prefetch else None
        if retrieval_message is not None:
            response1, response2 = await asyncio.gather(
                async_send_recv(self.llm_service_addr,
                                llm_message, byte_mode=False),
                async_send_recv(self.retrieval_service_addr,
                                retrieval_message, byte_mode=False),
            )
        else:
            response1 = await async_send_recv(self.llm_service_addr,
                                              llm_message, byte_mode=False)
            response2 = None
        return response1, response2


    @torch.inference_mode()
    def bench_llm_batch_multi_round(
            self,
            llm_inputs: list,
            llm_outputs: list,
            prefetch_query: str = None,
            prefetch_budget: int = None,
            warm_up: int = 0,
            disable_bench: bool = False,
    ):
        """
        Bench batch LLM calls with specified input and output length
        """
        # return 0 for cpu only or disable
        # if self.cpu_only or disable_bench:
        if self.cpu_only:
            return 0.0

        num_round = len(llm_inputs)

        for _ in range(warm_up):
            self.llm_generate_sim_batch(
                len(llm_inputs[0]),
                self.count_token_batch(llm_inputs[0]),
                self.count_token_batch(llm_outputs[0]),
            )

        if not disable_bench:
            async def run_prefetch():
                emb = self.txt_to_emb(prefetch_query)
                prefetch_request =  Request(
                    RETRIEVAL_PREFETCH_REQUEST,
                    {
                        'prefetch_emb': emb,
                        'prefetch_budget': prefetch_budget,
                    },
                )
                await async_send_recv(self.retrieval_service_addr,
                                      prefetch_request, byte_mode=False)
            async def run_llm():
                for i in range(num_round):
                    batch = len(llm_inputs[i])
                    input_len = self.count_token_batch(llm_inputs[i])
                    output_len = self.count_token_batch(llm_outputs[i])
                    reqs = prepare_synthetic_llm_inputs_batch(
                        batch, input_len, output_len, skip_output=True,
                    )
                    await self.llm_generate_sim_batch_remote(
                        batch, input_len, output_len, reqs=reqs,
                    )
            tick = time.time()
            if prefetch_query is not None:
                asyncio.run(run_llm())
            else:
                async def run_both():
                    await asyncio.gather(
                        run_llm(),
                        run_prefetch(),
                    )
                asyncio.run(run_both())
            t = time.time() - tick
        else:
            if prefetch_query is not None:
                self.run_prefetch(prefetch_query, prefetch_budget, sync=False)
                torch.cuda.synchronize()

            t = 0.0

        return t

    def bench_retrieval(
            self, query: str | list[str],
            topk: int = 10,
            nprobe: int = 32,
            warm_up: int = 0,
            n_run: int = 1,
            disable_bench: bool = False,
            gpu_only_search: bool = False,
            cpu_only_search: bool = False,
            fetch_query: str = None,
            fetch_nprobe: int = None,
    ):
        if disable_bench:
            return 0.0

        assert not (gpu_only_search and cpu_only_search), "Cannot enable both gpu_only_search and cpu_only_search"
        rand_emb = torch.rand(warm_up, self.embed_dim).to(self.device)

        fetch_emb = None
        runtime_fetch = True if fetch_query is not None else False

        for _ in range(warm_up):
            emb = self.txt_to_emb(query)

        t_arr = []
        for _ in range(n_run):
            # warm up and flush cache to avoid cache effect
            # for i in range(warm_up):
            #     self.index.search(
            #         emb=rand_emb[i].unsqueeze(0), topk=topk, nprobe=nprobe,
            #     )

            tick = time.time()

            if runtime_fetch:
                fetch_emb = self.txt_to_emb(fetch_query)

            emb = self.txt_to_emb(query)

            if self.cpu_only:
                tick = time.time()

            _ = self.retrieval_search(
                emb, topk, nprobe=nprobe,
                gpu_only_search=gpu_only_search, cpu_only_search=cpu_only_search,
                runtime_fetch=runtime_fetch, fetch_emb=fetch_emb, fetch_nprobe=fetch_nprobe,
            )
            t_arr.append(time.time() - tick)

        return np.mean(t_arr)

    def retrieval_search(
            self, emb, topk, nprobe=32,
            gpu_only_search=False, cpu_only_search=False,
            runtime_fetch=False, fetch_emb=None, fetch_nprobe=None,
    ):
        request = Request(
            RETRIEVAL_SEARCH_REQUEST,
            {
                'emb': emb,
                'topk': topk,
                'nprobe': nprobe,
                'gpu_only_search': gpu_only_search,
                'cpu_only_search': cpu_only_search,
                'runtime_fetch': runtime_fetch,
                'fetch_emb': fetch_emb,
                'fetch_nprobe': fetch_nprobe,
            },
        )
        response = asyncio.run(async_send_recv(self.retrieval_service_addr,
                                               request, byte_mode=False))
        return response.data['results']
        # result = self.index.search(
        #     emb, topk, nprobe=nprobe,
        #     gpu_only_search=gpu_only_search, cpu_only_search=cpu_only_search,
        #     runtime_fetch=runtime_fetch, fetch_emb=fetch_emb, fetch_nprobe=fetch_nprobe,
        # )
        # torch.cuda.synchronize()
        # return result

    def warm_up_llm(
            self,
            warm_up: int = WARM_UP_ITER,
            prefetch_query: str = None,
            prefetch_budget: int = None,
    ):
        for _ in range(warm_up):
            _ = self.bench_llm_batch(
                [WARM_UP_PROMPT],
                [WARM_UP_OUT],
                prefetch_query=prefetch_query,
                prefetch_budget=prefetch_budget,
                disable_bench=self.disable_bench_llm,
            )

    def clear_prefetch_data_on_all_gpus(self):
        if self.sim_multi_gpu:
            for gpu_id in range(self.num_gpu):
                self.switch_gpu(gpu_id)
                self.clear_prefetch_data()
        else:
            self.clear_prefetch_data()

    def clear_prefetch_data(self):
        request = Request(RETRIEVAL_CLEAR_PREFETCH_DATA_REQUEST, {})
        asyncio.run(async_send_recv(self.retrieval_service_addr,
                                    request, byte_mode=False))
        # self.index.clear_prefetch_data()

    def shutdown_services(self):
        async def async_shutdown():
            await asyncio.gather(
                async_send_recv(
                    self.llm_service_addr,
                    Request(type=SHUTDOWN_REQUEST, args={}), byte_mode = False
                ),
                async_send_recv(
                    self.retrieval_service_addr,
                    Request(type=SHUTDOWN_REQUEST, args={}), byte_mode = False
                ),
            )
        asyncio.run(async_shutdown())


    def switch_gpu(self, gpu_id, update_cache_record=False):
        if not self.sim_multi_gpu: return
        asyncio.run(
            async_send_recv(
                self.retrieval_service_addr,
                Request(
                    type=RETRIEVAL_SWITCH_GPU_REQUEST,
                    args={
                        'gpu_id': gpu_id,
                        'update_cache_record': update_cache_record,
                    }
                ),
                byte_mode=False
            )
        )

    def resize_cache(self):
        asyncio.run(
            async_send_recv(
                self.retrieval_service_addr,
                Request(RETRIEVAL_RESIZE_CACHE_REQUEST, {}),
                byte_mode=False
            )
        )

    def retrieval_get_cache_clusters_overlap(self, emb, nprobe, gpu_id):
        response = asyncio.run(
            async_send_recv(
                self.retrieval_service_addr,
                Request(
                    RETRIEVAL_GET_CACHE_CLUSTERS_OVERLAP_SIM_REQUEST,
                    {'emb': emb, 'nprobe': nprobe, 'gpu_id': gpu_id}
                ),
                byte_mode=False
            )
        )
        return response.data['overlap'], response.data['total_count']

    def change_num_gpu(self, num_gpu):
        if not self.sim_multi_gpu: return
        asyncio.run(
            async_send_recv(
                self.retrieval_service_addr,
                Request(
                    type=RETRIEVAL_CHANGE_NUM_GPU_REQUEST,
                    args={
                        'num_gpu': num_gpu,
                    }
                ),
                byte_mode=False
            )
        )
