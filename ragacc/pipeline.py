import asyncio

from .prompt_templates import *
from typing import Optional, Tuple
import json
import numpy as np
import os
import time
import torch
import tqdm
from .schedule import greedy_batch_requests, naive_batch_requests, greedy_grouping_mini_batch
from .pipeline_budgets import PREFETCH_BUDGET_DICT_SMALL, PREFETCH_BUDGET_DICT_LARGE, PREFETCH_BUDGET_DICT_22B
from .const import (
    MAX_ITER, WARM_UP_ITER, SMALL_NUMBER,
    data_path_template, data_path_template_selfrag, data_path_template_flare,
)
from .services import (
    Request, Reply, service_manager,
    RAG_PIPELINE_EVALUATION_REQUEST, RAG_WARM_UP_REQUEST,
    RAG_TXT_TO_EMB_REQUEST, RETRIEVAL_FIND_CLUSTERS_REQUEST,
    RETRIEVAL_GET_CACHE_CLUSTERS_OVERLAP_SIM_REQUEST,
    RETRIEVAL_GET_CACHE_CLUSTERS_OVERLAP_REQUEST,
    RAG_CLEAR_ALL_PREFETCH_DATA_REQUEST,
    RETRIEVAL_SWITCH_GPU_REQUEST, RETRIEVAL_RESIZE_CACHE_REQUEST,
    RETRIEVAL_CHANGE_NUM_GPU_REQUEST, RAG_UPDATE_NPROBE_REQUEST,
)
from .zmq_utils import async_send_recv


async def pipeline_evaluation_request(
        service_id: int, pipeline: str, data,
        use_cluster_prefetch: bool, prefetch_budget: int,
):
    request = Request(
        RAG_PIPELINE_EVALUATION_REQUEST,
        {
            'pipeline': pipeline,
            'data': data,
            'use_cluster_prefetch': use_cluster_prefetch,
            'prefetch_budget': prefetch_budget,
        }
    )
    reply: Reply = await async_send_recv(
        service_manager.get_service_address('rag', service_id),
        request, byte_mode=False,
    )
    return reply.data['result']

async def warm_up_all_llm_services(prefetch_query, prefetch_budget):
    """Warm up all LLM services to avoid cold start."""
    tasks = []
    for addr in service_manager.get_all_service_addresses('rag'):
        request = Request(
            RAG_WARM_UP_REQUEST,
            {
                'warm_up': WARM_UP_ITER,
                'prefetch_query': prefetch_query,
                'prefetch_budget': prefetch_budget,
            }
        )
        tasks.append(async_send_recv(addr, request, byte_mode=False))
    await asyncio.gather(*tasks)

def warm_up_all_llm_services_sync(prefetch_query, prefetch_budget):
    """Warm up all LLM services to avoid cold start."""
    asyncio.run(warm_up_all_llm_services(prefetch_query, prefetch_budget))

async def txt_to_emb(txt: str | list[str], service_id: Optional[int] = None):
    """Convert text to embedding using the specified service."""
    request = Request(RAG_TXT_TO_EMB_REQUEST, {'txt': txt,})
    # If service_id is None, use the default service ID 0
    service_id = service_id if service_id is not None else 0
    reply: Reply = await async_send_recv(
        service_manager.get_service_address('rag', service_id),
        request, byte_mode=False,
    )
    return reply.data['emb']

def txt_to_emb_sync(txt: str | list[str], service_id: Optional[int] = None):
    """Convert text to embedding using the specified service."""
    return asyncio.run(txt_to_emb(txt, service_id))

async def find_clusters(emb, nprobe, service_id: Optional[int] = None):
    """Find clusters for the given embedding using the specified retrieval service.

    Note: Here we directly contact the retrieval service instead of going through
    the RAG service to avoid unnecessary overhead.
    """
    request = Request(
        RETRIEVAL_FIND_CLUSTERS_REQUEST, {'emb': emb, 'nprobe': nprobe,}
    )
    service_id = service_id if service_id is not None else 0
    reply: Reply = await async_send_recv(
        service_manager.get_service_address('retrieval', service_id),
        request, byte_mode=False,
    )
    return reply.data['clusters']

def find_clusters_sync(emb, nprobe, service_id: Optional[int] = None):
    """Find clusters for the given embedding using the specified retrieval service."""
    return asyncio.run(find_clusters(emb, nprobe, service_id))

def load_eval_data(args, pipeline=None):
    pipeline_type = args.pipeline_type if pipeline is None else pipeline

    if pipeline_type == "selfrag":
        template = data_path_template_selfrag
    elif pipeline_type == "flare":
        template = data_path_template_flare
    else:
        template = data_path_template

    profile_path = template.format(
        data_dir=args.data_dir, pipeline=pipeline_type, dataset=args.datasets, topk=args.topk
    )

    with open(profile_path, 'r') as f:
        print(f"Loading data from: {profile_path}")
        input_data = json.load(f)

    return input_data

def check_cluster_prefetch(args):
    if args.index_type == "faiss" or args.disable_prefetch:
        return False
    else:
        return True

async def retrieval_get_cache_clusters_overlap(
        emb, nprobe, gpu_id: int, simulation: bool
) -> Tuple[int, int]:
    """Get the overlap clusters for the given embedding and nprobe.

    Return:
        - overlap: a int of the number of overlapping clusters
        - total_count: a int of the total number of clusters in the batch
    """
    if simulation:
        request = Request(
            RETRIEVAL_GET_CACHE_CLUSTERS_OVERLAP_SIM_REQUEST,
            {'emb': emb, 'nprobe': nprobe, 'gpu_id': gpu_id,}
        )
    else:
        request = Request(
            RETRIEVAL_GET_CACHE_CLUSTERS_OVERLAP_REQUEST,
            {'emb': emb, 'nprobe': nprobe,}
        )
    service_id = 0 if simulation else gpu_id
    reply: Reply = await async_send_recv(
        service_manager.get_service_address('retrieval', service_id),
        request, byte_mode=False,
    )
    return reply.data['overlap'], reply.data['total_count']

def retrieval_get_cache_clusters_overlap_sync(
        emb, nprobe, gpu_id: int, simulation: bool
) -> Tuple[int, int]:
    """Get the overlap clusters for the given embedding and nprobe."""
    return asyncio.run(
        retrieval_get_cache_clusters_overlap(emb, nprobe, gpu_id, simulation)
    )

async def clear_all_prefetch_data():
    """Clear the prefetch data on all GPUs (virtual and physical)."""
    tasks = []
    rag_services = service_manager.get_all_service_addresses('rag')
    for addr in rag_services:
        request = Request(RAG_CLEAR_ALL_PREFETCH_DATA_REQUEST, {})
        tasks.append(async_send_recv(addr, request, byte_mode=False))
    await asyncio.gather(*tasks)

def clear_all_prefetch_data_sync():
    """Clear the prefetch data on all GPUs (virtual and physical)."""
    asyncio.run(clear_all_prefetch_data())

async def update_nprobe_all_retrieval_services(nprobe: int):
    """Update the nprobe for all retrieval services."""
    tasks = []
    retrieval_services = \
        service_manager.get_all_service_addresses('rag')
    for addr in retrieval_services:
        request = Request(
            RAG_UPDATE_NPROBE_REQUEST,
            {'nprobe': nprobe},
        )
        tasks.append(async_send_recv(addr, request, byte_mode=False))
    await asyncio.gather(*tasks)

def update_nprobe_all_retrieval_services_sync(nprobe: int):
    """Update the nprobe for all retrieval services."""
    asyncio.run(update_nprobe_all_retrieval_services(nprobe))

async def change_num_gpu(num_gpu: int):
    """Change the number of GPUs used by the RAG service. Note: This only
    applies to the simulation mode!"""
    request = Request(RETRIEVAL_CHANGE_NUM_GPU_REQUEST, {'num_gpu': num_gpu})
    return await async_send_recv(
        service_manager.get_service_address('retrieval', 0),
        request, byte_mode=False,
    )

def change_num_gpu_sync(num_gpu: int):
    """Change the number of GPUs used by the RAG service. Note: This only
    applies to the simulation mode!"""
    return asyncio.run(change_num_gpu(num_gpu))

async def switch_gpu(gpu_id: int, update_cache_request: bool = False):
    """Switch the GPU used by the Retrieval Service. Note: This only applies
    to the simulation mode!"""
    request = Request(
        RETRIEVAL_SWITCH_GPU_REQUEST,
        {'gpu_id': gpu_id, 'update_cache_request': update_cache_request}
    )
    return await async_send_recv(
        service_manager.get_service_address('retrieval', 0),
        request, byte_mode=False,
    )

def switch_gpu_sync(gpu_id: int, update_cache_request: bool = False):
    """Switch the GPU used by the Retrieval Service. Note: This only applies
    to the simulation mode!"""
    return asyncio.run(switch_gpu(gpu_id, update_cache_request))

async def resize_cache(gpu_id: Optional[int]):
    """Resize the cache for the given GPU id. Note: in simulation mode, this
    can only resize the gpu that is currently being used."""
    request = Request(RETRIEVAL_RESIZE_CACHE_REQUEST, {})
    service_id = 0 if gpu_id is None else gpu_id
    return await async_send_recv(
        service_manager.get_service_address('retrieval', service_id),
        request, byte_mode=False,
    )

async def change_all_cache_fraction(fraction: float):
    """Change the cache fraction for all GPUs."""
    retrieval_services = \
        service_manager.get_all_service_addresses('retrieval')
    tasks = [
        async_send_recv(
            addr,
            Request(RETRIEVAL_RESIZE_CACHE_REQUEST, {'fraction': fraction}),
            byte_mode=False,
        ) for addr in retrieval_services
    ]
    return await asyncio.gather(*tasks)

def change_all_cache_fraction_sync(fraction: float):
    """Change the cache fraction for all GPUs."""
    return asyncio.run(change_all_cache_fraction(fraction))


class Pipeline:
    def __init__(self):
        self.sim_multi_gpu = None

    def evaluate(self, args, budget=None, enable_spec=False, batch_size=1, mini_batch=1, num_gpu=1):
        self.sim_multi_gpu = args.sim_multi_gpu
        if self.sim_multi_gpu:
            change_num_gpu_sync(num_gpu)
        if budget is None:
            if args.budget_type == "small":
                budget_dict = PREFETCH_BUDGET_DICT_SMALL
            elif args.budget_type == "22b":
                budget_dict = PREFETCH_BUDGET_DICT_22B
            else:
                budget_dict = PREFETCH_BUDGET_DICT_LARGE
            budget = budget_dict[args.datasets][args.gpu_model][args.pipeline_type]

        print("Batch size: ", batch_size)
        print("Mini batch size: ", mini_batch)
        print("Number of GPUs: ", num_gpu)
        self.eval_one_pipeline(
            args, budget, enable_spec=enable_spec,
            batch_size=batch_size, mini_batch=mini_batch,
            num_gpu=num_gpu
        )

    def schedule(self, n_data, args, input_data, batch_size=1, start_idx=0):
        if args.index_type == "faiss":
            assert args.batch_strategy == "naive", "Batch strategy for faiss index should be naive"

        if args.batch_strategy == "greedy":
            query_clusters = []
            for i in range(start_idx, n_data + start_idx):
                query = input_data[i]['question']
                emb = txt_to_emb_sync(query)
                query_clusters.append(find_clusters_sync(emb, nprobe=args.nprobe)[0])

            batch_results = greedy_batch_requests(query_clusters, batch_size, start_idx)
        elif args.batch_strategy == "naive":
            batch_results = naive_batch_requests(n_data, batch_size, start_idx)
        else:
            raise NotImplementedError(f"Batch strategy {args.batch_strategy} not implemented")

        return batch_results
    
    def schedule_mini_batch(self, input_data, args, batch_requests, mini_batch=1):
        emb = None
        if args.index_type == "faiss":
            assert args.mini_batch_strategy == "naive", "Mini-Batch strategy for faiss index should be naive"

        if args.mini_batch_strategy == "naive":
            mini_batch_requests = []
            for i in range(0, len(batch_requests), mini_batch):
                mini_batch_requests.append(batch_requests[i:i + mini_batch])
        elif args.mini_batch_strategy == "greedy":
            mini_batch_query = []
            for i in batch_requests:
                mini_batch_query.append(
                    input_data[i]['question']
                )
            emb = txt_to_emb_sync(mini_batch_query)
            mini_batch_requests = greedy_grouping_mini_batch(batch_requests, emb, mini_batch)
        else:
            raise NotImplementedError(f"Mini batch strategy {args.mini_batch_strategy} not implemented")
        
        return mini_batch_requests, emb

    def schedule_gpu_cache(self, emb, mini_batch_requests,
                           all_requests, args, num_gpu=1,
                           profile_overlap=False):
        """Schedule the batches according to the GPU cache simulation. This
        function works best when len(mini_batch_requests) == num_gpu.
        """
        if args.index_type == "faiss":
            assert args.mini_batch_strategy == "naive", "Mini-Batch strategy for faiss index should be naive"
        if args.mini_batch_strategy == "greedy":
            get_overlap_cluster_time = 0.0
            overlap_count = torch.empty(0, dtype=torch.int32, device=emb.device)
            total_cluster_in_a_batch = []
            for id, batch in enumerate(mini_batch_requests):
                # Put embeddings in the same order as mini_batch_requests
                batch_emb = emb[torch.isin(torch.tensor(all_requests), torch.tensor(batch))]

                # Get the overlap cluster data
                get_overlap_cluster_time_one_gpu = []
                overlap_count_one_batch = []
                if self.sim_multi_gpu:
                    for gpu_id in range(num_gpu):
                        start_time = time.time()
                        overlap_count_one, total_count = \
                            retrieval_get_cache_clusters_overlap(
                                batch_emb, args.nprobe, gpu_id, True
                            )
                        overlap_count_one_batch.append(overlap_count_one)
                        total_cluster_in_a_batch.append(total_count)
                        get_overlap_cluster_time_one_gpu.append(time.time() - start_time)
                    get_overlap_cluster_time += max(get_overlap_cluster_time_one_gpu)
                else:
                    async def get_all_overlaps():
                        tasks = [
                            retrieval_get_cache_clusters_overlap(
                                batch_emb, args.nprobe, gpu_id, False
                            ) for gpu_id in range(num_gpu)
                        ]
                        return await asyncio.gather(*tasks)
                    start_time = time.time()
                    overlap_results = asyncio.run(get_all_overlaps())
                    get_overlap_cluster_time += time.time() - start_time
                    overlap_count_one_batch = [res[0] for res in overlap_results]
                    total_cluster_in_a_batch.extend([res[1] for res in overlap_results])
                overlap_count = torch.cat(
                    [overlap_count,
                     torch.tensor(overlap_count_one_batch).to(emb.device)],
                    dim=0,
                )

            # Schedule the mini-batch requests
            # FIXME (Yiyu):
            #  When the number of mini batch is not divisible by num_gpu,
            #  There will be some problem with this algorithm since some
            #  gpu will have more requests than others by 2 or more.
            #  If you are going to fix this, remember to also change the
            #  return format to be a list of lists.
            schedule_start_time = time.time()
            big_batch = len(mini_batch_requests) // num_gpu
            _, sorted_idx = torch.sort(overlap_count, descending=True)
            scheduled = [False] * len(mini_batch_requests)
            gpu_schedule_count = [0] * num_gpu
            gpu_schedule_quota = len(mini_batch_requests) // num_gpu
            schedule_results = [[0] * big_batch] * num_gpu
            overlap_profile = []
            for i in sorted_idx:
                request_id = i // num_gpu
                gpu_id = i % num_gpu
                if scheduled[request_id]: continue
                if gpu_schedule_count[gpu_id] >= gpu_schedule_quota: continue
                # Start batching
                schedule_results[gpu_id][gpu_schedule_count[gpu_id]] = request_id
                scheduled[request_id] = True
                gpu_schedule_count[gpu_id] += 1
                if profile_overlap:
                    overlap_profile.append(overlap_count[i].item())
            # check if all requests are scheduled
            for i in scheduled:
                assert i, "Not all requests are scheduled"
            # Package the results
            schedule_final_results = []
            for i in range(big_batch):
                for j in range(num_gpu):
                    schedule_final_results.append(mini_batch_requests[schedule_results[j][i]])
            schedule_time = time.time() - schedule_start_time
            total_time = get_overlap_cluster_time + schedule_time

        else:
            raise NotImplementedError(f"Mini batch strategy {args.mini_batch_strategy} not implemented")

        if profile_overlap:
            return schedule_final_results, total_time, np.mean(overlap_profile), np.mean(total_cluster_in_a_batch)
        return schedule_final_results, total_time
    
    def construct_mini_batch(self, input_data, args, batch_requests,
                             mini_batch=1, num_gpu=1, profile_overlap=False):
        overlap_profile = 0
        total_clusters_in_a_batch = 0
        if mini_batch == len(batch_requests):
            mini_batch_requests = [batch_requests]
            mini_batch_t = 0.0
        else:
            if mini_batch == 1:
                mini_batch_requests = [[request] for request in batch_requests]
                t0 = time.time()
                emb = txt_to_emb_sync([input_data[i]['question'] for i in batch_requests])
                mini_batch_t = time.time() - t0
            else:
                assert len(batch_requests) % mini_batch == 0
                # warm up gpu for better time measurement
                for _ in range(3):
                    self.schedule_mini_batch(input_data, args, batch_requests, mini_batch=mini_batch)

                t0 = time.time()
                mini_batch_requests, emb = self.schedule_mini_batch(
                    input_data, args, batch_requests, mini_batch=mini_batch,
                )
                mini_batch_t = time.time() - t0
            if (args.cache_fraction > SMALL_NUMBER and
                    args.mini_batch_strategy == "greedy" and
                    args.no_cache_schedule is False):
                if args.multi_gpu:
                    if profile_overlap:
                        mini_batch_requests, schedule_cache_sim_time, \
                            overlap_profile, total_clusters_in_a_batch = \
                            self.schedule_gpu_cache(
                                emb, mini_batch_requests, batch_requests, args,
                                num_gpu=num_gpu,
                                profile_overlap=profile_overlap,
                            )
                    else:
                        mini_batch_requests, schedule_cache_sim_time = \
                            self.schedule_gpu_cache(
                                emb, mini_batch_requests, batch_requests, args,
                                num_gpu=num_gpu,
                        )
                    mini_batch_t += schedule_cache_sim_time
                else:
                    raise NotImplementedError("Multi-GPU simulation is not supported for non-simulated GPUs")

        if profile_overlap:
            return mini_batch_requests, mini_batch_t, overlap_profile, total_clusters_in_a_batch
        return mini_batch_requests, mini_batch_t

    def profile(self, args, batch_size, mini_batch_size, num_gpu,
                mini_batch_t, pre_llm_t, ret_t, post_llm_t, tot_t,
                profile_cache: bool, overlap_clusters, total_clusters):
        os.makedirs(args.log_dir, exist_ok=True)
        index_type = args.index_type
        if index_type == 'ragacc':
            if args.batch_strategy == 'greedy':
                index_type += '_global_greedy'
            if args.mini_batch_strategy == 'greedy':
                index_type += '_mini_greedy'
            if profile_cache:
                index_type += '_profile_cache'

        log_name = f"{args.datasets}_{index_type}_topk_{args.topk}_ndata_{args.num_samples}.csv"
        log_path = os.path.join(args.log_dir, log_name)
        if not os.path.isfile(log_path):
            print(f"Creating log file: {log_path}")
            with open(log_path, 'w') as f:
                if profile_cache:
                    f.write("Pipeline, Nprobe, Global-Batch, Mini-Batch, Num-GPU, Cache-Frac, "
                            "Mini-Batch-Time, Pre-Ret-LLM, Retrieval, Post-Ret-LLM, Total, "
                            "Overlap-Clusters, Total-Clusters\n")
                else:
                    f.write("Pipeline, Nprobe, Global-Batch, Mini-Batch, Num-GPU, Cache-Frac, "
                            "Mini-Batch-Time, Pre-Ret-LLM, Retrieval, Post-Ret-LLM, Total\n")

        with open(log_path, 'a') as f:
            print(f"Logging to {log_path}")
            if profile_cache:
                f.write(f"{args.pipeline_type}, {args.nprobe}, {batch_size}, "
                        f"{mini_batch_size}, {num_gpu}, {args.cache_fraction}, "
                        f"{mini_batch_t}, {pre_llm_t}, {ret_t}, {post_llm_t}, "
                        f"{tot_t}, {overlap_clusters}, {total_clusters}\n")
            else:
                f.write(f"{args.pipeline_type}, {args.nprobe}, {batch_size}, "
                        f"{mini_batch_size}, {num_gpu}, {args.cache_fraction}, "
                        f"{mini_batch_t}, {pre_llm_t}, {ret_t}, {post_llm_t}, {tot_t}\n")

    def eval_one_pipeline(self, args, prefetch_budget, enable_spec=False,
                          batch_size=1, mini_batch=1, num_gpu=1):
        """Evaluate a specific pipeline.

        The function loads the data, limits the number of requests and
        controls the evaluation flow.

        `eval` calls this function, and this function will do the following
        steps in the evaluation flow:
        - Warm up the llm services
        - Load the evaluation data to cache with the requests other than the
          requests used in the real evaluation
        - Conduct real evaluation with the requests
        """
        pipeline = args.pipeline_type
        print(f"Evaluating Pipeline: {pipeline}")
        use_cluster_prefetch = check_cluster_prefetch(args)
        if pipeline == "linear":
            input_data = load_eval_data(args, pipeline="hyde")
        elif pipeline == "iterative":
            input_data = load_eval_data(args, pipeline="llamaindex_iter")
        elif pipeline == "iterretgen":
            input_data = load_eval_data(args, pipeline="iterretgen")
        elif pipeline == "parallel":
            input_data = load_eval_data(args, pipeline="llamaindex_subquestion")
        elif pipeline == "selfrag":
            input_data = load_eval_data(args, pipeline="selfrag")
        elif pipeline == "flare":
            input_data = load_eval_data(args, pipeline="flare")
        else:
            raise NotImplementedError(f"Pipeline {pipeline} not implemented")

        n_data = len(input_data)
        if args.num_samples > 0:
            n_data = min(n_data, args.num_samples)

        # Set up cache fraction here
        change_all_cache_fraction_sync(args.cache_fraction)

        # Update the nprobe in retrieval services
        update_nprobe_all_retrieval_services_sync(args.nprobe)

        # Evaluation flow starts here
        warm_up_all_llm_services_sync(
            prefetch_query=WARM_UP_QUERY if use_cluster_prefetch else None,
            prefetch_budget=prefetch_budget,
        )

        # Clear all clusters on GPUs
        clear_all_prefetch_data_sync()

        # Load to cache
        if args.cache_fraction > SMALL_NUMBER:
            print("Loading data to cache...")
            self.global_schedule_and_eval(
                pipeline, n_data, args, input_data, batch_size, n_data,
                use_cluster_prefetch, prefetch_budget, mini_batch, num_gpu,
                enable_spec,
            )

        # Real evaluation
        print("Starting real evaluation...")
        avg_time, overlap_list, total_list = self.global_schedule_and_eval(
            pipeline, n_data, args, input_data, batch_size, 0,
            use_cluster_prefetch, prefetch_budget, mini_batch, num_gpu,
            enable_spec, args.profile_cache,
        )
        if args.profile_cache:
            print(f"Average overlap: {np.mean(overlap_list):.2f}, "
                  f"Average total clusters in a batch: {np.mean(total_list):.2f}")
        print("Average time: Batch Time, pre-LLM, Retrieval, post-LLM, end-to-end")
        print(f"{avg_time[0]:.4f}, {avg_time[1]:.4f}, {avg_time[2]:.4f}, {avg_time[3]:.4f}, {avg_time[5]:.4f}")
        if args.profile is True:
            overlap_clusters = np.mean(overlap_list)
            total_clusters = np.mean(total_list)
            self.profile(args, batch_size, mini_batch, num_gpu,
                         avg_time[0], avg_time[1], avg_time[2], avg_time[3], avg_time[5],
                         args.profile_cache, overlap_clusters, total_clusters)

    def global_schedule_and_eval(
            self, pipeline, n_data, args, input_data, batch_size: int,
            start_idx: int, use_cluster_prefetch: bool, prefetch_budget: int,
            mini_batch=1, num_gpu=1, enable_spec=False, profile_cache=False
    ):
        """This function is used to schedule the global requests and evaluate
        the pipeline. It is used for the global scheduling of the requests in
        the pipeline.

        FIXME: what kind of data do we need for the breakdown? Maximal one? Average one?

        Returns:
            - a numpy array of the following data:
                - time of mini-batch construction
                - time of pre-retrieval LLM
                - time of retrieval
                - time of post-retrieval LLM
                - time of one pipeline on one GPU
                - end-to-end time
            - overlap count list if profile_cache is True, else []
            - total clusters list if profile_cache is True, else []
        """
        batch_results = self.schedule(n_data, args, input_data,
                                      batch_size, start_idx)

        # Search clusters for each query
        rag_t = []
        overlap_list = []
        total_clusters_list = []
        for batch_group in tqdm.tqdm(batch_results):
            time_vector, overlap, total_clusters = self.eval_multi_gpu(
                pipeline, batch_group, input_data, args,
                use_cluster_prefetch, prefetch_budget,
                mini_batch, num_gpu, enable_spec, profile_cache
            )
            rag_t.append(time_vector)
            if profile_cache:
                overlap_list.append(overlap)
                total_clusters_list.append(total_clusters)
        return np.mean(rag_t, axis=0), overlap_list, total_clusters_list

    def eval_multi_gpu(
            self, pipeline, batch_requests: list, input_data, args,
            use_cluster_prefetch: bool, prefetch_budget: int,
            mini_batch=1, num_gpu=1, enable_spec=False, profile_cache=False,
    ):
        """Evaluate the pipeline with multiple GPUs.

        Returns:
            - a numpy array of the following data:
                - time of mini-batch construction
                - time of pre-retrieval LLM
                - time of retrieval
                - time of post-retrieval LLM
                - time of one pipeline on one GPU
                - end-to-end time
            - overlap count if profile_cache is True, else None
            - total clusters if profile_cache is True, else None
        """
        overlap_count = None
        total_clusters = None
        if profile_cache:
            assert args.index_type == "ragacc", \
                "Cache performance evaluation only works with ragacc index"
            mini_batch_requests, mini_batch_t, overlap_count, \
                total_clusters = self.construct_mini_batch(
                input_data, args, batch_requests,
                mini_batch=mini_batch, num_gpu=num_gpu,
                profile_overlap=True,
            )
        else:
            mini_batch_requests, mini_batch_t = self.construct_mini_batch(
                input_data, args, batch_requests,
                mini_batch=mini_batch, num_gpu=num_gpu,
            )

        num_mini_batch = len(mini_batch_requests)
        assert num_mini_batch % num_gpu == 0, f"Number of mini-batch {num_mini_batch} is not divisible by number of GPUs {num_gpu}"

        if pipeline == "linear":
            prepare = self.prepare_for_linear_pipeline
        elif pipeline == "iterative":
            prepare = self.prepare_for_iterative_pipeline
        elif pipeline == "iterretgen":
            prepare = self.prepare_for_iterretgen_pipeline
        elif pipeline == "parallel":
            prepare = self.prepare_for_parallel_pipeline
        elif pipeline == "selfrag":
            prepare = self.prepare_for_selfrag_pipeline
        elif pipeline == "flare":
            prepare = self.prepare_for_flare_pipeline
        else:
            raise NotImplementedError(f"Pipeline type {args.pipeline_type} not implemented")

        each_gpu_time = np.array([0.0, 0.0, 0.0, 0.0])
        end_to_end_total_time = mini_batch_t
        for mini_batch_i in range(0, num_mini_batch, num_gpu):
            multi_gpu_t = []
            if self.sim_multi_gpu:
                for gpu_i in range(num_gpu):
                    switch_gpu_sync(gpu_i)
                    data = asyncio.run(
                        prepare(gpu_i, args, mini_batch_requests, mini_batch_i,
                                input_data, use_cluster_prefetch, enable_spec)
                    )
                    multi_gpu_t.append(asyncio.run(
                        pipeline_evaluation_request(
                            gpu_i, pipeline, data,
                            use_cluster_prefetch, prefetch_budget,
                        )
                    ))
            else:
                # launch the pipeline
                async def pipeline_async():
                    # preparation phase
                    preparation_tasks = [
                        prepare(
                            gpu_id, args, mini_batch_requests, mini_batch_i,
                            input_data, use_cluster_prefetch, enable_spec,
                        )
                        for gpu_id in range(num_gpu)
                    ]
                    prepared_data = await asyncio.gather(*preparation_tasks)

                    # processing phase
                    tick = time.time()
                    tasks = [
                        pipeline_evaluation_request(
                            gpu_id, pipeline, prepared_data[gpu_id],
                            use_cluster_prefetch, prefetch_budget,
                        ) for gpu_id in range(num_gpu)
                    ]
                    multi_gpu_time = await asyncio.gather(*tasks)
                    tock = time.time()
                    end_to_end_time = tock - tick
                    return multi_gpu_time, end_to_end_time
                multi_gpu_t, end_to_end_time = asyncio.run(pipeline_async())
                end_to_end_total_time += end_to_end_time
            multi_gpu_t = np.array(multi_gpu_t)
            slowest_batch_idx = np.argmax(multi_gpu_t[:,-1], axis=0)
            if self.sim_multi_gpu:
                end_to_end_total_time += multi_gpu_t[slowest_batch_idx][-1]
            each_gpu_time += multi_gpu_t[slowest_batch_idx]
        return np.array([
            mini_batch_t, each_gpu_time[0], each_gpu_time[1],
            each_gpu_time[2], each_gpu_time[3], end_to_end_total_time,
        ]), overlap_count, total_clusters

    async def prepare_for_linear_pipeline(
            self, gpu_id, args, mini_batch_requests, mini_batch_i,
            input_data, use_cluster_prefetch, enable_spec,
    ):
        mini_batch_group = mini_batch_requests[mini_batch_i + gpu_id]
        if self.sim_multi_gpu:
            gpu_id = None
        await resize_cache(gpu_id)

        origin_queries = [input_data[i]['question'] for i in mini_batch_group]
        data_out = [input_data[i]['output'] for i in mini_batch_group]

        # QueryTransform
        prompt1 = [data['hyde_prompt'] for data in data_out]
        out1 = [data['hyde_gen_query'] for data in data_out]

        prompt2 = [data['prompt'] for data in data_out]
        out2 = [data['pred'] for data in data_out]

        request_data = {
            'origin_queries': origin_queries,
            'prompt1': prompt1,
            'out1': out1,
            'prompt2': prompt2,
            'out2': out2,
            'retrieval_queries': out1,
        }
        if args.prefetch_strategy == "runtime":
            request_data['fetch_queries'] = out1
        return request_data

    async def prepare_for_iterative_pipeline(
            self, gpu_id, args, mini_batch_requests, mini_batch_i,
            input_data, use_cluster_prefetch, enable_spec,
    ):
        # print(f"Processing mini-batch {mini_batch_i+gpu_i} on GPU {gpu_i}")
        mini_batch_group = mini_batch_requests[mini_batch_i + gpu_id]
        if self.sim_multi_gpu:
            gpu_id = None
        await resize_cache(gpu_id=gpu_id)

        query_data = []
        for i_iter in range(MAX_ITER):
            # query_retrieval_original = input_data[i_data]['question']
            origin_query = [input_data[i]['question'] for i in mini_batch_group]

            # QueryTransform
            prompt1 = []
            prompt2 = []
            prompt3 = []
            spec_prompt = []
            out1 = []
            out2 = []
            out3 = []
            spec_out = []
            prefetch_queries = []
            for i in mini_batch_group:
                if f'step_decompose_prompt_iter_{i_iter}' in input_data[i]['output']:
                    prompt1.append([input_data[i]['output'][f'step_decompose_prompt_iter_{i_iter}']])
                    out1.append(input_data[i]['output'][f'step_decomposed_retrieval_query_iter_{i_iter}'])
                    prompt2.append(input_data[i]['output'][f'llm_prompt_iter_{i_iter}'])
                    out2.append(input_data[i]['output'][f'llm_response_iter_{i_iter}'])
                    prompt3.append(input_data[i]['output'][f'judge_prompt_iter_{i_iter}'])
                    out3.append(input_data[i]['output'][f'judge_result_iter_{i_iter}'])
                    if i_iter == 0:
                        prefetch_query = origin_query
                    else:
                        prefetch_query = input_data[i]['output'][f'step_decomposed_retrieval_query_iter_{i_iter-1}']
                    if isinstance(prefetch_query, list):  # Actually, I don't understand why it is a list in practice
                        prefetch_queries.extend(prefetch_query)
                    else:
                        prefetch_queries.append(prefetch_query)
                    if (enable_spec and
                        f'step_decompose_prompt_iter_{i_iter+1}' in input_data[i]['output']):
                        spec_prompt.append(input_data[i]['output'][f'step_decompose_prompt_iter_{i_iter+1}'])
                        spec_out.append(input_data[i]['output'][f'step_decomposed_retrieval_query_iter_{i_iter+1}'])

            if len(prompt1) == 0:
                break

            if use_cluster_prefetch:
                if args.prefetch_strategy == "once" and i_iter > 0:
                    prefetch_queries = None
            else:
                prefetch_queries = None
            query_iter = {
                'origin_queries': origin_query,
                'prompt1': prompt1,
                'out1': out1,
                'prompt2': prompt2,
                'out2': out2,
                'prompt3': prompt3,
                'out3': out3,
                'spec_prompt': spec_prompt,
                'spec_out': spec_out,
            }
            if prefetch_queries is not None:
                query_iter['prefetch_queries'] = prefetch_queries

            query_data.append(query_iter)

        return {
            'enable_spec': enable_spec,
            'query_data': query_data,
        }

    async def prepare_for_iterretgen_pipeline(
            self, gpu_id, args, mini_batch_requests, mini_batch_i,
            input_data, use_cluster_prefetch, enable_spec,
    ):
        # print(f"Processing mini-batch {mini_batch_i+gpu_i} on GPU {gpu_i}")
        mini_batch_group = mini_batch_requests[mini_batch_i + gpu_id]
        if self.sim_multi_gpu:
            gpu_id = None
        await resize_cache(gpu_id=gpu_id)

        query_data = []
        max_iter = MAX_ITER
        for i_iter in range(max_iter):
            retrieval_queries = [input_data[i]['output'][f'retrieval_query_iter_{i_iter}'] for i in mini_batch_group]
            llm_prompt = [input_data[i]['output'][f'prompt_iter_{i_iter}'] for i in mini_batch_group]
            llm_out = [input_data[i]['output'][f'pred_iter_{i_iter}'] for i in mini_batch_group]
            query_data.append({
                'retrieval_queries': retrieval_queries,
                'prompt': llm_prompt,
                'out': llm_out,
            })
        return {
            'max_iter': max_iter,
            'query_data': query_data,
        }


    async def prepare_for_parallel_pipeline(
            self, gpu_id, args, mini_batch_requests, mini_batch_i,
            input_data, use_cluster_prefetch, enable_spec,
    ):
        # print(f"Processing mini-batch {mini_batch_i+gpu_i} on GPU {gpu_i}")
        mini_batch_group = mini_batch_requests[mini_batch_i + gpu_id]
        if self.sim_multi_gpu:
            gpu_id = None
        await resize_cache(gpu_id=gpu_id)

        request_data = {
            'questions': [
                input_data[i]['question'] for i in mini_batch_group
            ],
            'subquestion_prompt': [
                input_data[i]['output']['subquestion_prompt']['content']
                for i in mini_batch_group
            ],
            'gen_subquestions': [
                input_data[i]['output']['raw_subquestions_string']
                for i in mini_batch_group
            ],
            'retrieval_queries': [
                query
                for i in mini_batch_group
                for query in input_data[i]['output']['retrieval_queries']
            ],
            'llm_prompt': [
                input_data[i]['output']['llm_prompt']['content']
                for i in mini_batch_group
            ],
            'llm_gen': [
                input_data[i]['output']['pred'] for i in mini_batch_group
            ],
        }

        if args.prefetch_strategy == "runtime":
            request_data['fetch_queries'] = [
                entry for sublist in request_data['retrieval_queries']
                for entry in sublist
            ]
        return request_data

    async def prepare_for_selfrag_pipeline(
            self, gpu_id, args, mini_batch_requests, mini_batch_i,
            input_data, use_cluster_prefetch, enable_spec,
    ):
        # print(f"Processing mini-batch {mini_batch_i+gpu_i} on GPU {gpu_i}")
        mini_batch_group = mini_batch_requests[mini_batch_i + gpu_id]
        if self.sim_multi_gpu:
            gpu_id = None
        await resize_cache(gpu_id=gpu_id)

        question = [input_data[i]['question'] for i in mini_batch_group]
        retrieval_judge_prompt = [input_data[i]['output']["retrieval_judge_prompt"] for i in mini_batch_group]
        judge_output = [input_data[i]['output']["judge_pred_txt"] for i in mini_batch_group]
        retrieval_flag = [input_data[i]['output']["retrieval_flag"] for i in mini_batch_group]
        retrieval_questions = []
        llm_prompt_list = []
        llm_output_list = []
        critic_time_list = [0]
        select_time_list = [0]
        postproc_time = max([input_data[i]['output']["postproc_time"] for i in mini_batch_group])

        for index in range(len(retrieval_flag)):  # Iterate through the list of retrieval_flag booleans
            if retrieval_flag[index]:  # If the retrieval is enabled
                retrieval_questions.append(question[index])
                # Extend the llm lists by collecting the respective prompts and outputs for the mini_batch_group
                llm_prompt_list.extend(
                    input_data[mini_batch_group[index]]['output']["llm_prompt"]
                )
                llm_output_list.extend(
                    input_data[mini_batch_group[index]]['output']["critic_result"]
                )
                # Collect the maximum critic_time and select_time for the mini_batch_group
                critic_time_list.append(
                    input_data[mini_batch_group[index]]['output']["critic_time"]
                )
                select_time_list.append(
                    input_data[mini_batch_group[index]]['output']["select_time"]
                )
        critic_time = max(critic_time_list)
        select_time = max(select_time_list)

        return {
            'question': question,
            'retrieval_judge_prompt': retrieval_judge_prompt,
            'judge_output': judge_output,
            'retrieval_flag': retrieval_flag,
            'retrieval_questions': retrieval_questions,
            'llm_prompt_list': llm_prompt_list,
            'llm_output_list': llm_output_list,
            'critic_time': critic_time,
            'select_time': select_time,
            'postproc_time': postproc_time,
        }

    async def prepare_for_flare_pipeline(
            self, gpu_id, args, mini_batch_requests, mini_batch_i,
            input_data, use_cluster_prefetch, enable_spec,
    ):
        # print(f"Processing mini-batch {mini_batch_i+gpu_i} on GPU {gpu_i}")
        mini_batch_group = mini_batch_requests[mini_batch_i + gpu_id]
        if self.sim_multi_gpu:
            gpu_id = None
        await resize_cache(gpu_id=gpu_id)

        question = [input_data[i]['question'] for i in mini_batch_group]
        tot_iter = [(int(input_data[i]["output"]['total_iter']) + 1) for i in mini_batch_group]
        max_iter = max(tot_iter)

        query_data = []
        for i_iter in range(max_iter):
            requests_in_this_round = [mini_batch_group[i] for i, t in enumerate(tot_iter) if t > i_iter]
            forward_prompt = [
                input_data[i]['output'][f'forward_llm_prompt_iter_{i_iter}']['content']
                for i in requests_in_this_round
            ]
            forward_gen = [
                input_data[i]['output'][f'forward_llm_gen_iter_{i_iter}']
                for i in requests_in_this_round
            ]

            question_prompts = []
            gen_queries = []
            llm_prompt = []
            llm_gen = []
            spec_forward_prompt = []
            spec_forward_gen = []
            def get_question_prompts(data):
                return [i['content'] for i in data]
            for i in requests_in_this_round:
                if input_data[i]['output'][f"retrieval_flag_iter_{i_iter}"]:
                    question_prompts.extend(
                        get_question_prompts(input_data[i]['output'][f"quention_gen_prompts_iter_{i_iter}"])
                    )
                    gen_queries.extend(input_data[i]['output'][f"generated_retrieval_queries_iter_{i_iter}"])
                    llm_prompt.append(input_data[i]['output'][f"llm_prompt_iter_{i_iter}"]['content'])
                    llm_gen.append(input_data[i]['output'][f"llm_reponse_iter_{i_iter}"])
                    if enable_spec and i_iter+1 < max_iter:
                        spec_forward_prompt.append(
                            input_data[i]['output'][f'forward_llm_prompt_iter_{i_iter+1}']['content']
                        )
                        spec_forward_gen.append(input_data[i]['output'][f'forward_llm_gen_iter_{i_iter+1}'])
            query_data.append({
                'forward_prompt': forward_prompt,
                'forward_gen': forward_gen,
                'question_prompts': question_prompts,
                'gen_queries': gen_queries,
                'llm_prompt': llm_prompt,
                'llm_gen': llm_gen,
                'spec_forward_prompt': spec_forward_prompt,
                'spec_forward_gen': spec_forward_gen,
            })

        return {
            'enable_spec': enable_spec,
            'query_data': query_data,
            'question': question,
            'tot_iter': tot_iter,
        }

def rag_pipeline_evaluation(
        ragacc, pipeline, args, data,
        use_cluster_prefetch: bool, prefetch_budget: int
):
    """RAG pipeline evaluation.

    Returns a numpy array of the following data:
            - time of pre-retrieval LLM
            - time of retrieval
            - time of post-retrieval LLM
            - total time
    """
    if pipeline == "linear":
        return linear_pipeline_evaluation(
            ragacc, args, data, use_cluster_prefetch, prefetch_budget,
        )
    elif pipeline == "iterative":
        return iterative_pipeline_evaluation(
            ragacc, args, data, use_cluster_prefetch, prefetch_budget,
        )
    elif pipeline == "iterretgen":
        return iterretgen_pipeline_evaluation(
            ragacc, args, data, use_cluster_prefetch, prefetch_budget,
        )
    elif pipeline == "parallel":
        return parallel_pipeline_evaluation(
            ragacc, args, data, use_cluster_prefetch, prefetch_budget,
        )
    elif pipeline == "selfrag":
        return selfrag_pipeline_evaluation(
            ragacc, args, data, use_cluster_prefetch, prefetch_budget,
        )
    elif pipeline == "flare":
        return flare_pipeline_evaluation(
            ragacc, args, data, use_cluster_prefetch, prefetch_budget,
        )
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline}. "
                         f"Supported pipelines: linear, iterative, iterretgen,"
                         f" parallel, selfrag, flare.")


def linear_pipeline_evaluation(
        ragacc, args, data, use_cluster_prefetch: bool, prefetch_budget: int
):
    """
    Linear pipeline

    ┌───────────────┐     ┌────────────┐     ┌─────────────┐
    │ QueryTransform│ ───►│ Retrieval  │ ───►│ Postprocess │ ───►
    └───────────────┘     └────────────┘     └─────────────┘
    """
    prompt1 = data['prompt1']
    out1 = data['out1']
    prompt2 = data['prompt2']
    out2 = data['out2']
    retrieval_queries = data['retrieval_queries']
    fetch_queries = data.get('fetch_queries', None)
    prefetch_queries = data['origin_queries'] if use_cluster_prefetch else None
    llm_t = ragacc.bench_llm_batch(
        prompt1,
        out1,
        prefetch_query=prefetch_queries,
        prefetch_budget=prefetch_budget,
        disable_bench=args.disable_bench_llm,
    )

    ret_t = ragacc.bench_retrieval(
        retrieval_queries, args.topk, args.nprobe,
        disable_bench=args.disable_bench_retrieval,
        gpu_only_search=args.gpu_only_search,
        fetch_query=fetch_queries, fetch_nprobe=args.nprobe,
        # warm_up=0, n_run=1,
    )

    llm2_t = ragacc.bench_llm_batch(
        prompt2,
        out2,
        disable_bench=args.disable_bench_llm,
    )
    return np.array([llm_t, ret_t, llm2_t, np.sum([llm_t, ret_t, llm2_t])])


def iterative_pipeline_evaluation(ragacc, args, data, use_cluster_prefetch: bool, prefetch_budget: int):
    """
    Iterative pipeline evaluation

    ┌───────────────┐     ┌────────────┐     ┌─────────────┐     ┌───────┐
    │ QueryTransform│ ───►│ Retrieval  │ ───►│ Postprocess │ ───►│ Judge │ ───►
    └───────────────┘     └────────────┘     └─────────────┘     └───────┘
        ▲                                                            │
        │                                                            │
        └────────────────────────────────────────────────────────────┘

    """
    enable_spec = data['enable_spec']
    query_data = data['query_data']
    spec_cnt = 0
    iter_t = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # batch, llm, retrieval, llm2, judge, sum
    for i_iter in range(len(query_data)):
        query_data_iter = query_data[i_iter]

        origin_query = query_data_iter['origin_queries']
        prompt1 = query_data_iter['prompt1']
        out1 = query_data_iter['out1']
        prompt2 = query_data_iter['prompt2']
        out2 = query_data_iter['out2']
        prompt3 = query_data_iter['prompt3']
        out3 = query_data_iter['out3']
        spec_prompt = query_data_iter['spec_prompt']
        spec_out = query_data_iter['spec_out']
        prefetch_queries = query_data_iter.get('prefetch_queries', None)

        # QueryTransform
        if use_cluster_prefetch:
            if args.prefetch_strategy == "all":
                ragacc.resize_cache()

        # step decompose
        # if self.enable_spec_branch and i_iter > 0:
        if enable_spec and i_iter > 0:
            llm_t = 0.0
        else:
            llm_t = ragacc.bench_llm_batch(
                prompt1,
                out1,
                prefetch_query=prefetch_queries,
                prefetch_budget=prefetch_budget,
                disable_bench=args.disable_bench_llm,
            )

        # Retrieval
        retrieval_query = out1

        if args.prefetch_strategy == "runtime":
            fetch_query = retrieval_query
        else:
            fetch_query = None

        ret_t = ragacc.bench_retrieval(
            retrieval_query, args.topk, args.nprobe,
            disable_bench=args.disable_bench_retrieval,
            gpu_only_search=args.gpu_only_search,
            fetch_query=fetch_query, fetch_nprobe=args.nprobe,
        )

        # Postprocess
        llm2_t = ragacc.bench_llm_batch(
            prompt2,
            out2,
            disable_bench=args.disable_bench_llm,
        )

        # Judge
        if enable_spec and len(spec_prompt) > 0:
            spec_cnt += 1

            if use_cluster_prefetch:
                if args.prefetch_strategy == "once":
                    prefetch_query = None
                else:
                    prefetch_query = retrieval_query

                if args.prefetch_strategy == "all":
                    ragacc.resize_cache()
            else:
                prefetch_query = None

            spec_t = ragacc.bench_llm_batch(
                prompt3 + spec_prompt,
                out3 + spec_out,
                prefetch_query=prefetch_query,
                prefetch_budget=prefetch_budget,
                disable_bench=args.disable_bench_llm,
                )

            # assign judge time to the speculative branch time
            judge_t = spec_t
        else:
            judge_t = ragacc.bench_llm_batch(
                prompt3,
                out3,
                disable_bench=args.disable_bench_llm,
            )

        t_arr = [0, llm_t, ret_t, llm2_t, judge_t]
        t_arr.append(np.sum(t_arr))

        iter_t += np.array(t_arr)
    return np.array([iter_t[1], iter_t[2], iter_t[3] + iter_t[4], iter_t[5]])


def iterretgen_pipeline_evaluation(ragacc, args, data, use_cluster_prefetch: bool, prefetch_budget: int):
    """
    IterRetGen: another iterative pipeline
    """
    # print(f"Processing mini-batch {mini_batch_i+gpu_i} on GPU {gpu_i}")
    max_iter = data['max_iter']
    query_data = data['query_data']
    iter_t = np.array([0.0, 0.0, 0.0, 0.0]) # batch, retrieval, llm, sum
    for i_iter in range(max_iter):
        query_data_iter = query_data[i_iter]
        retrieval_queries = query_data_iter['retrieval_queries']
        llm_prompt = query_data_iter['prompt']
        llm_out = query_data_iter['out']

        if args.prefetch_strategy == "runtime":
            fetch_queries = retrieval_queries
        else:
            fetch_queries = None

        if i_iter == 0 and args.prefetch_strategy != "runtime":
            gpu_only_search = False
        else:
            gpu_only_search = args.gpu_only_search

        ret_t = ragacc.bench_retrieval(
            retrieval_queries, args.topk, args.nprobe,
            disable_bench=args.disable_bench_retrieval,
            gpu_only_search=gpu_only_search,
            fetch_query=fetch_queries, fetch_nprobe=args.nprobe,
        )

        if use_cluster_prefetch and (i_iter < max_iter - 1):
            if args.prefetch_strategy == "once" and i_iter > 0 :
                prefetch_query = None
            else:
                prefetch_query = retrieval_queries

            if args.prefetch_strategy == "all":
                # clear prefetch data before the next prefetching
                ragacc.resize_cache()
        else:
            prefetch_query = None

        # llm generation
        llm_t = ragacc.bench_llm_batch(
            llm_prompt,
            llm_out,
            prefetch_query=prefetch_query,
            prefetch_budget=prefetch_budget,
            disable_bench=args.disable_bench_llm,
        )

        t_arr = [0, llm_t, ret_t, llm_t + ret_t]
        iter_t += np.array(t_arr)
    return np.array([0.0, iter_t[2], iter_t[1], iter_t[3]])


def parallel_pipeline_evaluation(ragacc, args, data, use_cluster_prefetch: bool, prefetch_budget: int):
    """Parallel pipeline evaluation

    ┌───────────────┐     ┌────────────┐     ┌─────────────┐
    │ QueryTransform│ ───►│ Retrieval  │ ───►│ Postprocess │ ───►
    └───────────────┘  │  └────────────┘  │  └─────────────┘
                       │  ┌────────────┐  │
                       └─►│ Retrieval  │ ─┘
                       │  └────────────┘  │
                       │  ┌────────────┐  │
                       └─►│ Retrieval  │ ─┘
                          └────────────┘

    """
    questions = data['questions']
    subquestion_prompt = data['subquestion_prompt']
    gen_subquestions = data['gen_subquestions']
    retrieval_queries = data['retrieval_queries']
    fetch_query = data.get('fetch_queries', None)
    llm_prompt = data['llm_prompt']
    llm_gen = data['llm_gen']

    prefetch_queries = questions if use_cluster_prefetch else None

    llm_t = ragacc.bench_llm_batch(
        subquestion_prompt,
        gen_subquestions,
        prefetch_query=prefetch_queries,
        prefetch_budget=prefetch_budget,
        disable_bench=args.disable_bench_llm,
    )

    ret_t = ragacc.bench_retrieval(
        retrieval_queries, args.topk, args.nprobe,
        disable_bench=args.disable_bench_retrieval,
        gpu_only_search=args.gpu_only_search,
        fetch_query=fetch_query, fetch_nprobe=args.nprobe,
    )

    llm2_t = ragacc.bench_llm_batch(
        llm_prompt,
        llm_gen,
        disable_bench=args.disable_bench_llm,
    )
    return np.array([llm_t, ret_t, llm2_t, np.sum([llm_t, ret_t, llm2_t])])


def selfrag_pipeline_evaluation(ragacc, args, data, use_cluster_prefetch: bool, prefetch_budget: int):
    """
    Self-RAG pipeline evaluation
    (https://arxiv.org/abs/2310.11511)
    """
    # print(f"Processing mini-batch {mini_batch_i+gpu_i} on GPU {gpu_i}")
    question = data['question']
    retrieval_judge_prompt = data['retrieval_judge_prompt']
    judge_output = data['judge_output']
    retrieval_flag = data['retrieval_flag']
    postproc_time = data['postproc_time']
    critic_time = data['critic_time']
    select_time = data['select_time']
    retrieval_questions = data['retrieval_questions']
    llm_prompt_list = data['llm_prompt_list']
    llm_output_list = data['llm_output_list']

    if use_cluster_prefetch:
        prefetch_query = question
    else:
        prefetch_query = None

    llm_t = ragacc.bench_llm_batch(
        retrieval_judge_prompt,
        judge_output,
        prefetch_query=prefetch_query,
        prefetch_budget=prefetch_budget,
        disable_bench=args.disable_bench_llm,
    )

    if any(retrieval_flag):
        if args.prefetch_strategy == "runtime":
            fetch_queries = retrieval_questions
        else:
            fetch_queries = None

        ret_t = ragacc.bench_retrieval(
            retrieval_questions, args.topk, args.nprobe,
            disable_bench=args.disable_bench_retrieval,
            gpu_only_search=args.gpu_only_search,
            fetch_query=fetch_queries, fetch_nprobe=args.nprobe,
        )

        llm2_t = ragacc.bench_llm_batch(
            llm_inputs=llm_prompt_list,
            llm_outputs=llm_output_list,
            disable_bench=args.disable_bench_llm,
        )

    else:
        ret_t = 0.0
        llm2_t = 0.0
        critic_time = 0
        select_time = 0

    misc_t = critic_time + select_time + postproc_time
    return np.array([llm_t, ret_t, llm2_t + misc_t, llm_t + ret_t + llm2_t + misc_t])


def flare_pipeline_evaluation(ragacc, args, data, use_cluster_prefetch: bool, prefetch_budget: int):
    # print(f"Processing mini-batch {mini_batch_i+gpu_i} on GPU {gpu_i}")
    enable_spec = data['enable_spec']
    query_data = data['query_data']

    retrieval_count = 0
    iter_cnt = 0
    spec_cnt = 0
    question = data['question']
    tot_iter = data['tot_iter']
    max_iter = max(tot_iter)
    iter_cnt += max_iter

    iter_t = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # batch, llm1, llm2, retrieval, llm3, sum
    used_spec_branch = False
    for i_iter in range(max_iter):
        query_data_iter = query_data[i_iter]
        forward_prompt = query_data_iter['forward_prompt']
        forward_gen = query_data_iter['forward_gen']
        question_prompts = query_data_iter['question_prompts']
        gen_queries = query_data_iter['gen_queries']
        llm_prompt = query_data_iter['llm_prompt']
        llm_gen = query_data_iter['llm_gen']
        spec_forward_prompt = query_data_iter['spec_forward_prompt']
        spec_forward_gen = query_data_iter['spec_forward_gen']

        if use_cluster_prefetch and i_iter == 0:
            # For flare, only the first iteration will do prefetching, since prefetch query does not change
            prefetch_query = question
        else:
            prefetch_query = None

        have_forward_and_query = len(question_prompts) > 0 and (prefetch_query is not None)

        if used_spec_branch:
            forward_llm_t = 0.0
        elif not have_forward_and_query:
            forward_llm_t = ragacc.bench_llm_batch(
                forward_prompt,
                forward_gen,
                disable_bench=args.disable_bench_llm,
            )

        if len(question_prompts) > 0:
            if prefetch_query is None:
                query_gen_llm_t = ragacc.bench_llm_batch(
                    question_prompts,
                    gen_queries,
                    prefetch_query=prefetch_query,
                    prefetch_budget=prefetch_budget,
                    disable_bench=args.disable_bench_llm,
                )
            else:
                forward_and_query_gen_t = ragacc.bench_llm_batch_multi_round(
                    [forward_prompt, question_prompts],
                    [forward_gen, gen_queries],
                    prefetch_query=prefetch_query,
                    prefetch_budget=prefetch_budget,
                    disable_bench=args.disable_bench_llm,
                )
                forward_llm_t = 0.0
                query_gen_llm_t = forward_and_query_gen_t

            if args.prefetch_strategy == "runtime":
                fetch_query = gen_queries[0]
            else:
                fetch_query = None

            ret_t = ragacc.bench_retrieval(
                gen_queries, args.topk, args.nprobe,
                disable_bench=args.disable_bench_retrieval,
                gpu_only_search=args.gpu_only_search,
                fetch_query=fetch_query, fetch_nprobe=args.nprobe,
            )

            if enable_spec and i_iter+1 < max_iter:
                spec_cnt += 1

                spec_t = ragacc.bench_llm_batch(
                    llm_prompt + spec_forward_gen,
                    llm_gen + spec_forward_gen,
                    disable_bench=args.disable_bench_llm,
                    )

                # assign llm response time to the speculative branch time
                response_llm_t = spec_t
                used_spec_branch = True
            else:
                response_llm_t = ragacc.bench_llm_batch(
                    llm_prompt,
                    llm_gen,
                    disable_bench=args.disable_bench_llm,
                )
        else:
            query_gen_llm_t, ret_t, response_llm_t = 0.0, 0.0, 0.0

        t_arr = [0, forward_llm_t, query_gen_llm_t, ret_t, response_llm_t]
        t_arr.append(np.sum(t_arr))
        iter_t += np.array(t_arr)
    return np.array([iter_t[1] + iter_t[2], iter_t[3], iter_t[4], iter_t[5]])
