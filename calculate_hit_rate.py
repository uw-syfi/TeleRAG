
import argparse
import json
import sys
import numpy as np
import tqdm
import torch
from transformers import BertModel, AutoTokenizer
from typing import Optional, Any, List, Union

from ragacc.arguments import add_args_for_ragacc
from ragacc.index import RAGAccIndex
from ragacc.pipeline import load_eval_data
from ragacc.pipeline_budgets import PREFETCH_BUDGET_DICT_SMALL, PREFETCH_BUDGET_DICT_LARGE, PREFETCH_BUDGET_DICT_22B


def get_args():
    """Creates an argument parser and adds arguments for the script."""
    parser = argparse.ArgumentParser(description="Calculate cluster prefetch hit rate for RAGAcc pipelines.")
    add_args_for_ragacc(parser)
    parser.add_argument(
        '--output',
        type=str,
        default="hit_rate_results_all.json",
        help='Output filename for the results'
    )
    return parser.parse_args()


def get_all_round_queries_for_pipeline(pipeline_type: str, data_point: dict) -> list[tuple[Optional[Union[str, list[str]]], Optional[Union[str, list[str]]]]]:
    """
    Extracts the prefetch and retrieval queries for all rounds of a given pipeline.
    Returns a list of tuples, where each tuple is (prefetch_query, retrieval_query) for a round.
    """
    output = data_point.get('output', {})
    question = data_point.get('question')
    queries = []

    if not output or not question:
        return []

    if pipeline_type == "linear":
        prefetch_query = question
        retrieval_query = output.get('hyde_gen_query')
        if retrieval_query:
            queries.append((prefetch_query, retrieval_query))

    elif pipeline_type == "iterative":
        prefetch_query = question
        i = 0
        while f'step_decomposed_retrieval_query_iter_{i}' in output:
            retrieval_query = output.get(f'step_decomposed_retrieval_query_iter_{i}')
            if retrieval_query:
                queries.append((prefetch_query, retrieval_query))
            prefetch_query = retrieval_query  # For the next round
            i += 1

    elif pipeline_type == "iterretgen":
        i = 0
        # In iterretgen, prefetch for round i+1 is the retrieval from round i.
        # We start from round 0, where prefetch is None.
        while f'retrieval_query_iter_{i}' in output:
            retrieval_query = output.get(f'retrieval_query_iter_{i}')
            
            if i == 0:
                prefetch_query = None
            else:
                prefetch_query = output.get(f'retrieval_query_iter_{i-1}')
            
            if retrieval_query:
                queries.append((prefetch_query, retrieval_query))
            i += 1

    elif pipeline_type == "parallel":
        prefetch_query = question
        retrieval_query = output.get('retrieval_queries')
        if retrieval_query:
            queries.append((prefetch_query, retrieval_query))

    elif pipeline_type == "selfrag":
        if output.get("retrieval_flag"):
            prefetch_query = question
            retrieval_query = question
            queries.append((prefetch_query, retrieval_query))

    elif pipeline_type == "flare":
        i = 0
        # In flare, only the first round has a prefetch query.
        while f'generated_retrieval_queries_iter_{i}' in output:
            retrieval_query = output.get(f'generated_retrieval_queries_iter_{i}')
            if not retrieval_query:
                i += 1
                continue

            if i == 0:
                prefetch_query = question
                queries.append((prefetch_query, retrieval_query))
            else:
                # Subsequent rounds have no prefetch, so hit rate is effectively 0.
                # We add a (None, retrieval_query) tuple to represent this.
                queries.append((None, retrieval_query))
            i += 1

    return queries

def main(args: argparse.Namespace):
    """
    Main function to calculate hit rate for all pipelines and datasets.
    """
    if not hasattr(args, 'retrieval_gpu_id') or args.retrieval_gpu_id is None:
        print("Error: --retrieval-gpu-id is required.")
        sys.exit(1)
        
    args.gpu_id = args.retrieval_gpu_id

    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    device = torch.device(f"cuda:{args.gpu_id}")
    print(f"Using device: {device}")

    print("Initializing RAGAccIndex...")
    index = RAGAccIndex(args)

    print(f"Loading embedding model: {args.emb_model}...")
    emb_model = BertModel.from_pretrained(args.emb_model).to(device)
    emb_tokenizer = AutoTokenizer.from_pretrained(args.emb_model)

    @torch.inference_mode()
    def text_to_embedding(text: Union[str, List[str]]) -> torch.Tensor:
        tokens = emb_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        tokens = tokens.to(device)
        emb = emb_model(**tokens).last_hidden_state.mean(dim=1)
        return emb

    dataset_list = ["hotpotqa", "triviaqa", "nq"]
    dataset_list = ["nq"]
    pipelines_to_run = ["linear", "parallel", "iterative", "iterretgen", "flare", "selfrag"]
    all_results = {}

    for dataset in dataset_list:
        print(f"\n{'='*25} Processing dataset: {dataset.upper()} {'='*25}")
        args.datasets = dataset
        dataset_results = {}

        for pipeline_type in pipelines_to_run:
            print(f"\n--- Evaluating pipeline: {pipeline_type.upper()} ---")
            args.pipeline_type = pipeline_type

            # Determine and print prefetch budget
            budget = 0
            if args.budget_type == "small":
                budget_dict = PREFETCH_BUDGET_DICT_SMALL
            elif args.budget_type == "22b":
                budget_dict = PREFETCH_BUDGET_DICT_22B
            else:
                budget_dict = PREFETCH_BUDGET_DICT_LARGE
            
            try:
                budget = budget_dict[args.datasets][args.gpu_model][pipeline_type]
                print(f"Using prefetch budget: {budget} GB")
            except KeyError:
                print(f"Prefetch budget not defined for this configuration.")
            
            budget_bytes = budget * 1024 * 1024 * 1024 if isinstance(budget, (int, float)) else 0

            # Map pipeline type to the correct data directory name used in load_eval_data
            data_pipeline_name = None
            if pipeline_type == "linear":
                data_pipeline_name = "hyde"
            elif pipeline_type == "iterative":
                data_pipeline_name = "llamaindex_iter"
            elif pipeline_type == "parallel":
                data_pipeline_name = "llamaindex_subquestion"

            try:
                input_data = load_eval_data(args, pipeline=data_pipeline_name)
            except FileNotFoundError:
                print(f"Data for pipeline '{pipeline_type}' on dataset '{dataset}' not found. Skipping.")
                continue

            if args.num_samples > 0:
                input_data = input_data[:args.num_samples]

            sample_hit_rates = []
            failure_cases = []

            print(f"Analyzing {len(input_data)} data points...")

            for i, data_point in enumerate(tqdm.tqdm(input_data, desc=f"Processing {dataset}-{pipeline_type}")):
                rounds_queries = get_all_round_queries_for_pipeline(pipeline_type, data_point)

                if not rounds_queries:
                    continue

                round_hit_rates = []
                failure_details = None

                # Clear previous prefetch data from the index
                index.clear_prefetch_data() 
                            
                for round_idx, (prefetch_query, retrieval_query) in enumerate(rounds_queries):
                    if prefetch_query is not None:
                        with torch.inference_mode():
                            prefetch_emb = text_to_embedding(prefetch_query)
                            # Simulate prefetching into the index's internal cache
                            index.prefetch_batch(prefetch_emb, budget=budget)
                    
                    # Calculate hit rate using the index's method
                    # Even if prefetch_query is None (e.g. Flare round > 0), we still check the cache
                    with torch.inference_mode():
                        retrieval_emb = text_to_embedding(retrieval_query)
                        hit_rate = index.get_cluster_hit_rate(retrieval_emb, nprobe=args.nprobe)

                    round_hit_rates.append(hit_rate)

                    # Capture failure details from the first bad round
                    if hit_rate < 0.05 and failure_details is None:
                         with torch.inference_mode():
                             if prefetch_query is not None:
                                 # index.prefetch_clusters contains data from the recent prefetch_batch
                                 cached = index.prefetch_clusters.flatten().tolist() if index.prefetch_clusters.numel() > 0 else []
                                 ret_tensor = index.find_clusters(retrieval_emb, nprobe=args.nprobe)
                                 ret = ret_tensor.flatten().tolist()
                             else:
                                 cached = []
                                 ret_emb = text_to_embedding(retrieval_query)
                                 ret_tensor = index.find_clusters(ret_emb, nprobe=args.nprobe)
                                 ret = ret_tensor.flatten().tolist()
                             
                             cached_set = set(cached)
                             ret_set = set(ret)
                             
                             failure_details = {
                                "round_idx": round_idx,
                                "prefetch_query": prefetch_query,
                                "retrieval_query": retrieval_query,
                                "cached_prefetch_clusters": sorted(list(cached_set)),
                                "retrieval_clusters": sorted(list(ret_set)),
                                "missed_clusters": sorted(list(ret_set - cached_set)),
                             }

                datapoint_avg_hit_rate = np.mean(round_hit_rates) if round_hit_rates else 0
                if round_hit_rates:
                    sample_hit_rates.append(datapoint_avg_hit_rate)

                if datapoint_avg_hit_rate < 0.05 and failure_details:
                    failure_cases.append({
                        "data_index": i,
                        "hit_rate": datapoint_avg_hit_rate,
                        "question": data_point.get('question'),
                        **failure_details
                    })

            overall_hit_rate = np.mean(sample_hit_rates) if sample_hit_rates else 0
            print(f"Result: Overall Hit Rate (average of sample averages) = {overall_hit_rate:.4f}")

            # --- Failure Case Analysis ---
            if failure_cases:
                zero_hit_rate_failures = [f for f in failure_cases if f['hit_rate'] == 0.0]
                print(f"Failure Analysis: {len(failure_cases)} total failures.")
                if zero_hit_rate_failures:
                    print(f"  - Cases with 0% hit rate: {len(zero_hit_rate_failures)}")
                    print("\n  --- Example 0% Hit Rate Cases ---")
                    for i, failure in enumerate(zero_hit_rate_failures[:3]):
                        print(f"  Case #{i+1} (Index: {failure['data_index']})")
                        print(f"    Prefetch Query: '{str(failure['prefetch_query'])[:150]}...'")
                        print(f"    Retrieval Query: '{str(failure['retrieval_query'])[:150]}...'")
                        print(f"    Missed Clusters: {failure['missed_clusters'][:10]}...")
                    print("  ---------------------------------")

            dataset_results[pipeline_type] = {
                "num_samples": len(input_data),
                "overall_hit_rate": float(overall_hit_rate),
                "failure_cases_count": len(failure_cases),
                "failure_cases": failure_cases,
            }
        all_results[dataset] = dataset_results

    print(f"\n{'='*25} Overall Summary {'='*25}")
    for dataset, dataset_results in all_results.items():
        print(f"\nDataset: {dataset.upper()}")
        for pipeline, results in dataset_results.items():
            print(f"  - {pipeline.capitalize():<12}: {results['overall_hit_rate']:.4f} hit rate")

    output_filename = args.output
    with open(output_filename, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nConsolidated results saved to '{output_filename}'")


if __name__ == "__main__":
    args = get_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)
