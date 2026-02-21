import argparse
import csv
import sys
import os

sys.path.append(os.path.dirname(__file__))
from plot_batch import plot_batch_figures_per_pipeline

def parse_csv_cache_comparison(filepath):
    """
    Parses the evaluation CSV to extract two sets of data:
    1. Cache-Frac == 0.0 (No Cache)
    2. Cache-Frac == 0.5 (With Cache)
    Returns:
        data_no_cache: List of lists (for each pipeline, containing throughputs across 1, 2, 4, 8 GPUs)
        data_cache: List of lists (for each pipeline, containing throughputs across 1, 2, 4, 8 GPUs)
    """
    pipelines_order = ['linear', 'parallel', 'iterative', 'iterretgen', 'flare', 'selfrag']
    data_no_cache_dict = {p: {} for p in pipelines_order}
    data_cache_dict = {p: {} for p in pipelines_order}

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {k.strip(): v.strip() for k, v in raw_row.items()}
            
            p_name = row['Pipeline']
            if p_name not in pipelines_order:
                continue
                
            num_gpu = int(row['Num-GPU'])
            if num_gpu not in [1, 2, 4, 8]:
                continue
            
            # Use 'Global-Batch' / 'Total' for throughput
            global_batch = int(row['Global-Batch'])
            total_time = float(row['Total'])
            throughput = global_batch / total_time
            
            cache_frac = float(row['Cache-Frac'])
            if cache_frac == 0.0:
                data_no_cache_dict[p_name][num_gpu] = throughput
            elif cache_frac == 0.5:
                data_cache_dict[p_name][num_gpu] = throughput

    def format_data_and_avg(data_dict):
        formatted_data = []
        gpu_totals = {1: 0, 2: 0, 4: 0, 8: 0}
        
        for p in pipelines_order:
            gpu_data = []
            for g in [1, 2, 4, 8]:
                val = data_dict[p].get(g, 0.0)
                gpu_data.append(val)
                gpu_totals[g] += val
            formatted_data.append(gpu_data)
            
        # Add average across pipelines
        avg_data = [gpu_totals[g] / len(pipelines_order) for g in [1, 2, 4, 8]]
        formatted_data.append(avg_data)
        
        return formatted_data

    return format_data_and_avg(data_no_cache_dict), format_data_and_avg(data_cache_dict)


def main():
    parser = argparse.ArgumentParser(description="Plot H200 throughput cache comparison figures")
    parser.add_argument("--csv", required=True, help="Path to input dataset CSV (e.g. nq)")
    parser.add_argument("--output", required=True, help="Path to save the output PDF")
    
    args = parser.parse_args()
    
    data_no_cache, data_cache = parse_csv_cache_comparison(args.csv)
    
    # Generate cache comparison plot using plot_batch_figures_per_pipeline
    plot_batch_figures_per_pipeline(data_no_cache, data_cache, args.output, data_type='cache-comparison')
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
