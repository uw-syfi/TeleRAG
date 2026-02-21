import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np

FONT_SIZE = 20
LABEL_SIZE = 20
ANNOTATION_SIZE = 16
LEGEND_SIZE = 14

pipeline_order = ['linear', 'parallel', 'iterative', 'iterretgen', 'flare', 'selfrag']
pipeline_labels = ['HyDE', 'SubQ', 'Iter', 'IRG', 'FLARE', 'S-RAG']

schedule_colors = {
    'naive': '#00c5c1',
    'prefetch-only': '#429b44',
    'prefetch-cache': '#0098e7',
    'overhead': '#EE7002',
}

def parse_overhead_data(no_schedule_csv, prefetch_only_csv, with_cache_csv):
    """
    Extracts Data from 4-GPU & Cache-Fraction 0.5.
    Returns format mapping arrays by index of pipeline_order:
        naive_total_time,
        prefetch_only_scheudle_time, prefetch_only_total_time,
        prefetch_cache_schedule_time, prefetch_cache_total_time
    """
    naive_total_time = [0] * len(pipeline_order)
    
    prefetch_only_schedule_time = [0] * len(pipeline_order)
    prefetch_only_total_time = [0] * len(pipeline_order)
    
    prefetch_cache_schedule_time = [0] * len(pipeline_order)
    prefetch_cache_total_time = [0] * len(pipeline_order)
    
    def extract_from_csv(filepath, schedule_time_out, total_time_out):
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                row = {k.strip(): v.strip() for k, v in raw_row.items()}
                p_name = row['Pipeline']
                if p_name not in pipeline_order:
                    continue
                num_gpu = int(row['Num-GPU'])
                cache_frac = float(row['Cache-Frac'])
                if num_gpu == 4 and cache_frac == 0.5:
                    idx = pipeline_order.index(p_name)
                    schedule_time_out[idx] = float(row['Mini-Batch-Time'])
                    total_time_out[idx] = float(row['Total'])

    # Naive is extracted as total-time only (scheduling overhead not separated/is negligible initially)
    # The dummy dict is just to absorb unused schedule output
    extract_from_csv(no_schedule_csv, [0] * len(pipeline_order), naive_total_time)
    extract_from_csv(prefetch_only_csv, prefetch_only_schedule_time, prefetch_only_total_time)
    extract_from_csv(with_cache_csv, prefetch_cache_schedule_time, prefetch_cache_total_time)

    return (naive_total_time, prefetch_only_schedule_time, prefetch_only_total_time, 
            prefetch_cache_schedule_time, prefetch_cache_total_time)

def plot_group_overhead(naive_total_time, prefetch_only_scheudle_time, prefetch_only_total_time, 
                        prefetch_cache_schedule_time, prefetch_cache_total_time, output_filepath):
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(naive_total_time))
    bar_width = 0.22
    
    prefetch_only_rest_time = [prefetch_only_total_time[i] - prefetch_only_scheudle_time[i] for i in range(len(prefetch_only_total_time))]
    prefetch_cache_rest_time = [prefetch_cache_total_time[i] - prefetch_cache_schedule_time[i] for i in range(len(prefetch_cache_total_time))]
    
    naive = ax.bar(x_pos - 1 * bar_width, naive_total_time, bar_width, color=schedule_colors['naive'], hatch='/', edgecolor='black', label='Naive Retrieval')
    
    prefetch_only_rest = ax.bar(x_pos, prefetch_only_rest_time, bar_width, color=schedule_colors['prefetch-only'], hatch='\\', edgecolor='black', label='Prefetch Only --- Rest')
    prefetch_only_schedule = ax.bar(x_pos, prefetch_only_scheudle_time, bar_width, bottom=prefetch_only_rest_time, color=schedule_colors['overhead'], hatch='o', edgecolor='black', label='Prefetch Only --- Schedule')
    
    prefetch_cache_rest = ax.bar(x_pos + bar_width, prefetch_cache_rest_time, bar_width, color=schedule_colors['prefetch-cache'], hatch='x', edgecolor='black', label='Prefetch + Cache --- Rest')
    prefetch_cache_schedule = ax.bar(x_pos + bar_width, prefetch_cache_schedule_time, bar_width, bottom=prefetch_cache_rest_time, color=schedule_colors['overhead'], hatch='o', edgecolor='black', label='Prefetch + Cache --- Schedule')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pipeline_labels, fontsize=LABEL_SIZE, ha='center')
    ax.set_ylabel('Time (s)', fontsize=FONT_SIZE)
    ax.tick_params(axis='y', labelsize=LABEL_SIZE)
    
    ax.legend(
        (prefetch_only_schedule, naive, prefetch_only_rest, prefetch_cache_rest),
        ('Schedule Overhead', 'No Schedule',
         'Prefetch Schedule Only', 'Prefetch & Cache Schedule'),
        fontsize=LABEL_SIZE - 3, loc='upper left'
    )
    
    ax.yaxis.grid()
    ax.set_ylim([0, 20])
    ax.set_axisbelow(True)
    
    plt.savefig(output_filepath, dpi=200, bbox_inches='tight', pad_inches=0.1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_schedule", required=True)
    parser.add_argument("--prefetch_only", required=True)
    parser.add_argument("--with_cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = parse_overhead_data(args.no_schedule, args.prefetch_only, args.with_cache)
    plot_group_overhead(*data, args.output)
    print(f"Schedule Overhead plot saved to {args.output}")

if __name__ == "__main__":
    main()
