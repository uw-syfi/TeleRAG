import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
import argparse

colors = [
    ['#b32414', '#de6000', '#e4c315'],
    [
        '#a4e5ff',
        '#00c1f9',
        '#0098e7',
        '#00549f',
    ],
    ['#12b9cd', '#009e54'],
    ['#a0cfa0', '#429b44', '#38953b', '#107515'],
    ['#b32414', '#de6000'],
]
pipeline = ['HyDE', 'SubQ', 'Iter', 'IRG', 'FLARE', 'S-RAG', 'Avg']
SYS_NAME = 'TeleRAG'


def plot(ax, data, name, plot_type, show_y_label=True, show_legend=True):
    ind = np.array([1])
    width = 0.5
    hatches = ['/', '\\', 'x', '-', '+', '|']
    x_pos = []
    if plot_type == 0 or plot_type == 3:
        ax.bar(15, [100], 2.5, color='#00000050')
    elif plot_type == 1:
        ax.bar(18.75, [100], 4, color='#00000050')
    elif plot_type == 2:
        ax.bar(12, [100], 2.5, color='#00000050')
    color_index = 0 if plot_type == 3 else plot_type

    def add_one_workload(data, offset):
        ret = []
        x_pos.append(ind + offset + 0.5 * width * (len(data) - 1))
        for i in range(len(data)):
            ret.append(ax.bar(ind + offset + i * width, data[i], width, color=colors[color_index][i], hatch=hatches[i], edgecolor=['black'] * len(ind)))
        return ret

    rects = []
    for i in range(len(data)):
        rects.append(add_one_workload(data[i], (1.5 + len(data[0])) * width * i))

    ax.set_xticks(np.concatenate(x_pos))
    ax.set_xticklabels(pipeline, fontsize=20, ha='center')
    ax.tick_params(axis='y', labelsize=20)
    if show_y_label:
        ax.set_ylabel('Speedup' if plot_type == 0 else 'Throughput (Req/s)', fontsize=24)
    if plot_type == 0:
        ax.set_ylim([0, 2.5])
    elif plot_type == 1:
        ax.set_ylim([0, 60])
    elif plot_type == 2:
        ax.set_ylim([0, 12])
    elif plot_type == 3:
        ax.set_ylim([0, 30])
    else:
        raise Exception('Invalid plot type')
    
    if plot_type == 0 or plot_type == 3:
        ax.set_xlim([0, 16.25])
    elif plot_type == 1:
        ax.set_xlim([0, 19.75])
    elif plot_type == 2:
        ax.set_xlim([0.25, 12.75])
        
    ax.xaxis.grid(False)
    ax.yaxis.grid()
    ax.set_axisbelow(True)

    if show_legend:
        if plot_type == 0:
            ax.legend((rects[0][0], rects[0][1], rects[0][2]),
                      ('NQ', 'HotpotQA', 'TriviaQA'),
                      fontsize=16, bbox_to_anchor=(0.6, 1.0), ncol=3,
            )
        elif plot_type == 1:
            ax.legend((rects[0][0], rects[0][1], rects[0][2], rects[0][3]),
                      ('1 GPU', '2 GPUs', '4 GPUs', '8 GPUs'),
                      fontsize=16, loc='upper left', ncol=2,
            )
        elif plot_type == 2:
            ax.legend((rects[0][0], rects[0][1]), ('Faiss', SYS_NAME),
                        fontsize=16, bbox_to_anchor=(0.4, 1.0), ncol=2,
                )
        elif plot_type == 3:
            ax.legend((rects[0][0], rects[0][1], rects[0][2]),
                      ('Naive', 'Greedy without Cache', 'Greedy with Cache'),
                      fontsize=16, bbox_to_anchor=(0.9, 1.15), ncol=3,
            )
    if name != '':
        ax.set_title(name, fontsize=24)

def plot_single(data, filename: str, type_id: int):
    fig, ax = plt.subplots(figsize=(10, 5))
    plot(ax, data, '', type_id)
    plt.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0.1)


def parse_csv_multi_gpu(filepath):
    """
    Parses the evaluation CSV for cache fraction == 0.5 (where applicable) and
    extracts the throughput for gpu sizes = [1, 2, 4, 8].
    Returns a list of lists corresponding to pipelines = ['linear', 'parallel', 'iterative', 'iterretgen', 'flare', 'selfrag', 'Avg']
    Each sublist contains 4 throughput elements.
    """
    pipelines_order = ['linear', 'parallel', 'iterative', 'iterretgen', 'flare', 'selfrag']
    data_dict = {p: {} for p in pipelines_order}
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {k.strip(): v.strip() for k, v in raw_row.items()}
            
            p_name = row['Pipeline']
            if p_name not in pipelines_order:
                continue
            
            cache_frac = float(row['Cache-Frac'])
            if cache_frac != 0.5:
                continue
                
            num_gpu = int(row['Num-GPU'])
            if num_gpu not in [1, 2, 4, 8]:
                continue
                
            global_batch = int(row['Global-Batch'])
            total_time = float(row['Total'])
            throughput = global_batch / total_time
            
            data_dict[p_name][num_gpu] = throughput
            
    formatted_data = []
    gpu_totals = {1: 0, 2: 0, 4: 0, 8: 0}
    
    for p in pipelines_order:
        gpu_data = []
        for g in [1, 2, 4, 8]:
            val = data_dict[p].get(g, 0.0)
            gpu_data.append(val)
            gpu_totals[g] += val
        formatted_data.append(gpu_data)
        
    # Calculate Average across pipelines
    avg_data = [gpu_totals[g] / len(pipelines_order) for g in [1, 2, 4, 8]]
    formatted_data.append(avg_data)
    
    return formatted_data


def main():
    parser = argparse.ArgumentParser(description="Plot H200 Multi-gpu figures from evaluation CSVs")
    parser.add_argument("--csv", required=True, help="Path to input dataset CSV (e.g. nq)")
    parser.add_argument("--output", required=True, help="Path to save the output PDF")
    
    args = parser.parse_args()
    
    data = parse_csv_multi_gpu(args.csv)
    
    # Use the plot type 1 for multi GPU as in speedups.py
    plot_single(data, args.output, 1)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
