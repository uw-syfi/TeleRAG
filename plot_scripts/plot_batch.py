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

def plot_double_parameters(ax, data1, data2, name, show_left_label=True, show_right_label=True, show_legend=True, data_type='throughput'):
    ind = np.array([1])
    width = 0.5
    hatches = ['/', '\\', 'x', '-', '+', '|']
    x_pos = [i for i in range(len(data1))]
    bars = ax.bar(x_pos, data1, width, color=colors[1], hatch=hatches[:len(data1)], edgecolor='black')
    if show_left_label:
        ax.set_ylabel('Throughput (Req/s)', fontsize=24, labelpad=10)
    if data2 is not None:
        ax2 = ax.twinx()
        line = ax2.plot(x_pos, data2, color='black', marker='o', linewidth=4, markerfacecolor='black')
        if show_right_label:
            ax2.set_ylabel('Speedup', fontsize=24, labelpad=40)
    ax.xaxis.grid(False)
    ax.yaxis.grid()
    ax.set_axisbelow(True)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(2**i) for i in x_pos], fontsize=20, ha='center')
    if data2 is not None:
        ax2.set_ylim([0, 12])
    ax.set_ylim([0, 60])
    ax.tick_params(axis='y', labelsize=20)
    ax.text(1.5, 62, name, fontsize=24, ha='center')

    if show_legend:
        if data2 is not None:
            ax.legend(
                (bars[0], bars[1], bars[2], bars[3], line[0]),
                ('Batch 1', 'Batch 2', 'Batch 4', 'Batch 8', 'Speedup'),
                fontsize=16, bbox_to_anchor=(2.9, 1.2), ncol=5,
            )
        else:
            ax.legend(
                (bars[0], bars[1], bars[2], bars[3]),
                ('1 GPU', '2 GPUs', '4 GPUs', '8 GPUs'),
                fontsize=16, bbox_to_anchor=(0.75, 1), ncol=1,
            )

def plot_batch_parameters_v2(ax, data1, data2, name, show_left_label=True,
                             show_right_label=True, show_legend=True,
                             data_type='throughput', use_small_scale=False):
    width = 0.4
    x_pos = [i for i in range(len(data1))]
    hatches = ['/', '\\', 'x', '-', '+', '|']
    if data_type == 'throughput' or data_type == 'multi-gpu':
        speedup = [data2[i] / data1[i] for i in range(len(data1))]
    elif data_type == 'latency':
        speedup = [data1[i] / data2[i] for i in range(len(data1))]
    elif data_type == 'cache-comparison':
        speedup = None
    else:
        raise Exception('Invalid data type')
    color_index = 4 if data_type == 'cache-comparison' else 2
    def add_one_workload(index):
        return [
            ax.bar(index - width * 0.5, data1[index], width, color=colors[color_index][0], hatch=hatches[0], edgecolor='black'),
            ax.bar(index + width * 0.5, data2[index], width, color=colors[color_index][1], hatch=hatches[1], edgecolor='black'),
        ]
    bars = []
    for i in range(len(data1)):
        bars.append(add_one_workload(i))
    if show_left_label:
        if data_type == 'throughput' or data_type == 'multi-gpu':
            ax.set_ylabel('Throughput (Req/s)', fontsize=24)
        elif data_type == 'latency':
            ax.set_ylabel('Latency (s)', fontsize=24, labelpad=10)
        elif data_type == 'cache-comparison':
            ax.set_ylabel('Throughput (Req/s)', fontsize=24, labelpad=10)
    if speedup is not None:
        ax2 = ax.twinx()
        line = ax2.plot(x_pos, speedup, color='black', marker='o', linewidth=4, markerfacecolor='black')
        if show_right_label:
            ax2.set_ylabel('Speedup', fontsize=24)
    ax.xaxis.grid(False)
    ax.yaxis.grid()
    ax.set_axisbelow(True)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(2**i) for i in x_pos], fontsize=16, ha='center')
    if data_type == 'throughput':
        if use_small_scale:
            ax.set_ylim([0, 8.0])
            ax2.set_ylim([0, 2.0])
            name_height = 8.2
        else:
            ax.set_ylim([0, 12])
            ax2.set_ylim([0, 3])
            name_height = 12.3
        x_label = 'Batch Size'
    elif data_type == 'latency':
        ax.set_ylim([0, 5])
        ax2.set_ylim([0, 5])
        name_height = 5.2
        x_label = 'Batch Size'
    elif data_type == 'multi-gpu':
        ax.set_ylim([0, 60])
        ax2.set_ylim([0, 3])
        name_height = 60.2
        x_label = 'Number of GPUs'
    elif data_type == 'cache-comparison':
        ax.set_ylim([0, 60])
        name_height = 61.8
        x_label = 'Number of GPUs'
    
    ax.text(1.5, name_height, name, fontsize=20, ha='center')
    ax.set_xlabel(x_label, fontsize=20)
    ax.tick_params(axis='y', labelsize=16)
    if speedup is not None:
        ax2.tick_params(axis='y', labelsize=16)
    if show_legend:
        if speedup is not None:
            ax.legend(
                (bars[0][0], bars[0][1], line[0]),
                ('Baseline', SYS_NAME, 'Speedup'),
                fontsize=16, loc='upper left',
            )
        else:
            if data_type == 'cache-comparison':
                ax.legend(
                    (bars[0][0], bars[0][1]),
                    ('w/o cache', 'w/ cache'),
                    fontsize=16, loc='upper left',
                )
            else:
                ax.legend(
                    (bars[0][0], bars[1][1]),
                    ('Faiss', SYS_NAME),
                    fontsize=16, loc='upper left',
                )

def plot_batch_figures_per_pipeline(faiss: list, ragacc: list, filename: str, data_type='throughput'):
    type_id = 3
    num_pipeline = 7
    fig, ax = plt.subplots(1, num_pipeline, figsize=(30, 4.5))

    use_small_scale = False
    if data_type == 'throughput':
        max_throughput = 0
        max_speedup = 0
        for i in range(num_pipeline):
            for j in range(len(faiss[i])):
                max_throughput = max(max_throughput, faiss[i][j], ragacc[i][j])
                if faiss[i][j] > 0:
                    max_speedup = max(max_speedup, ragacc[i][j] / faiss[i][j])
        if max_throughput <= 8 and max_speedup < 2.0:
            use_small_scale = True

    def plot_one_figure(index):
        data1 = faiss[index]
        data2 = ragacc[index]
        left = index == 0
        right = index == num_pipeline - 1
        if data_type == 'multi-gpu':
            plot_double_parameters(ax[index], data2, None, pipeline[index], left, right, left, data_type=data_type)
        else:
            plot_batch_parameters_v2(ax[index], data1, data2, pipeline[index], left, right, left, data_type=data_type, use_small_scale=use_small_scale)
    for i in range(num_pipeline):
        plot_one_figure(i)
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0.1)

def parse_csv(filepath):
    """
    Parses the CSV file and returns throughput arrays corresponding to the 
    batch sizes 1, 2, 4, 8 for each pipeline.
    
    Returns a list of lists, order matching old-figure-scripts/speedups.py:
    ['HyDE', 'SubQ', 'Iter', 'IRG', 'FLARE', 'S-RAG', 'Avg']
    """
    pipelines = ['linear', 'parallel', 'iterative', 'iterretgen', 'flare', 'selfrag']
    data = {p: {} for p in pipelines}
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {k.strip(): v for k, v in raw_row.items()}
            pipeline_name = row['Pipeline'].strip()
            if pipeline_name not in data:
                continue
            
            global_batch = int(row['Global-Batch'])
            if global_batch not in [1, 2, 4, 8]:
                continue
            
            total_time = float(row['Total'])
            throughput = global_batch / total_time
            data[pipeline_name][global_batch] = throughput
            
    # Format according to the expected speedups.py lists (1, 2, 4, 8)
    formatted_data = []
    
    # Store throughputs for averaging later
    batch_totals = {1: 0, 2: 0, 4: 0, 8: 0}
    
    for p in pipelines:
        pipeline_data = []
        for b in [1, 2, 4, 8]:
            val = data[p].get(b, 0.0)
            pipeline_data.append(val)
            batch_totals[b] += val
        formatted_data.append(pipeline_data)
        
    # Calculate Average
    avg_data = [batch_totals[b] / len(pipelines) for b in [1, 2, 4, 8]]
    formatted_data.append(avg_data)
    
    return formatted_data

def main():
    parser = argparse.ArgumentParser(description="Plot batch figures from evaluation CSVs")
    parser.add_argument("--baseline", required=True, help="Path to the baseline Faiss CSV")
    parser.add_argument("--ragacc", required=True, help="Path to the RAGAcc CSV")
    parser.add_argument("--output", required=True, help="Path to save the output PDF")
    
    args = parser.parse_args()
    
    faiss_data = parse_csv(args.baseline)
    ragacc_data = parse_csv(args.ragacc)
    
    plot_batch_figures_per_pipeline(faiss_data, ragacc_data, args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
