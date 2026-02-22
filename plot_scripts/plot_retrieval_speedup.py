import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
import csv
import argparse
from collections import defaultdict

colors = ['#70cb70', '#4c984e', '#1b591e']
pipeline_names = ['HyDE', 'SubQ', 'Iter', 'IRG', 'FLARE', 'S-RAG', 'Avg']
SYS_NAME = 'TeleRAG'

def plot(ax, data, name, show_y_label=True, show_legend=True):
    ind = np.array([1])
    width = 0.5
    hatches = ['/', '\\', 'x', '+', '-', '|']
    x_pos = []
    ax.bar(15, [100], 2.5, color='#00000050')

    def add_one_workload(data, offset):
        ret = []
        x_pos.append(ind + offset + 0.5 * width * (len(data) - 1))
        for i in range(len(data)):
            ret.append(ax.bar(ind + offset + i * width, data[i], width, color=colors[i], hatch=hatches[i], edgecolor=['black'] * len(ind)))
        return ret

    rects = []
    for i in range(len(data)):
        rects.append(add_one_workload(data[i], (1.5 + len(data[0])) * width * i))

    ax.set_xticks(np.concatenate(x_pos))
    ax.set_xticklabels(pipeline_names, fontsize=20, ha='center')
    if show_y_label:
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylabel('Speedup', fontsize=24)
    else:
        ax.tick_params(axis='y', labelsize=20)
    ax.set_ylim([0, 12])
    ax.set_xlim([0, 16.25])
    ax.xaxis.grid(False)
    ax.yaxis.grid()
    ax.set_axisbelow(True)

    if show_legend:
        ax.legend((rects[0][0], rects[0][1], rects[0][2]),
                  ('Nprobe 128', 'Nprobe 256', 'Nprobe 512'), fontsize=18, loc='upper right')

def parse_csv(filepath):
    pipelines = ['linear', 'parallel', 'iterative', 'iterretgen', 'flare', 'selfrag']
    data = defaultdict(dict)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {k.strip(): v for k, v in raw_row.items()}
            pipeline_name = row['Pipeline'].strip()
            if pipeline_name in pipelines:
                nprobe = int(row['Nprobe'])
                data[pipeline_name][nprobe] = float(row['Retrieval'])
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faiss", required=True)
    parser.add_argument("--ragacc", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    faiss_data = parse_csv(args.faiss)
    ragacc_data = parse_csv(args.ragacc)

    pipelines = ['linear', 'parallel', 'iterative', 'iterretgen', 'flare', 'selfrag']
    nprobes = [128, 256, 512]

    plot_data = []
    for p in pipelines:
        p_data = []
        for n in nprobes:
            faiss_val = faiss_data.get(p, {}).get(n, 0)
            ragacc_val = ragacc_data.get(p, {}).get(n, 0)
            if ragacc_val > 0:
                p_data.append(faiss_val / ragacc_val)
            else:
                p_data.append(0)
        plot_data.append(p_data)

    # Average over pipelines
    avg_data = []
    for i, n in enumerate(nprobes):
        sum_val = sum(plot_data[j][i] for j in range(len(pipelines)))
        avg_data.append(sum_val / len(pipelines))
    plot_data.append(avg_data)

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    ax1 = plt.subplot(gs[0])
    plot(ax1, plot_data, args.name, show_y_label=True, show_legend=True)
    plt.savefig(args.output, dpi=200, bbox_inches='tight', pad_inches=0.1)
    print(f"Plot saved to {args.output}")

if __name__ == '__main__':
    main()
