import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
import argparse
import sys

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
    if name != '':
        ax.set_title(name, fontsize=24)

def parse_csv(filepath):
    pipelines = ['linear', 'parallel', 'iterative', 'iterretgen', 'flare', 'selfrag']
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {k.strip(): v for k, v in raw_row.items()}
            pipeline_name = row['Pipeline'].strip()
            if pipeline_name in pipelines:
                data[pipeline_name] = float(row['Total'])
    
    res = []
    for p in pipelines:
        val = data.get(p, 0)
        res.append(val)
    return res

def compute_speedup(faiss_file, ragacc_file):
    faiss_times = parse_csv(faiss_file)
    ragacc_times = parse_csv(ragacc_file)
    speedups = []
    for i in range(len(faiss_times)):
        f, r = faiss_times[i], ragacc_times[i]
        if r > 0:
            speedups.append(f / r)
        else:
            speedups.append(0)
    avg = sum(speedups) / len(speedups) if len(speedups) > 0 else 0
    speedups.append(avg)
    return speedups

def plot_average(nq: list, hotpot: list, trivia: list, filename: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    data = [[nq[i], hotpot[i], trivia[i]] for i in range(len(nq))]
    plot(ax, data, '', 0)
    plt.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0.1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faiss_nq", required=True)
    parser.add_argument("--ragacc_nq", required=True)
    parser.add_argument("--faiss_hotpot", required=True)
    parser.add_argument("--ragacc_hotpot", required=True)
    parser.add_argument("--faiss_trivia", required=True)
    parser.add_argument("--ragacc_trivia", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    nq = compute_speedup(args.faiss_nq, args.ragacc_nq)
    hotpot = compute_speedup(args.faiss_hotpot, args.ragacc_hotpot)
    trivia = compute_speedup(args.faiss_trivia, args.ragacc_trivia)

    plot_average(nq, hotpot, trivia, args.output)
    print(f"Plot saved to {args.output}")

if __name__ == '__main__':
    main()
