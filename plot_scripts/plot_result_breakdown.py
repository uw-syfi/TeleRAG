import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import argparse
import csv

FONT_SIZE = 20
LABEL_SIZE = 20
LEGEND_SIZE = 14

BASELINE_NAME = 'Faiss'
SYSTEM_NAME = 'TeleRAG'

colors = {
    'llm': '#8c4885',
    'cpu-1': '#80dce5',
    'cpu-2': '#28c4d3',
    'cpu-4': '#01aaba',
    'cpu-8': '#048189',
    'gpu-1': '#8bd5af',
    'gpu-2': '#0fba77',
    'gpu-4': '#009e56',
    'gpu-8': '#007b3e',
    'misc': '#555',
}
hatch = {
    'llm': '',
    'cpu': '/',
    'gpu': '\\',
    'misc': 'o',
}
pipeline = ['HyDE', 'SubQ', 'Iter', 'IRG', 'FLARE', 'S-RAG']

def to_four_lists(data) -> tuple:
    return (
        [data[i][0] for i in range(len(data))],
        [data[i][1] for i in range(len(data))],
        [data[i][2] for i in range(len(data))],
        [data[i][3] for i in range(len(data))],
    )

def plot_batch(faiss_llm, faiss_ret, faiss_misc, ragacc_llm, ragacc_ret, ragacc_misc, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    faiss_llm_1, faiss_llm_2, faiss_llm_4, faiss_llm_8 = to_four_lists(faiss_llm)
    faiss_ret_1, faiss_ret_2, faiss_ret_4, faiss_ret_8 = to_four_lists(faiss_ret)
    faiss_misc_1, faiss_misc_2, faiss_misc_4, faiss_misc_8 = to_four_lists(faiss_misc)
    ragacc_llm_1, ragacc_llm_2, ragacc_llm_4, ragacc_llm_8 = to_four_lists(ragacc_llm)
    ragacc_ret_1, ragacc_ret_2, ragacc_ret_4, ragacc_ret_8 = to_four_lists(ragacc_ret)
    ragacc_misc_1, ragacc_misc_2, ragacc_misc_4, ragacc_misc_8 = to_four_lists(ragacc_misc)

    width = 0.1
    padding = 0.15
    x_pos = []
    pipeline_pos = []
    x_label = []
    x_pos_faiss_1 = []
    x_pos_faiss_2 = []
    x_pos_faiss_4 = []
    x_pos_faiss_8 = []
    x_pos_ragacc_1 = []
    x_pos_ragacc_2 = []
    x_pos_ragacc_4 = []
    x_pos_ragacc_8 = []
    for i in range(len(pipeline)):
        pipeline_pos.append(i + 3 * width + padding / 2)
        x_pos_faiss_1.append(i)
        x_pos.append(i + width * 1.5)
        x_label.append('base')
        x_pos_faiss_2.append(i + width)
        x_pos_faiss_4.append(i + 2 * width)
        x_pos_faiss_8.append(i + 3 * width)

        x_pos_ragacc_1.append(i + 3 * width + padding)
        x_pos.append(i + 3 * width + padding + width * 1.5)
        x_label.append('ours')
        x_pos_ragacc_2.append(i + 3 * width + padding + width)
        x_pos_ragacc_4.append(i + 3 * width + padding + 2 * width)
        x_pos_ragacc_8.append(i + 3 * width + padding + 3 * width)

    def plot_bar(ax, data, x_pos, base, color, label, hatch):
        if base is None:
            base = [0 for _ in range(len(data))]
        return ax.bar(x_pos, data, width, bottom=base, color=color, edgecolor='black', label=label, hatch=hatch)

    llm_faiss_1_bar = plot_bar(ax, faiss_llm_1, x_pos_faiss_1, None, colors['llm'], 'LLM', hatch['llm'])
    llm_faiss_2_bar = plot_bar(ax, faiss_llm_2, x_pos_faiss_2, None, colors['llm'], 'LLM', hatch['llm'])
    llm_faiss_4_bar = plot_bar(ax, faiss_llm_4, x_pos_faiss_4, None, colors['llm'], 'LLM', hatch['llm'])
    llm_faiss_8_bar = plot_bar(ax, faiss_llm_8, x_pos_faiss_8, None, colors['llm'], 'LLM', hatch['llm'])

    ret_faiss_1_bar = plot_bar(ax, faiss_ret_1, x_pos_faiss_1, faiss_llm_1, colors['cpu-1'], 'CPU-1', hatch['cpu'])
    ret_faiss_2_bar = plot_bar(ax, faiss_ret_2, x_pos_faiss_2, faiss_llm_2, colors['cpu-2'], 'CPU-2', hatch['cpu'])
    ret_faiss_4_bar = plot_bar(ax, faiss_ret_4, x_pos_faiss_4, faiss_llm_4, colors['cpu-4'], 'CPU-4', hatch['cpu'])
    ret_faiss_8_bar = plot_bar(ax, faiss_ret_8, x_pos_faiss_8, faiss_llm_8, colors['cpu-8'], 'CPU-8', hatch['cpu'])

    misc_faiss_1_bar = plot_bar(ax, faiss_misc_1, x_pos_faiss_1, [i + j for i, j in zip(faiss_llm_1, faiss_ret_1)], colors['misc'], 'Misc', hatch['misc'])
    misc_faiss_2_bar = plot_bar(ax, faiss_misc_2, x_pos_faiss_2, [i + j for i, j in zip(faiss_llm_2, faiss_ret_2)], colors['misc'], 'Misc', hatch['misc'])
    misc_faiss_4_bar = plot_bar(ax, faiss_misc_4, x_pos_faiss_4, [i + j for i, j in zip(faiss_llm_4, faiss_ret_4)], colors['misc'], 'Misc', hatch['misc'])
    misc_faiss_8_bar = plot_bar(ax, faiss_misc_8, x_pos_faiss_8, [i + j for i, j in zip(faiss_llm_8, faiss_ret_8)], colors['misc'], 'Misc', hatch['misc'])

    llm_ragacc_1_bar = plot_bar(ax, ragacc_llm_1, x_pos_ragacc_1, None, colors['llm'], 'LLM', hatch['llm'])
    llm_ragacc_2_bar = plot_bar(ax, ragacc_llm_2, x_pos_ragacc_2, None, colors['llm'], 'LLM', hatch['llm'])
    llm_ragacc_4_bar = plot_bar(ax, ragacc_llm_4, x_pos_ragacc_4, None, colors['llm'], 'LLM', hatch['llm'])
    llm_ragacc_8_bar = plot_bar(ax, ragacc_llm_8, x_pos_ragacc_8, None, colors['llm'], 'LLM', hatch['llm'])

    ret_ragacc_1_bar = plot_bar(ax, ragacc_ret_1, x_pos_ragacc_1, ragacc_llm_1, colors['gpu-1'], 'GPU-1', hatch['gpu'])
    ret_ragacc_2_bar = plot_bar(ax, ragacc_ret_2, x_pos_ragacc_2, ragacc_llm_2, colors['gpu-2'], 'GPU-2', hatch['gpu'])
    ret_ragacc_4_bar = plot_bar(ax, ragacc_ret_4, x_pos_ragacc_4, ragacc_llm_4, colors['gpu-4'], 'GPU-4', hatch['gpu'])
    ret_ragacc_8_bar = plot_bar(ax, ragacc_ret_8, x_pos_ragacc_8, ragacc_llm_8, colors['gpu-8'], 'GPU-8', hatch['gpu'])

    misc_ragacc_1_bar = plot_bar(ax, ragacc_misc_1, x_pos_ragacc_1, [i + j for i, j in zip(ragacc_llm_1, ragacc_ret_1)], colors['misc'], 'Misc', hatch['misc'])
    misc_ragacc_2_bar = plot_bar(ax, ragacc_misc_2, x_pos_ragacc_2, [i + j for i, j in zip(ragacc_llm_2, ragacc_ret_2)], colors['misc'], 'Misc', hatch['misc'])
    misc_ragacc_4_bar = plot_bar(ax, ragacc_misc_4, x_pos_ragacc_4, [i + j for i, j in zip(ragacc_llm_4, ragacc_ret_4)], colors['misc'], 'Misc', hatch['misc'])
    misc_ragacc_8_bar = plot_bar(ax, ragacc_misc_8, x_pos_ragacc_8, [i + j for i, j in zip(ragacc_llm_8, ragacc_ret_8)], colors['misc'], 'Misc', hatch['misc'])

    for i, pos in enumerate(pipeline_pos):
        ax.text(pos, -1.2, pipeline[i], ha='center', fontsize=LABEL_SIZE)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_label, fontsize=16)
    ax.set_ylabel('Time (s)', fontsize=FONT_SIZE)
    ax.set_ylim((0, 8))
    ax.tick_params(axis='y', labelsize=LABEL_SIZE)
    ax.legend(
        (
            ret_faiss_1_bar, ret_faiss_2_bar, ret_faiss_4_bar, ret_faiss_8_bar,
            llm_faiss_1_bar,
            ret_ragacc_1_bar, ret_ragacc_2_bar, ret_ragacc_4_bar, ret_ragacc_8_bar,
            misc_faiss_1_bar,
        ),
        (
            f'{BASELINE_NAME} Batch 1', f'{BASELINE_NAME} Batch 2', f'{BASELINE_NAME} Batch 4', f'{BASELINE_NAME} Batch 8',
            'LLM',
            f'{SYSTEM_NAME} Batch 1', f'{SYSTEM_NAME} Batch 2', f'{SYSTEM_NAME} Batch 4', f'{SYSTEM_NAME} Batch 8',
            'Misc',
        ),
        fontsize=LEGEND_SIZE,
        loc='upper right',
        ncol=2,
    )
    ax.yaxis.grid()
    ax.set_axisbelow(True)

    plt.savefig(title, dpi=300, bbox_inches='tight', pad_inches=0.1)

def parse_csv(filepath):
    pipelines = ['linear', 'parallel', 'iterative', 'iterretgen', 'flare', 'selfrag']
    batches = [1, 2, 4, 8]
    
    # Structure: dict[pipeline][batch] = {'llm': x, 'retrieval': y, 'misc': z}
    data = {p: {b: {'llm': 0, 'retrieval': 0, 'misc': 0} for b in batches} for p in pipelines}
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {k.strip(): v for k, v in raw_row.items()}
            pl = row['Pipeline'].strip()
            if pl not in pipelines:
                continue
            
            b = int(row['Global-Batch'])
            if b not in batches:
                continue
                
            pre_llm = float(row['Pre-Ret-LLM'])
            post_llm = float(row['Post-Ret-LLM'])
            retrieval = float(row['Retrieval'])
            total = float(row['Total'])
            
            llm = pre_llm + post_llm
            misc = max(0, total - llm - retrieval)
            
            data[pl][b] = {
                'llm': llm,
                'retrieval': retrieval,
                'misc': misc
            }
            
    # Convert dict into list of lists to match plotting function expectations
    # e.g. [[batch 1, batch 2, batch 4, batch 8] for each pipeline]
    llm_matrix = []
    retrieval_matrix = []
    misc_matrix = []
    
    for p in pipelines:
        p_llm = []
        p_ret = []
        p_misc = []
        for b in batches:
            p_llm.append(data[p][b]['llm'])
            p_ret.append(data[p][b]['retrieval'])
            p_misc.append(data[p][b]['misc'])
        llm_matrix.append(p_llm)
        retrieval_matrix.append(p_ret)
        misc_matrix.append(p_misc)
        
    return llm_matrix, retrieval_matrix, misc_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faiss", required=True)
    parser.add_argument("--ragacc", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    faiss_llm, faiss_ret, faiss_misc = parse_csv(args.faiss)
    ragacc_llm, ragacc_ret, ragacc_misc = parse_csv(args.ragacc)

    plot_batch(faiss_llm, faiss_ret, faiss_misc, ragacc_llm, ragacc_ret, ragacc_misc, args.output)
    print(f"Plot saved to {args.output}")

if __name__ == '__main__':
    main()
