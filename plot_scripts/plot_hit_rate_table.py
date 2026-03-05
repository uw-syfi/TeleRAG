import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import json
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ragacc.pipeline_budgets import PREFETCH_BUDGET_DICT_22B, PREFETCH_BUDGET_DICT_LARGE, PREFETCH_BUDGET_DICT_SMALL

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_22b', default='evaluation/hit_rate/hit_rate_h100_22b.json')
    parser.add_argument('--json_8b', default='evaluation/hit_rate/hit_rate_h100_8b.json')
    parser.add_argument('--json_3b', default='evaluation/hit_rate/hit_rate_4090_3b.json')
    parser.add_argument('--output', default='figure/hit_rate_table.pdf')
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.json_22b, 'r') as f: d_22b = json.load(f)['nq']
    with open(args.json_8b, 'r') as f: d_8b = json.load(f)['nq']
    with open(args.json_3b, 'r') as f: d_3b = json.load(f)['nq']

    pipelines = ["linear", "parallel", "iterative", "iterretgen", "flare", "selfrag"]
    pipeline_names = ["HyDE", "SubQ", "Iter", "IRG", "FLARE", "S-RAG"]

    get_hr = lambda d, p: d[p]['overall_hit_rate']
    hr_22b = [get_hr(d_22b, p) for p in pipelines]
    hr_8b = [get_hr(d_8b, p) for p in pipelines]
    hr_3b = [get_hr(d_3b, p) for p in pipelines]

    b_22b = PREFETCH_BUDGET_DICT_22B["nq"]["h100"]
    b_8b = PREFETCH_BUDGET_DICT_LARGE["nq"]["h100"]
    b_3b = PREFETCH_BUDGET_DICT_SMALL["nq"]["rtx4090"]

    budgets_22b = [b_22b[p] for p in pipelines]
    budgets_8b = [b_8b[p] for p in pipelines]
    budgets_3b = [b_3b[p] for p in pipelines]

    plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(9, 3.5))
    ax = fig.add_axes([0.05, 0.25, 0.9, 0.7])
    ax.axis('off')

    x_starts = [0, 1.8, 4.4, 7.0, 9.6]
    y_top = 3.0
    rh = 0.35
    
    # Top horizontal line
    ax.plot([0, x_starts[-1]], [y_top, y_top], color='black', lw=1.5)
    
    # Main headers
    ax.text((x_starts[0]+x_starts[1])/2, y_top-rh*0.8, 'Pipeline', ha='center', va='center', fontsize=11)
    ax.text((x_starts[1]+x_starts[2])/2, y_top-rh*0.5, 'H100 (Mst-22B)', ha='center', va='center', fontsize=11)
    ax.text((x_starts[2]+x_starts[3])/2, y_top-rh*0.5, 'H100 (Llm3-8B)', ha='center', va='center', fontsize=11)
    ax.text((x_starts[3]+x_starts[4])/2, y_top-rh*0.5, '4090 (Llm3-3B)', ha='center', va='center', fontsize=11)
    
    # Sub headers
    for i in range(1, 4):
        ax.text(x_starts[i]+0.65, y_top-rh*1.5, 'Budget', ha='center', va='center', fontsize=11)
        ax.text(x_starts[i]+1.95, y_top-rh*1.5, 'Hit Rate', ha='center', va='center', fontsize=11)

    # Line below header
    ax.plot([0, x_starts[-1]], [y_top-rh*2, y_top-rh*2], color='black', lw=1)
    
    # Vertical lines
    for xs in x_starts[1:4]:
        ax.plot([xs, xs], [y_top, y_top-rh*(2+len(pipelines))], color='black', lw=1)
    
    # Colormap
    cmap = LinearSegmentedColormap.from_list('hit_rate', ['#ffffff', '#8bc37a'])
    
    y_data_start = y_top - rh*2
    
    def format_gb(v):
        if int(v) == v:
            return f"{int(v)} GB"
        return f"{v} GB"

    for i in range(len(pipelines)):
        y_cell = y_data_start - i*rh - rh/2
        
        # Pipeline name
        ax.text((x_starts[0]+x_starts[1])/2, y_cell, pipeline_names[i], ha='center', va='center', fontsize=11)
        
        # Draw cells
        for col_idx, (b, hr) in enumerate([
            (budgets_22b[i], hr_22b[i]), 
            (budgets_8b[i], hr_8b[i]), 
            (budgets_3b[i], hr_3b[i])
        ]):
            xs = x_starts[col_idx+1]
            
            # Budget text
            ax.text(xs+0.65, y_cell, format_gb(b), ha='center', va='center', fontsize=11)
            
            # Hit rate background
            rect = patches.Rectangle((xs+1.3, y_cell-rh/2), 1.3, rh, color=cmap(hr), ec='none', zorder=-1)
            ax.add_patch(rect)
            
            # Hit rate text
            ax.text(xs+1.95, y_cell, f"{hr*100:.1f}%", ha='center', va='center', fontsize=11)

    # Bottom line
    ax.plot([0, x_starts[-1]], [y_data_start - len(pipelines)*rh, y_data_start - len(pipelines)*rh], color='black', lw=1.5)
    
    ax.set_xlim(0, x_starts[-1])
    ax.set_ylim(y_data_start - len(pipelines)*rh, y_top)
    
    # Add Colorbar at the bottom
    cb_ax = fig.add_axes([0.15, 0.08, 0.7, 0.05])
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = mpl.colorbar.ColorbarBase(cb_ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    cb.outline.set_edgecolor('black')
    cb.outline.set_linewidth(1)
    cb.ax.tick_params(labelsize=11)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, bbox_inches='tight')
    print(f"Plot saved to {args.output}")

if __name__ == '__main__':
    main()
