#!/usr/bin/env python3
"""
Visualize training logs from MMSegmentation JSON Lines log file.
Creates multiple plots for different metrics over training iterations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_log(log_path):
    """Load JSON Lines log file."""
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_metric(data, metric_key):
    """Extract a specific metric from the data."""
    steps = []
    values = []
    for entry in data:
        if metric_key in entry and 'iter' in entry:
            steps.append(entry['iter'])
            values.append(entry[metric_key])
    return np.array(steps), np.array(values)


def extract_val_metric(data, metric_key):
    """Extract a val metric using the preceding train entry's iter as x-axis."""
    steps = []
    values = []
    last_train_iter = None
    for entry in data:
        if entry.get('mode') == 'train':
            last_train_iter = entry.get('iter')
        elif entry.get('mode') == 'val' and metric_key in entry:
            if last_train_iter is not None:
                steps.append(last_train_iter)
            elif 'iter' in entry:
                steps.append(entry['iter'])
            values.append(entry[metric_key])
    return np.array(steps), np.array(values)


def plot_metric(ax, data, metric_key, title, ylabel, color='blue', smooth=False, window=50):
    """Plot a single metric on the given axis."""
    steps, values = extract_metric(data, metric_key)
    
    if len(steps) == 0:
        ax.text(0.5, 0.5, f'No data for {metric_key}', 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    ax.plot(steps, values, color=color, alpha=0.3 if smooth else 0.8, linewidth=1)
    
    if smooth and len(values) > window:
        smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
        smooth_steps = steps[window-1:]
        ax.plot(smooth_steps, smoothed, color=color, linewidth=2, label='Smoothed')
    
    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if smooth and len(values) > window:
        ax.legend(fontsize=8)


def plot_training_overview(data, save_path=None):
    """Plot training overview metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('UNetFormer + DAPCN Training Overview', fontsize=14, fontweight='bold')
    
    plot_metric(axes[0, 0], data, 'lr', 'Learning Rate', 'LR', color='tab:blue')
    plot_metric(axes[0, 1], data, 'loss', 'Total Loss', 'Loss', color='tab:red', smooth=True)
    plot_metric(axes[1, 0], data, 'decode.acc_seg', 'Segmentation Accuracy', 'Accuracy (%)', 
                color='tab:green', smooth=True)
    plot_metric(axes[1, 1], data, 'memory', 'GPU Memory Usage', 'Memory (MB)', color='tab:orange')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training overview to {save_path}")
    plt.show()


def plot_loss_components(data, save_path=None):
    """Plot all loss components in one figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Loss Components Over Training', fontsize=14, fontweight='bold')
    
    loss_configs = [
        ('loss', 'Total Loss', 'tab:blue'),
        ('decode.loss_seg', 'Segmentation Loss', 'tab:orange'),
        ('decode.loss_boundary', 'Boundary Loss', 'tab:green'),
        ('decode.loss_dapg', 'DAPG Loss', 'tab:red'),
        ('grad_norm', 'Gradient Norm', 'tab:purple'),
    ]
    
    for idx, (key, title, color) in enumerate(loss_configs):
        ax = axes[idx // 3, idx % 3]
        plot_metric(ax, data, key, title, 'Loss', color=color, smooth=True)
    
    ax = axes[1, 2]
    steps = None
    for key, label, color in [
        ('decode.dapg_loss_intra', 'Intra', 'tab:cyan'),
        ('decode.dapg_loss_inter', 'Inter', 'tab:pink'),
        ('decode.dapg_loss_quality', 'Quality', 'tab:olive'),
    ]:
        s, v = extract_metric(data, key)
        if len(s) > 0:
            ax.plot(s, v, label=label, alpha=0.7, linewidth=1)
            steps = s
    
    if steps is not None:
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title('DAPG Loss Components', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No DAPG sub-loss data', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss components plot to {save_path}")
    plt.show()


def plot_evaluation_metrics(data, save_path=None):
    """Plot evaluation metrics with mIoU and Accuracy on separate subplots."""
    fig, (ax_miou, ax_acc) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Evaluation Metrics', fontsize=14, fontweight='bold')

    miou_steps, miou_values = extract_val_metric(data, 'mIoU')
    if len(miou_steps) > 0:
        ax_miou.plot(miou_steps, miou_values, 'o-', color='tab:orange',
                     linewidth=2, markersize=6, label='mIoU')
        ax_miou.set_xlabel('Iteration', fontsize=12)
        ax_miou.set_ylabel('mIoU', fontsize=12)
        ax_miou.set_title('Mean IoU', fontsize=13, fontweight='bold')
        ax_miou.legend(fontsize=10)
        ax_miou.grid(True, alpha=0.3)
    else:
        ax_miou.text(0.5, 0.5, 'No mIoU data found',
                     ha='center', va='center', transform=ax_miou.transAxes, fontsize=12)

    acc_metrics = [
        ('aAcc', 'Overall Accuracy (aAcc)', 'tab:blue'),
        ('mAcc', 'Mean Accuracy (mAcc)', 'tab:green'),
    ]
    acc_has_data = False
    for key, label, color in acc_metrics:
        steps, values = extract_val_metric(data, key)
        if len(steps) > 0:
            ax_acc.plot(steps, values, 'o-', label=label, color=color,
                        linewidth=2, markersize=6)
            acc_has_data = True

    if acc_has_data:
        ax_acc.set_xlabel('Iteration', fontsize=12)
        ax_acc.set_ylabel('Accuracy', fontsize=12)
        ax_acc.set_title('Accuracy Metrics', fontsize=13, fontweight='bold')
        ax_acc.legend(fontsize=10)
        ax_acc.grid(True, alpha=0.3)
    else:
        ax_acc.text(0.5, 0.5, 'No accuracy data found',
                    ha='center', va='center', transform=ax_acc.transAxes, fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved evaluation metrics to {save_path}")
    plt.show()


def plot_time_metrics(data, save_path=None):
    """Plot timing metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Timing Metrics', fontsize=14, fontweight='bold')
    
    plot_metric(axes[0], data, 'data_time', 'Data Loading Time', 'Time (s)', 
                color='tab:blue', smooth=True, window=20)
    plot_metric(axes[1], data, 'time', 'Iteration Time', 'Time (s)', 
                color='tab:orange', smooth=True, window=20)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved time metrics to {save_path}")
    plt.show()


def plot_loss_vs_accuracy(data, save_path=None):
    """Plot loss vs accuracy phase space."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iters, losses = extract_metric(data, 'loss')
    _, accuracies = extract_metric(data, 'decode.acc_seg')
    
    if len(iters) > 0:
        scatter = ax.scatter(losses, accuracies, c=iters, cmap='viridis', s=30, alpha=0.7)
        ax.set_xlabel('Total Loss', fontsize=12)
        ax.set_ylabel('Segmentation Accuracy (%)', fontsize=12)
        ax.set_title('Loss vs Accuracy (color = iteration)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Iteration')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss vs accuracy plot to {save_path}")
    plt.show()


def create_all_plots(log_path, output_dir=None):
    """Create all plots from the log file."""
    print(f"Loading data from {log_path}...")
    data = load_log(log_path)
    print(f"Loaded {len(data)} log entries")
    
    train_data = [r for r in data if r.get('mode') == 'train']
    print(f"Found {len(train_data)} training records")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(log_path).parent / 'plots'
        output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating plots in {output_dir}...\n")
    
    plot_training_overview(train_data, output_dir / 'training_overview.png')
    plot_loss_components(train_data, output_dir / 'loss_components.png')
    plot_evaluation_metrics(data, output_dir / 'evaluation_metrics.png')
    plot_time_metrics(train_data, output_dir / 'time_metrics.png')
    plot_loss_vs_accuracy(train_data, output_dir / 'loss_vs_accuracy.png')
    
    print("\n✓ All plots generated successfully!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize MMSegmentation training logs')
    parser.add_argument('log_path', type=str, 
                        default='work_dirs/openearthmap/unetformer_train1000/20260402_032811.log.json',
                        nargs='?',
                        help='Path to .log.json file')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory for plots (default: plots/ next to log file)')
    parser.add_argument('--no-display', action='store_true',
                        help='Save plots without displaying (useful for servers)')
    
    args = parser.parse_args()
    
    if args.no_display:
        plt.switch_backend('Agg')
    
    create_all_plots(args.log_path, args.output)
