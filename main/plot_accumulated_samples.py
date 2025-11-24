"""
Plot Accumulated Training Samples vs Accuracy
Plot accuracy as a function of accumulated training samples for different methods.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100


def plot_accumulated_samples_vs_accuracy(
    csv_path,
    output_path="accumulated_samples_vs_accuracy.png",
    title="Accuracy vs Accumulated Training Samples",
    figsize=(10, 6),
    methods=None
):
    """
    Plot accuracy as a function of accumulated training samples.
    
    Args:
        csv_path: Path to CSV file with results
        output_path: Where to save the plot
        title: Plot title
        figsize: Figure size (width, height)
        methods: List of methods to plot (default: None = all methods)
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Check if required columns exist
    required_columns = ['method', 'round', 'test_accuracy', 'accumulated_training_samples']
    if not all(col in df.columns for col in required_columns):
        print(f"ERROR: Missing required columns in CSV")
        print(f"Required: {required_columns}")
        print(f"Available: {df.columns.tolist()}")
        return None
    
    # Filter by methods if specified
    if methods is not None:
        df = df[df['method'].isin(methods)]
    
    # Get unique methods
    unique_methods = df['method'].unique()
    if len(unique_methods) == 0:
        print("ERROR: No methods found in CSV")
        return None
    
    # Color and style mapping for methods
    method_colors = {
        'FedSTS': '#1f77b4',           # Blue
        'FedSTaS_no_DP': '#ff7f0e',    # Orange
        'FedSTaS_DP': '#2ca02c',       # Green
    }
    
    method_linestyles = {
        'FedSTS': '-',
        'FedSTaS_no_DP': '--',
        'FedSTaS_DP': '-.',
    }
    
    method_markers = {
        'FedSTS': 'o',
        'FedSTaS_no_DP': 's',
        'FedSTaS_DP': '^',
    }
    
    method_labels = {
        'FedSTS': 'FedSTS',
        'FedSTaS_no_DP': 'FedSTaS (no-DP)',
        'FedSTaS_DP': 'FedSTaS (with DP)',
    }
    
    # Auto-assign colors if method not in predefined map
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, method in enumerate(unique_methods):
        if method not in method_colors:
            method_colors[method] = default_colors[i % len(default_colors)]
            method_linestyles[method] = '-'
            method_markers[method] = 'o'
            method_labels[method] = method
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each method
    for method in unique_methods:
        method_df = df[df['method'] == method].sort_values('round')
        
        # Extract data
        accumulated_samples = method_df['accumulated_training_samples'].values
        accuracy = method_df['test_accuracy'].values * 100  # Convert to percentage
        
        # Plot
        ax.plot(
            accumulated_samples,
            accuracy,
            marker=method_markers.get(method, 'o'),
            markersize=6,
            linewidth=2,
            color=method_colors.get(method, default_colors[0]),
            linestyle=method_linestyles.get(method, '-'),
            label=method_labels.get(method, method),
            markevery=max(1, len(accumulated_samples) // 20),  # Show markers every ~5% of points
            alpha=0.8
        )
    
    # Formatting
    ax.set_xlabel('Accumulated Training Samples', fontsize=13)
    ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend
    ax.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='gray')
    
    # Format x-axis to show numbers in readable format (e.g., 50K, 100K)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'))
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    # Show
    plt.show()
    
    return fig, ax


def plot_comparison_rounds_vs_samples(
    csv_path,
    output_path="rounds_vs_samples_comparison.png",
    title="Comparison: Rounds vs Accumulated Samples",
    figsize=(14, 5)
):
    """
    Create a side-by-side comparison: rounds vs accuracy and accumulated samples vs accuracy.
    
    Args:
        csv_path: Path to CSV file with results
        output_path: Where to save the plot
        title: Plot title
        figsize: Figure size (width, height)
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Check if required columns exist
    required_columns = ['method', 'round', 'test_accuracy', 'accumulated_training_samples']
    if not all(col in df.columns for col in required_columns):
        print(f"ERROR: Missing required columns in CSV")
        print(f"Required: {required_columns}")
        print(f"Available: {df.columns.tolist()}")
        return None
    
    # Get unique methods
    unique_methods = df['method'].unique()
    
    # Color and style mapping
    method_colors = {
        'FedSTS': '#1f77b4',
        'FedSTaS_no_DP': '#ff7f0e',
        'FedSTaS_DP': '#2ca02c',
    }
    
    method_linestyles = {
        'FedSTS': '-',
        'FedSTaS_no_DP': '--',
        'FedSTaS_DP': '-.',
    }
    
    method_markers = {
        'FedSTS': 'o',
        'FedSTaS_no_DP': 's',
        'FedSTaS_DP': '^',
    }
    
    method_labels = {
        'FedSTS': 'FedSTS',
        'FedSTaS_no_DP': 'FedSTaS (no-DP)',
        'FedSTaS_DP': 'FedSTaS (with DP)',
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Rounds vs Accuracy
    for method in unique_methods:
        method_df = df[df['method'] == method].sort_values('round')
        rounds = method_df['round'].values
        accuracy = method_df['test_accuracy'].values * 100
        
        ax1.plot(
            rounds,
            accuracy,
            marker=method_markers.get(method, 'o'),
            markersize=5,
            linewidth=2,
            color=method_colors.get(method, '#1f77b4'),
            linestyle=method_linestyles.get(method, '-'),
            label=method_labels.get(method, method),
            markevery=max(1, len(rounds) // 20),
            alpha=0.8
        )
    
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy vs Rounds', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', frameon=True, framealpha=0.95)
    
    # Plot 2: Accumulated Samples vs Accuracy
    for method in unique_methods:
        method_df = df[df['method'] == method].sort_values('round')
        accumulated_samples = method_df['accumulated_training_samples'].values
        accuracy = method_df['test_accuracy'].values * 100
        
        ax2.plot(
            accumulated_samples,
            accuracy,
            marker=method_markers.get(method, 'o'),
            markersize=5,
            linewidth=2,
            color=method_colors.get(method, '#1f77b4'),
            linestyle=method_linestyles.get(method, '-'),
            label=method_labels.get(method, method),
            markevery=max(1, len(accumulated_samples) // 20),
            alpha=0.8
        )
    
    ax2.set_xlabel('Accumulated Training Samples', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy vs Accumulated Samples', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', frameon=True, framealpha=0.95)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'))
    
    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {output_path}")
    
    # Show
    plt.show()
    
    return fig, (ax1, ax2)


# ============================================================================
# Main script
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot accumulated training samples vs accuracy")
    parser.add_argument("csv", type=str, help="Path to CSV file with results")
    parser.add_argument("--output", type=str, default="accumulated_samples_vs_accuracy.png",
                        help="Output plot filename")
    parser.add_argument("--title", type=str, default="Accuracy vs Accumulated Training Samples",
                        help="Plot title")
    parser.add_argument("--comparison", action="store_true",
                        help="Create side-by-side comparison (rounds vs accumulated samples)")
    parser.add_argument("--methods", nargs='+', type=str, default=None,
                        help="Methods to plot (e.g., --methods FedSTS FedSTaS_no_DP)")
    
    args = parser.parse_args()
    
    if args.comparison:
        # Create comparison plot
        plot_comparison_rounds_vs_samples(
            csv_path=args.csv,
            output_path=args.output,
            title=args.title
        )
    else:
        # Create single plot
        plot_accumulated_samples_vs_accuracy(
            csv_path=args.csv,
            output_path=args.output,
            title=args.title,
            methods=args.methods
        )
    
    print("\n✅ Done!")

