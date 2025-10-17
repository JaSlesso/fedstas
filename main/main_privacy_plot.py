"""
Privacy-Utility Tradeoff Plot
Plot Macro-F1 vs privacy budget (epsilon) for different clip cap values (M).
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

def compute_macro_f1(df_group):
    """
    Get Macro-F1 from CSV (uses the final round's macro_f1 score).
    """
    # Use the final round's macro_f1 (already computed in eval)
    if 'macro_f1' in df_group.columns:
        final_macro_f1 = df_group['macro_f1'].iloc[-1]
        return final_macro_f1 * 100  # Convert to percentage
    else:
        # Fallback to accuracy if macro_f1 not available
        final_acc = df_group['test_accuracy'].iloc[-1]
        return final_acc * 100


def plot_privacy_utility_tradeoff(
    csv_path,
    output_path="privacy_utility_tradeoff.png",
    metric="macro_f1",
    title="Privacy-Utility Tradeoff (β=0.5)",
    centralization_score=None,
    figsize=(8, 5.5),
    M_values=None,
    epsilon_values=None
):
    """
    Plot privacy-utility tradeoff (Macro-F1 vs epsilon for different M values).
    
    Args:
        csv_path: Path to CSV file with results
        output_path: Where to save the plot
        metric: Metric column name (default: "macro_f1")
        title: Plot title
        centralization_score: Optional baseline score for centralized training
        figsize: Figure size (width, height)
        M_values: List of M values to plot (default: None = all)
        epsilon_values: List of epsilon values to plot (default: None = all)
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter to only DP runs (epsilon not null)
    df_dp = df[df['epsilon'].notna()].copy()
    
    # Filter by M_values and epsilon_values if specified
    if M_values is not None:
        df_dp = df_dp[df_dp['M'].isin(M_values)]
    if epsilon_values is not None:
        df_dp = df_dp[df_dp['epsilon'].isin(epsilon_values)]
    
    # Get no-DP baseline if available
    df_no_dp = df[(df['epsilon'].isna()) & (df['method'] == 'FedSTaS_no_DP')]
    
    if len(df_no_dp) == 0:
        print("WARNING: No-DP baseline not found in CSV")
        no_dp_score = None
    else:
        # Group by run and take final round
        no_dp_groups = df_no_dp.groupby(['beta', 'n_star', 'M'])
        no_dp_score = compute_macro_f1(df_no_dp)
        print(f"No-DP FedSTaS Macro-F1: {no_dp_score:.2f}%")
    
    # Group by M and epsilon
    grouped = df_dp.groupby(['M', 'epsilon'])
    
    # Compute metric for each group (using final round)
    results = []
    for (M, eps), group in grouped:
        score = compute_macro_f1(group)
        results.append({
            'M': M,
            'epsilon': eps,
            'score': score
        })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("ERROR: No DP results found in CSV")
        return
    
    # Get unique M values and sort
    M_values = sorted(results_df['M'].unique())
    
    # Create color map for M values (matching the uploaded graph style)
    colors = {
        10: '#d62728',      # Red
        100: '#1f77b4',     # Blue
        500: '#ff7f0e',     # Orange (added)
        1000: '#ff7f0e',    # Orange
        2000: '#2ca02c',    # Green (added)
        5000: '#9467bd',    # Purple (added)
        10000: '#9467bd',   # Purple
    }
    
    markers = {
        10: 's',      # Square
        100: 's',     # Square
        500: 'o',     # Circle
        1000: 'o',    # Circle
        2000: 'D',    # Diamond
        5000: 'D',    # Diamond
        10000: 'D',   # Diamond
    }
    
    # Auto-assign colors if M not in predefined map
    default_colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
    for i, M in enumerate(M_values):
        if M not in colors:
            colors[M] = default_colors[i % len(default_colors)]
            markers[M] = 'o'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot lines for each M value
    for M in M_values:
        data = results_df[results_df['M'] == M].sort_values('epsilon', ascending=False)
        
        # Format M value for legend (use scientific notation for large M)
        if M >= 1000:
            M_label = f"$M = 10^{int(np.log10(M))}$" if M in [10, 100, 1000, 10000] else f"$M = {M}$"
        else:
            M_label = f"$M = {M}$"
        
        ax.plot(
            data['epsilon'], 
            data['score'],
            marker=markers.get(M, 'o'),
            markersize=8,
            linewidth=2,
            color=colors[M],
            label=M_label,
            linestyle='-',
            markeredgewidth=1.5,
            markeredgecolor='white'
        )
    
    # Add baseline lines
    xlim = ax.get_xlim()
    
    if no_dp_score is not None:
        ax.axhline(
            y=no_dp_score, 
            color='green', 
            linestyle='--', 
            linewidth=2.5,
            label='FedSTaS (no-DP)',
            alpha=0.8,
            zorder=1
        )
    
    if centralization_score is not None:
        ax.axhline(
            y=centralization_score, 
            color='black', 
            linestyle='--', 
            linewidth=2,
            label='Centralization',
            zorder=1
        )
    
    # Formatting
    ax.set_xlabel('privacy budget $\\varepsilon$', fontsize=13)
    ax.set_ylabel('Macro-F1', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Reverse x-axis (higher epsilon = less privacy on left)
    ax.invert_xaxis()
    
    # Set x-axis to show all epsilon values
    epsilon_values = sorted(results_df['epsilon'].unique(), reverse=True)
    ax.set_xticks(epsilon_values)
    ax.set_xticklabels([str(int(e) if e >= 1 else e) for e in epsilon_values])
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend (top right, matching uploaded graph)
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='gray')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    # Show
    plt.show()
    
    return fig, ax


def plot_from_multiple_csvs(
    csv_paths,
    output_path="privacy_utility_combined.png",
    labels=None,
    centralization_score=None,
    fedadam_score=None
):
    """
    Plot privacy-utility tradeoff from multiple CSV files on the same plot.
    Useful for comparing different beta values or methods.
    
    Args:
        csv_paths: List of CSV file paths
        output_path: Where to save the plot
        labels: List of labels for each CSV (e.g., ["β=0.1", "β=0.5", "IID"])
        centralization_score: Optional baseline score
        fedadam_score: Optional baseline score
    """
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(csv_paths))]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color cycle for different runs
    colors_cycle = plt.cm.tab10(np.linspace(0, 1, len(csv_paths)))
    
    for i, (csv_path, label) in enumerate(zip(csv_paths, labels)):
        df = pd.read_csv(csv_path)
        df_dp = df[df['epsilon'].notna()].copy()
        
        # Group by M and epsilon
        grouped = df_dp.groupby(['M', 'epsilon'])
        results = []
        for (M, eps), group in grouped:
            score = compute_macro_f1(group)
            results.append({'M': M, 'epsilon': eps, 'score': score})
        
        results_df = pd.DataFrame(results)
        
        # Get unique M values
        M_values = sorted(results_df['M'].unique())
        
        # Plot each M with different line style
        linestyles = ['-', '--', '-.', ':']
        for j, M in enumerate(M_values):
            data = results_df[results_df['M'] == M].sort_values('epsilon', ascending=False)
            ax.plot(
                data['epsilon'],
                data['score'],
                marker='o',
                markersize=6,
                linewidth=2,
                color=colors_cycle[i],
                linestyle=linestyles[j % len(linestyles)],
                label=f"{label}, M={M}",
                alpha=0.8
            )
    
    # Add baselines
    if centralization_score is not None:
        ax.axhline(y=centralization_score, color='black', linestyle='--', linewidth=2, label='Centralization')
    if fedadam_score is not None:
        ax.axhline(y=fedadam_score, color='gray', linestyle='--', linewidth=2, label='FedAdam')
    
    ax.set_xlabel('privacy budget $\\varepsilon$', fontsize=13)
    ax.set_ylabel('Macro-F1', fontsize=13)
    ax.set_title('Privacy-Utility Tradeoff Comparison', fontsize=14, fontweight='bold')
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Combined plot saved to: {output_path}")
    plt.show()
    
    return fig, ax


# ============================================================================
# Main script
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot privacy-utility tradeoff")
    parser.add_argument("csv", type=str, help="Path to CSV file with results")
    parser.add_argument("--output", type=str, default="privacy_utility_tradeoff.png", 
                        help="Output plot filename")
    parser.add_argument("--title", type=str, default="Privacy-Utility Tradeoff (β=0.5)",
                        help="Plot title")
    parser.add_argument("--centralization", type=float, default=None,
                        help="Centralized training Macro-F1 baseline")
    parser.add_argument("--M", nargs='+', type=int, default=None,
                        help="M values to plot (e.g., --M 500 1000)")
    parser.add_argument("--epsilon", nargs='+', type=float, default=None,
                        help="Epsilon values to plot (e.g., --epsilon 3 5 7)")
    
    args = parser.parse_args()
    
    # Single CSV plot
    plot_privacy_utility_tradeoff(
        csv_path=args.csv,
        output_path=args.output,
        title=args.title,
        centralization_score=args.centralization,
        M_values=args.M,
        epsilon_values=args.epsilon
    )
    
    print("\n✅ Done!")

