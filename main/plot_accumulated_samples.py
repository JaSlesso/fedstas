"""
Plot Accumulated Training Samples vs Accuracy and Macro F1
Matches the exact style and layout of main_plot.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def load_results_from_csv(csv_file):
    """Load training results from a single CSV file containing all methods."""
    results = {}
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return results
    
    try:
        # Load CSV file
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        required_columns = ['method', 'round', 'test_accuracy', 'accumulated_training_samples']
        if not all(col in df.columns for col in required_columns):
            print(f"Missing required columns in {csv_file}.")
            print(f"Required: {required_columns}")
            print(f"Available: {df.columns.tolist()}")
            return results
        
        # Map method names to display names (matching main_plot.py)
        method_map = {
            'FedSTS': 'FedSTS',
            'FedSTaS_no_DP': 'FedSTaS (no DP)',
            'FedSTaS_DP': 'FedSTaS (with DP)'
        }
        
        # Get unique methods in the CSV
        unique_methods = df['method'].unique()
        print(f"Found methods in CSV: {unique_methods.tolist()}")
        
        # Extract data for each method
        for method in unique_methods:
            method_df = df[df['method'] == method].sort_values('round')
            
            display_name = method_map.get(method, method)
            results[display_name] = {
                'rounds': method_df['round'].tolist(),
                'accumulated_samples': method_df['accumulated_training_samples'].tolist(),
                'test_acc': method_df['test_accuracy'].tolist(),
                'macro_f1': method_df['macro_f1'].tolist() if 'macro_f1' in method_df.columns else None
            }
            print(f"Loaded {len(method_df)} rounds for {display_name}")
        
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
    
    return results


def plot_accumulated_samples_comparison(results, output_file, skip_points=1):
    """
    Plot comparison of accumulated samples vs test accuracy and macro F1.
    Matches the exact style of main_plot.py
    """
    if not results:
        print("No results to plot. Ensure valid CSV file is present.")
        return
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Accumulated Samples vs Test Accuracy
    plt.subplot(1, 2, 1)
    for method, data in results.items():
        if 'test_acc' in data and len(data['test_acc']) > 0:
            accumulated_samples = data.get('accumulated_samples', [])
            x_values = accumulated_samples[::skip_points]
            y_values = data['test_acc'][::skip_points]
            plt.plot(x_values, y_values, '-', linewidth=2, label=method, marker='o', markersize=4)
    
    plt.title('Test Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Accumulated Training Samples', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=10)
    
    # Plot 2: Accumulated Samples vs Macro F1
    plt.subplot(1, 2, 2)
    for method, data in results.items():
        if 'macro_f1' in data and data['macro_f1'] is not None and len(data['macro_f1']) > 0:
            accumulated_samples = data.get('accumulated_samples', [])
            x_values = accumulated_samples[::skip_points]
            y_values = data['macro_f1'][::skip_points]
            plt.plot(x_values, y_values, '-', linewidth=2, label=method, marker='o', markersize=4)
    
    plt.title('Macro F1', fontsize=14, fontweight='bold')
    plt.xlabel('Accumulated Training Samples', fontsize=12)
    plt.ylabel('Macro F1', fontsize=12)
    plt.ylim(0, 0.45)  # Cap y-axis at 0.45
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Save plot
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"\nSaved comparison plot to {output_file}")
    plt.close()




def main():
    parser = argparse.ArgumentParser(description='Plot accumulated samples vs accuracy and macro F1')
    parser.add_argument('--csv', type=str, default='cifar10_beta_eps_results.csv',
                        help='Input CSV file containing results (default: cifar10_beta_eps_results.csv)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output plot file path (default: plots/accumulated_samples_comparison.png)')
    parser.add_argument('--skip_points', type=int, default=1,
                        help='Skip points in plot for clarity (default: 1, no skipping)')
    
    args = parser.parse_args()
    
    # Set default output if not specified
    if args.output is None:
        csv_basename = os.path.splitext(os.path.basename(args.csv))[0]
        args.output = f'plots/{csv_basename}_accumulated_samples.png'
    
    print("="*70)
    print(f"Generating accumulated samples comparison plots...")
    print(f"  Input CSV: {args.csv}")
    print(f"  Output Plot: {args.output}")
    print(f"  Skip Points: {args.skip_points}")
    print("="*70)
    print()
    
    # Load results from CSV file
    results = load_results_from_csv(args.csv)
    
    # Plot comparison
    if results:
        plot_accumulated_samples_comparison(results, args.output, args.skip_points)
        print("\nPlotting completed successfully!")
    else:
        print("\nNo valid results found. Please check your CSV file.")
        print(f"\nExpected CSV file: {args.csv}")
        print(f"Required CSV columns: method, round, test_accuracy, accumulated_training_samples, macro_f1")

if __name__ == "__main__":
    main()
#python plot_accumulated_samples.py --csv results_beta0.1_300_all_new.csv --output result/new_plot.png
