#!/usr/bin/env python3
"""
Visualization script for checkpoint evaluation results.
Reads evaluation_results.json files from multiple checkpoint directories
and generates box plots for comparison.
"""

import os
import json
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set matplotlib style for research papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.5,
    'patch.facecolor': 'lightgray'
})


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    runs_directory: str
    output_directory: str = "evaluation_plots"
    plot_title: str = "Performance Comparison Across Training Configurations"
    figure_size: Tuple[int, int] = (10, 6)
    save_formats: List[str] = None
    dpi: int = 300
    show_stats_annotations: bool = False  # For cleaner research paper style
    color_palette: str = "Set2"  # Professional color palette
    
    def __post_init__(self):
        if self.save_formats is None:
            self.save_formats = ['pdf', 'png', 'svg']


def find_evaluation_results(runs_directory: str) -> Dict[str, str]:
    """
    Find all evaluation_results.json files in subdirectories.
    
    Args:
        runs_directory: Path to directory containing run folders
        
    Returns:
        Dictionary mapping run names to evaluation results file paths
    """
    results_files = {}
    
    # Pattern to find evaluation_results.json in any subdirectory
    pattern = os.path.join(runs_directory, "**/evaluation_results.json")
    
    for results_file in glob.glob(pattern, recursive=True):
        # Extract run name from directory structure
        run_dir = os.path.dirname(results_file)
        run_name = os.path.basename(run_dir)
        
        # If the results file is deeper (e.g., in runs/run_name/models/evaluation_results.json)
        # we might want to use a more descriptive name
        if run_name in ['models', 'checkpoints']:
            run_name = os.path.basename(os.path.dirname(run_dir))
        
        results_files[run_name] = results_file
    
    return results_files


def load_evaluation_data(results_files: Dict[str, str]) -> pd.DataFrame:
    """
    Load evaluation data from JSON files into a pandas DataFrame.
    
    Args:
        results_files: Dictionary mapping run names to file paths
        
    Returns:
        DataFrame with columns: 'run_name', 'episode', 'return', 'configuration'
    """
    data = []
    
    for run_name, file_path in results_files.items():
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            episodic_returns = results.get('episodic_returns', [])
            
            # Extract configuration info from run name for better labeling
            config_name = extract_config_name(run_name)
            
            for episode, episode_return in enumerate(episodic_returns):
                data.append({
                    'run_name': run_name,
                    'configuration': config_name,
                    'episode': episode,
                    'return': episode_return
                })
                
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    return pd.DataFrame(data)


def extract_config_name(run_name: str) -> str:
    """
    Extract a clean configuration name from run directory name.
    
    Args:
        run_name: Original run directory name
        
    Returns:
        Clean configuration name for display
    """
    # Handle common naming patterns
    if 'grad-compression=None' in run_name:
        return 'No Compression'
    elif 'grad-compression=stats' in run_name:
        return 'Stats-based Compression'
    elif 'accumulate-grads' in run_name:
        return 'Gradient Accumulation'
    elif 'baseline' in run_name.lower():
        return 'Baseline'
    else:
        # Extract meaningful parts or use timestamp
        parts = run_name.split('-')
        if len(parts) > 1:
            return ' '.join(part.title() for part in parts[1:] if not part.isdigit())
        return run_name


def create_box_plot(df: pd.DataFrame, config: VisualizationConfig):
    """
    Create and save research-style box plot comparing different configurations.
    
    Args:
        df: DataFrame with evaluation data
        config: Visualization configuration
    """
    # Create figure with research paper proportions
    fig, ax = plt.subplots(figsize=config.figure_size)
    
    # Set color palette for research papers (colorblind-friendly)
    colors = sns.color_palette(config.color_palette, n_colors=len(df['configuration'].unique()))
    
    # Create box plot with professional styling
    bp = ax.boxplot(
        [df[df['configuration'] == config]['return'].values for config in df['configuration'].unique()],
        labels=df['configuration'].unique(),
        patch_artist=True,
        showmeans=True,
        meanline=False,
        meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6},
        medianprops={"color": "black", "linewidth": 2},
        boxprops={"linewidth": 1.5, "alpha": 0.8},
        whiskerprops={"linewidth": 1.5, "color": "black"},
        capprops={"linewidth": 1.5, "color": "black"},
        flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 4, "alpha": 0.7}
    )
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Customize axes
    ax.set_xlabel('Training Configuration', fontweight='bold')
    ax.set_ylabel('Episodic Return', fontweight='bold')
    ax.set_title(config.plot_title, pad=20, fontweight='bold')
    
    # Rotate x-axis labels if needed
    if any(len(label.get_text()) > 10 for label in ax.get_xticklabels()):
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    
    # Add statistical information if requested
    if config.show_stats_annotations:
        configurations = df['configuration'].unique()
        for i, config_name in enumerate(configurations):
            config_data = df[df['configuration'] == config_name]['return']
            n = len(config_data)
            mean_val = config_data.mean()
            
            # Add sample size annotation
            ax.text(i + 1, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                   f'n={n}', ha='center', va='bottom', fontsize=9)
    
    # Add statistical summary table below the plot
    summary_data = []
    for config_name in df['configuration'].unique():
        config_data = df[df['configuration'] == config_name]['return']
        summary_data.append([
            config_name,
            f"{config_data.mean():.2f}",
            f"{config_data.std():.2f}",
            f"{config_data.median():.2f}",
            f"{len(config_data)}"
        ])
    
    # Create table
    table = ax.table(
        cellText=summary_data,
        colLabels=['Configuration', 'Mean', 'Std', 'Median', 'n'],
        cellLoc='center',
        loc='bottom',
        bbox=[0.0, -0.4, 1.0, 0.25]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#E6E6E6')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#F8F8F8')
    
    # Adjust layout to accommodate table
    plt.subplots_adjust(bottom=0.3)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save in different formats
    os.makedirs(config.output_directory, exist_ok=True)
    for fmt in config.save_formats:
        output_file = os.path.join(config.output_directory, f'performance_comparison.{fmt}')
        plt.savefig(output_file, dpi=config.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved plot: {output_file}")
    
    plt.show()


def create_summary_statistics(df: pd.DataFrame, config: VisualizationConfig):
    """
    Create and save comprehensive summary statistics table.
    
    Args:
        df: DataFrame with evaluation data
        config: Visualization configuration
    """
    # Calculate summary statistics including quartiles
    summary = df.groupby('configuration')['return'].agg([
        'count', 'mean', 'std', 'min', 
        lambda x: np.percentile(x, 25),  # Q1
        'median', 
        lambda x: np.percentile(x, 75),  # Q3
        'max'
    ]).round(3)
    
    # Rename columns for clarity
    summary.columns = ['Episodes', 'Mean', 'Std Dev', 'Min', 'Q1', 'Median', 'Q3', 'Max']
    
    # Add 95% confidence intervals for the mean
    conf_intervals = []
    for config_name in summary.index:
        config_data = df[df['configuration'] == config_name]['return']
        n = len(config_data)
        mean = config_data.mean()
        std = config_data.std()
        # 95% CI assuming normal distribution
        margin_error = 1.96 * (std / np.sqrt(n)) if n > 1 else 0
        conf_intervals.append(f"[{mean - margin_error:.2f}, {mean + margin_error:.2f}]")
    
    summary['95% CI'] = conf_intervals
    
    # Save to CSV
    os.makedirs(config.output_directory, exist_ok=True)
    summary_file = os.path.join(config.output_directory, 'summary_statistics.csv')
    summary.to_csv(summary_file)
    
    print(f"Summary statistics saved: {summary_file}")
    print("\nSummary Statistics:")
    print(summary)


def create_individual_plots(df: pd.DataFrame, config: VisualizationConfig):
    """
    Create individual distribution plots for each run.
    
    Args:
        df: DataFrame with evaluation data
        config: Visualization configuration
    """
    run_names = df['run_name'].unique()
    n_runs = len(run_names)
    
    if n_runs == 0:
        print("No data to plot!")
        return
    
    # Create subplots
    cols = min(3, n_runs)
    rows = (n_runs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_runs == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, run_name in enumerate(run_names):
        run_data = df[df['run_name'] == run_name]['return']
        
        # Create histogram with KDE
        axes[i].hist(run_data, bins=10, alpha=0.7, density=True, color='skyblue')
        
        # Add KDE line if we have enough data
        if len(run_data) > 1:
            sns.kdeplot(run_data, ax=axes[i], color='red', linewidth=2)
        
        axes[i].set_title(f'{run_name}\nMean: {run_data.mean():.2f} ± {run_data.std():.2f}')
        axes[i].set_xlabel('Episodic Return')
        axes[i].set_ylabel('Density')
        axes[i].grid(alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_runs, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    for fmt in config.save_formats:
        output_file = os.path.join(config.output_directory, f'individual_distributions.{fmt}')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved individual plots: {output_file}")
    
    plt.show()


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Generate research-style plots for checkpoint evaluation results")
    parser.add_argument("runs_directory", help="Directory containing run folders with evaluation results")
    parser.add_argument("--output-dir", default="evaluation_plots", help="Output directory for plots")
    parser.add_argument("--title", default="Performance Comparison Across Training Configurations", help="Plot title")
    parser.add_argument("--show-annotations", action="store_true", help="Show statistical annotations on plot")
    parser.add_argument("--format", nargs="+", default=["pdf", "png"], help="Output formats (pdf, png, svg)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = VisualizationConfig(
        runs_directory=args.runs_directory,
        output_directory=args.output_dir,
        plot_title=args.title,
        show_stats_annotations=args.show_annotations,
        save_formats=args.format
    )
    
    print(f"Looking for evaluation results in: {config.runs_directory}")
    
    # Find evaluation results
    results_files = find_evaluation_results(config.runs_directory)
    
    if not results_files:
        print("No evaluation_results.json files found!")
        print(f"Make sure you have run evaluations and saved results in {config.runs_directory}")
        return
    
    print(f"Found {len(results_files)} evaluation results:")
    for run_name, file_path in results_files.items():
        print(f"  - {run_name}: {file_path}")
    
    # Load data
    print("\nLoading evaluation data...")
    df = load_evaluation_data(results_files)
    
    if df.empty:
        print("No valid evaluation data found!")
        return
    
    print(f"Loaded data for {len(df['run_name'].unique())} runs, {len(df)} total episodes")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_box_plot(df, config)
    create_summary_statistics(df, config)
    create_individual_plots(df, config)
    
    print(f"\nAll visualizations saved to: {config.output_directory}")


if __name__ == "__main__":
    main()