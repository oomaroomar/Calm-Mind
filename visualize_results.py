#!/usr/bin/env python3
"""Visualize training results showing winrate changes over time."""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_training_history(history_file: str = "eval_results/training_history.json"):
    """Load training history from JSON file."""
    with open(history_file, 'r') as f:
        return json.load(f)


def plot_winrates(history: list, save_dir: str = "eval_results"):
    """Create matplotlib graphs showing winrate changes over time.
    
    Args:
        history: List of evaluation results with timesteps and winrates
        save_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    output_dir = Path(save_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract data
    timesteps = [entry["timesteps"] for entry in history]
    random_winrates = [entry["winrates"]["Random"] for entry in history]
    maxpower_winrates = [entry["winrates"]["MaxBasePower"] for entry in history]
    heuristics_winrates = [entry["winrates"]["SimpleHeuristics"] for entry in history]
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create individual plots for each opponent
    opponents = [
        ("Random", random_winrates, "blue"),
        ("MaxBasePower", maxpower_winrates, "green"),
        ("SimpleHeuristics", heuristics_winrates, "red")
    ]
    
    for opponent_name, winrates, color in opponents:
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, winrates, marker='o', linewidth=2, 
                markersize=8, color=color, label=f'vs {opponent_name}')
        plt.xlabel('Training Timesteps', fontsize=12)
        plt.ylabel('Winrate', fontsize=12)
        plt.title(f'Model Winrate vs {opponent_name} Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.ylim(0, 1)
        
        # Add horizontal line at 50% winrate
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% Winrate')
        
        output_file = output_dir / f"winrate_vs_{opponent_name.lower()}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
        plt.close()
    
    # Create combined plot with all opponents
    plt.figure(figsize=(12, 7))
    plt.plot(timesteps, random_winrates, marker='o', linewidth=2, 
            markersize=8, color='blue', label='vs Random')
    plt.plot(timesteps, maxpower_winrates, marker='s', linewidth=2, 
            markersize=8, color='green', label='vs MaxBasePower')
    plt.plot(timesteps, heuristics_winrates, marker='^', linewidth=2, 
            markersize=8, color='red', label='vs SimpleHeuristics')
    
    plt.xlabel('Training Timesteps', fontsize=12)
    plt.ylabel('Winrate', fontsize=12)
    plt.title('Model Winrate vs All Opponents Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    plt.ylim(0, 1)
    
    # Add horizontal line at 50% winrate
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    output_file = output_dir / "winrate_combined.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to {output_file}")
    plt.close()
    
    # Create a summary statistics plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (opponent_name, winrates, color) in enumerate(opponents):
        ax = axes[idx]
        ax.plot(timesteps, winrates, marker='o', linewidth=2, 
               markersize=6, color=color)
        ax.set_xlabel('Training Timesteps', fontsize=10)
        ax.set_ylabel('Winrate', fontsize=10)
        ax.set_title(f'vs {opponent_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add final winrate annotation
        if winrates:
            final_winrate = winrates[-1]
            ax.annotate(f'Final: {final_winrate:.2f}', 
                       xy=(timesteps[-1], final_winrate),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                       fontsize=9)
    
    plt.suptitle('Training Progress Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / "winrate_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot to {output_file}")
    plt.close()
    
    print("\nTraining Statistics:")
    print(f"Total evaluations: {len(history)}")
    print(f"Total timesteps: {timesteps[-1] if timesteps else 0}")
    print("\nFinal Winrates:")
    if history:
        final = history[-1]["winrates"]
        print(f"  vs Random: {final['Random']:.2%}")
        print(f"  vs MaxBasePower: {final['MaxBasePower']:.2%}")
        print(f"  vs SimpleHeuristics: {final['SimpleHeuristics']:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument(
        "--history_file",
        type=str,
        default="eval_results/training_history.json",
        help="Path to training history JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save plots"
    )
    args = parser.parse_args()
    
    # Load and plot results
    try:
        history = load_training_history(args.history_file)
        if not history:
            print("No training history found. Train your model first!")
            return
        
        plot_winrates(history, args.output_dir)
        print("\nVisualization complete!")
        
    except FileNotFoundError:
        print(f"Error: Training history file not found at {args.history_file}")
        print("Make sure to train your model first with --train_selfplay flag.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

