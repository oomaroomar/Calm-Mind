#!/usr/bin/env python3
"""Visualize training results showing winrate changes over time."""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import numpy as np


def load_training_history(history_file: str = "eval_results/training_history.json"):
    """Load training history from JSON file."""
    with open(history_file, "r") as f:
        return json.load(f)


def _evenly_spaced_indices(length: int, k: int) -> list:
    """Return up to k evenly spaced indices over range(length)."""
    if length <= 0 or k <= 0:
        return []
    if k >= length:
        return list(range(length))
    # Include first and last indices and distribute the rest evenly
    return [round(i * (length - 1) / (k - 1)) for i in range(k)]


def _linear_fit(x_vals, y_vals):
    """Return slope and intercept of a linear fit if possible, else None."""
    if x_vals is None or y_vals is None:
        return None
    if len(x_vals) < 2 or len(y_vals) < 2:
        return None
    try:
        m, b = np.polyfit(
            np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float), 1
        )
        return m, b
    except Exception:
        return None


def plot_winrates(history: list, save_dir: str = "eval_results", num_points: int = 20):
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

    # Downsample to evenly spaced points for plotting
    idx = _evenly_spaced_indices(len(timesteps), num_points)
    plot_steps = [timesteps[i] for i in idx]
    random_plot = [random_winrates[i] for i in idx]
    maxpower_plot = [maxpower_winrates[i] for i in idx]
    heuristics_plot = [heuristics_winrates[i] for i in idx]

    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")

    # Create individual plots for each opponent
    opponents = [
        ("Random", random_plot, "blue"),
        ("MaxBasePower", maxpower_plot, "green"),
        ("SimpleHeuristics", heuristics_plot, "red"),
    ]

    for opponent_name, winrates, color in opponents:
        plt.figure(figsize=(10, 6))
        plt.plot(
            plot_steps,
            winrates,
            marker="o",
            linewidth=2,
            markersize=8,
            color=color,
            label=f"vs {opponent_name}",
        )
        # Linear regression fit
        fit = _linear_fit(plot_steps, winrates)
        if fit is not None:
            m, b = fit
            x_line = np.asarray(plot_steps, dtype=float)
            y_line = m * x_line + b
            plt.plot(
                x_line,
                y_line,
                linestyle="--",
                color=color,
                alpha=0.8,
                label=f"Fit vs {opponent_name}",
            )
        plt.xlabel("Training Timesteps", fontsize=12)
        plt.ylabel("Winrate", fontsize=12)
        plt.title(
            f"Model Winrate vs {opponent_name} Over Time",
            fontsize=14,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.ylim(0, 1)

        # Add horizontal line at 50% winrate
        plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% Winrate")

        output_file = output_dir / f"winrate_vs_{opponent_name.lower()}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_file}")
        plt.close()

    # Create combined plot with all opponents
    plt.figure(figsize=(12, 7))
    plt.plot(
        plot_steps,
        random_plot,
        marker="o",
        linewidth=2,
        markersize=8,
        color="blue",
        label="vs Random",
    )
    fit = _linear_fit(plot_steps, random_plot)
    if fit is not None:
        m, b = fit
        x_line = np.asarray(plot_steps, dtype=float)
        y_line = m * x_line + b
        plt.plot(
            x_line,
            y_line,
            linestyle="--",
            color="blue",
            alpha=0.8,
            label="Fit vs Random",
        )
    plt.plot(
        plot_steps,
        maxpower_plot,
        marker="s",
        linewidth=2,
        markersize=8,
        color="green",
        label="vs MaxBasePower",
    )
    fit = _linear_fit(plot_steps, maxpower_plot)
    if fit is not None:
        m, b = fit
        x_line = np.asarray(plot_steps, dtype=float)
        y_line = m * x_line + b
        plt.plot(
            x_line,
            y_line,
            linestyle="--",
            color="green",
            alpha=0.8,
            label="Fit vs MaxBasePower",
        )
    plt.plot(
        plot_steps,
        heuristics_plot,
        marker="^",
        linewidth=2,
        markersize=8,
        color="red",
        label="vs SimpleHeuristics",
    )
    fit = _linear_fit(plot_steps, heuristics_plot)
    if fit is not None:
        m, b = fit
        x_line = np.asarray(plot_steps, dtype=float)
        y_line = m * x_line + b
        plt.plot(
            x_line,
            y_line,
            linestyle="--",
            color="red",
            alpha=0.8,
            label="Fit vs SimpleHeuristics",
        )

    plt.xlabel("Training Timesteps", fontsize=12)
    plt.ylabel("Winrate", fontsize=12)
    plt.title(
        "Model Winrate vs All Opponents Over Time", fontsize=14, fontweight="bold"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc="best")
    plt.ylim(0, 1)

    # Add horizontal line at 50% winrate
    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    output_file = output_dir / "winrate_combined.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved combined plot to {output_file}")
    plt.close()

    # Create a summary statistics plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx_ax, (opponent_name, winrates, color) in enumerate(opponents):
        ax = axes[idx_ax]
        ax.plot(
            plot_steps, winrates, marker="o", linewidth=2, markersize=6, color=color
        )
        fit = _linear_fit(plot_steps, winrates)
        if fit is not None:
            m, b = fit
            x_line = np.asarray(plot_steps, dtype=float)
            y_line = m * x_line + b
            ax.plot(x_line, y_line, linestyle="--", color=color, alpha=0.8)
        ax.set_xlabel("Training Timesteps", fontsize=10)
        ax.set_ylabel("Winrate", fontsize=10)
        ax.set_title(f"vs {opponent_name}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

        # Add final winrate annotation
        if winrates:
            final_winrate = winrates[-1]
            ax.annotate(
                f"Final: {final_winrate:.2f}",
                xy=(plot_steps[-1], final_winrate),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
                fontsize=9,
            )

    plt.suptitle("Training Progress Summary", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_file = output_dir / "winrate_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
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
        help="Path to training history JSON file",
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_results", help="Directory to save plots"
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
