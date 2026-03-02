"""Visualization tools for experiment analysis."""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from data.storage import TrajectoryStore
from .metrics import (
    compute_metrics,
    compare_experiments,
    compute_env_breakdown,
)


def setup_style():
    """Set up consistent plot styling."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12


def plot_behavior_distribution(
    store: TrajectoryStore,
    experiment_id: str,
    save_path: Optional[str] = None,
    show: bool = True,
    subdir: Optional[str] = None,
) -> plt.Figure:
    """Plot distribution of behavior labels for an experiment."""
    setup_style()

    metrics = compute_metrics(store, experiment_id, subdir=subdir)
    exp = store.load_experiment(experiment_id)

    labels = ["Aligned", "Covert", "Overrefusal", "Unclear"]
    values = [
        metrics.aligned_rate,
        metrics.covert_action_rate,
        metrics.overrefusal_rate,
        metrics.unclear_rate,
    ]
    colors = ["#2ecc71", "#e74c3c", "#f39c12", "#95a5a6"]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2)

    ax.set_ylabel("Rate")
    ax.set_title(f"Behavior Distribution\n{exp.model_name} on {', '.join(exp.env_names)}")
    ax.set_ylim(0, 1)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1%}",
            ha="center",
            fontsize=11,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_covert_rate_comparison(
    store: TrajectoryStore,
    experiment_ids: list[str],
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Compare covert action rates across experiments."""
    setup_style()

    df = compare_experiments(store, experiment_ids)

    fig, ax = plt.subplots()
    x = range(len(df))
    ax.bar(x, df["covert_rate"], color="#e74c3c", edgecolor="black")

    ax.set_ylabel("Covert Action Rate")
    ax.set_title("Covert Behavior Across Experiments")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['model']}\n{row['env']}" for _, row in df.iterrows()])
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_model_env_heatmap(
    store: TrajectoryStore,
    experiment_ids: list[str],
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot heatmap of covert action rate per model × environment."""
    setup_style()

    df = compute_env_breakdown(store, experiment_ids)

    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No environment data available", ha="center", va="center")
        return fig

    pivot = df.pivot_table(
        index="model",
        columns="env",
        values="covert_rate",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Covert Action Rate"},
    )

    ax.set_title("Covert Action Rate: Model × Environment")
    ax.set_xlabel("Environment")
    ax.set_ylabel("Model")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_before_after(
    store: TrajectoryStore,
    before_exp_id: str,
    after_exp_id: str,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot before/after comparison (e.g., base vs fine-tuned model)."""
    setup_style()

    before = compute_metrics(store, before_exp_id)
    after = compute_metrics(store, after_exp_id)

    metrics = ["Covert Rate", "Aligned Rate", "Overrefusal Rate"]
    before_vals = [before.covert_action_rate, before.aligned_rate, before.overrefusal_rate]
    after_vals = [after.covert_action_rate, after.aligned_rate, after.overrefusal_rate]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width / 2, before_vals, width, label=f"Before ({before_exp_id[:20]}...)",
           color="#3498db", edgecolor="black")
    ax.bar(x + width / 2, after_vals, width, label=f"After ({after_exp_id[:20]}...)",
           color="#2ecc71", edgecolor="black")

    for i, (b, a) in enumerate(zip(before_vals, after_vals)):
        change = a - b
        better = change < 0 if i != 1 else change > 0  # aligned rate: higher is better
        color = "#27ae60" if better else "#e74c3c"
        ax.annotate(
            f"{change:+.1%}",
            xy=(i + width / 2, max(b, a) + 0.05),
            ha="center",
            fontsize=10,
            color=color,
            fontweight="bold",
        )

    ax.set_ylabel("Rate")
    ax.set_title("Before vs After Training")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.15)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def generate_report(
    store: TrajectoryStore,
    experiment_ids: list[str],
    output_dir: str = "reports",
) -> str:
    """Generate a full analysis report with all visualizations."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    df = compare_experiments(store, experiment_ids)
    df.to_csv(output_path / "comparison.csv", index=False)

    for eid in experiment_ids:
        plot_behavior_distribution(
            store, eid,
            save_path=str(output_path / f"{eid}_distribution.png"),
            show=False,
        )

    plot_covert_rate_comparison(
        store, experiment_ids,
        save_path=str(output_path / "covert_comparison.png"),
        show=False,
    )

    plot_model_env_heatmap(
        store, experiment_ids,
        save_path=str(output_path / "model_env_heatmap.png"),
        show=False,
    )

    return str(output_path)
