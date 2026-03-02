#!/usr/bin/env python3
"""Analysis script for experiment results."""

import argparse
from data import TrajectoryStore
from analysis import (
    compute_metrics,
    compare_experiments,
    plot_behavior_distribution,
    plot_covert_rate_comparison,
    generate_report,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("experiments", nargs="+", help="Experiment IDs to analyze")
    parser.add_argument("--data-path", default="data/experiments", help="Data directory")
    parser.add_argument("--output", "-o", default="reports", help="Output directory")
    parser.add_argument("--subdir", default=None, help="Phase subdir to analyze (e.g. 'test'). Defaults to collect-phase.")
    parser.add_argument("--compare", action="store_true", help="Generate comparison plots")
    parser.add_argument("--report", action="store_true", help="Generate full report")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")

    args = parser.parse_args()

    store = TrajectoryStore(args.data_path)

    if args.report:
        output_path = generate_report(store, args.experiments, args.output)
        print(f"Report generated at: {output_path}")
        return

    # Print metrics for each experiment
    for exp_id in args.experiments:
        metrics = compute_metrics(store, exp_id, subdir=args.subdir)
        print(f"\n{'='*50}")
        print(f"Experiment: {exp_id}" + (f" [{args.subdir}]" if args.subdir else ""))
        print(f"Model: {metrics.model_name}")
        print(f"Environment: {metrics.env_name}")
        print(f"Trajectories: {metrics.total_trajectories}")
        print(f"\nBehavior Rates:")
        print(f"  Aligned:     {metrics.aligned_rate:.1%}")
        print(f"  Covert:      {metrics.covert_action_rate:.1%}")
        print(f"  Overrefusal: {metrics.overrefusal_rate:.1%}")
        print(f"  Unclear:     {metrics.unclear_rate:.1%}")
        print(f"\nEvaluation Awareness: {metrics.evaluation_awareness_rate:.1%}")

    if args.compare and len(args.experiments) > 1:
        print(f"\n{'='*50}")
        print("Comparison Table:")
        df = compare_experiments(store, args.experiments, subdir=args.subdir)
        print(df.to_string(index=False))

        plot_covert_rate_comparison(
            store, args.experiments,
            show=not args.no_show,
        )
    elif not args.no_show:
        for exp_id in args.experiments:
            plot_behavior_distribution(store, exp_id, subdir=args.subdir, show=True)


if __name__ == "__main__":
    main()
