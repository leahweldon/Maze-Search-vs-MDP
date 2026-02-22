from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def analyze_experiments(input_csv: str, out_dir: str):
    """
    Create summary tables and plots from the raw experiment CSV.

    Output:
      - summary.csv (mean/std grouped by algorithm and size)
      - plots for runtime, nodes_expanded, path_length
      - MDP gamma sensitivity plots (runtime and iterations vs gamma)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Normalize missing values
    for col in ["gamma", "theta", "iterations", "path_length"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["size"] = df["width"].astype(int)

    # Summary table
    group_cols = ["maze_type", "size", "algorithm", "gamma"]
    metrics = ["runtime_ms", "nodes_expanded", "frontier_max", "path_length", "iterations"]

    summary = (
        df.groupby(group_cols)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_file = out_path / "summary.csv"
    summary.to_csv(summary_file, index=False)

    # Plot helpers
    def boxplot_metric(metric: str, title: str, filename: str):
        data = df.dropna(subset=[metric])

        # Only plot if we have data
        if data.empty:
            return

        # Boxplot by size for each algorithm (separate figures per algorithm keeps it readable)
        for algo in sorted(data["algorithm"].unique()):
            sub = data[data["algorithm"] == algo]
            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(7, 4))
            sizes = sorted(sub["size"].unique())
            series = [sub[sub["size"] == s][metric].values for s in sizes]

            ax.boxplot(series, labels=[str(s) for s in sizes])
            ax.set_title(f"{title} | {algo}")
            ax.set_xlabel("Maze size (N x N)")
            ax.set_ylabel(metric)

            fig.tight_layout()
            fig.savefig(out_path / f"{filename}_{algo}.png", dpi=200)
            plt.close(fig)

    boxplot_metric("runtime_ms", "Runtime distribution", "runtime_boxplot")
    boxplot_metric("nodes_expanded", "Nodes expanded distribution", "expanded_boxplot")
    boxplot_metric("path_length", "Path length distribution", "pathlen_boxplot")

    # MDP gamma sensitivity (iterations and runtime)
    mdp = df[df["algorithm"].isin(["mdp_vi", "mdp_pi"])].dropna(subset=["gamma"])
    if not mdp.empty:
        for metric, fname in [("iterations", "mdp_iterations_vs_gamma"), ("runtime_ms", "mdp_runtime_vs_gamma")]:
            fig, ax = plt.subplots(figsize=(7, 4))
            for algo in ["mdp_vi", "mdp_pi"]:
                sub = mdp[mdp["algorithm"] == algo].dropna(subset=[metric])
                if sub.empty:
                    continue
                # Plot mean by gamma
                means = sub.groupby("gamma")[metric].mean().sort_index()
                ax.plot(means.index.values, means.values, marker="o", label=algo)

            ax.set_title(f"MDP sensitivity: {metric} vs gamma")
            ax.set_xlabel("gamma")
            ax.set_ylabel(metric)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_path / f"{fname}.png", dpi=200)
            plt.close(fig)

    print(f"Summary written to: {summary_file}")
    print(f"Plots written to: {out_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input_csv", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    analyze_experiments(args.input_csv, args.out_dir)