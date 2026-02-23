from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def analyze_experiments(input_csv: str, out_dir: str):
    """
    Create summary tables and plots from the raw experiment CSV.

    Outputs:
      - summary.csv: mean/std grouped by maze_type, size, algorithm, gamma
      - plots:
          * runtime boxplots (discrete sizes)
          * nodes_expanded boxplots
          * frontier_max boxplots (memory proxy, search only)
          * efficiency ratio boxplots (path_length / nodes_expanded, search only)
          * success-rate plots (especially for MDP vs gamma)
          * MDP gamma sensitivity (runtime and iterations vs gamma)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Normalize numeric columns
    df["gamma"] = pd.to_numeric(df.get("gamma"), errors="coerce")
    df["theta"] = pd.to_numeric(df.get("theta"), errors="coerce")
    df["iterations"] = pd.to_numeric(df.get("iterations"), errors="coerce")
    df["path_length"] = pd.to_numeric(df.get("path_length"), errors="coerce")
    df["runtime_ms"] = pd.to_numeric(df.get("runtime_ms"), errors="coerce")
    df["nodes_expanded"] = pd.to_numeric(df.get("nodes_expanded"), errors="coerce")
    df["frontier_max"] = pd.to_numeric(df.get("frontier_max"), errors="coerce")

    df["size"] = df["width"].astype(int)

    # Efficiency ratio (search only; for MDP nodes_expanded is state count and frontier_max = 0)
    search_algos = {"bfs", "dfs", "astar_manhattan", "astar_euclidean"}
    df["efficiency"] = None
    mask_search = df["algorithm"].isin(search_algos) & df["path_length"].notna() & (df["nodes_expanded"] > 0)
    df.loc[mask_search, "efficiency"] = df.loc[mask_search, "path_length"] / df.loc[mask_search, "nodes_expanded"]
    df["efficiency"] = pd.to_numeric(df["efficiency"], errors="coerce")

    # Summary table
    group_cols = ["maze_type", "size", "algorithm", "gamma"]
    metrics = ["runtime_ms", "nodes_expanded", "frontier_max", "path_length", "iterations", "efficiency"]

    summary = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    summary_file = out_path / "summary.csv"
    summary.to_csv(summary_file, index=False)

    # Plot helpers
    def boxplot_metric(data: pd.DataFrame, metric: str, title: str, filename_prefix: str):
        data = data.dropna(subset=[metric])
        if data.empty:
            return

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
            fig.savefig(out_path / f"{filename_prefix}_{algo}.png", dpi=200)
            plt.close(fig)

    # Runtime/nodes/path length for all algos
    boxplot_metric(df, "runtime_ms", "Runtime distribution", "runtime_boxplot")
    boxplot_metric(df, "nodes_expanded", "Nodes expanded distribution", "expanded_boxplot")
    boxplot_metric(df, "path_length", "Path length distribution", "pathlen_boxplot")

    # Frontier max (search only)
    boxplot_metric(df[df["algorithm"].isin(search_algos)], "frontier_max", "Frontier max (memory proxy)", "frontier_boxplot")

    # Efficiency ratio (search only)
    boxplot_metric(df[df["algorithm"].isin(search_algos)], "efficiency", "Efficiency (path_length / nodes_expanded)", "efficiency_boxplot")

    # Success rate plots
    # Overall success rate by algorithm and size
    try:
        succ = df.groupby(["maze_type", "size", "algorithm"])["success"].mean().reset_index()
        for maze_type in sorted(succ["maze_type"].unique()):
            sub = succ[succ["maze_type"] == maze_type]
            for algo in sorted(sub["algorithm"].unique()):
                a = sub[sub["algorithm"] == algo]
                if a.empty:
                    continue
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(a["size"].astype(str), a["success"])
                ax.set_ylim(0, 1.05)
                ax.set_title(f"Success rate vs size | {maze_type} | {algo}")
                ax.set_xlabel("Maze size (N x N)")
                ax.set_ylabel("Success rate")
                fig.tight_layout()
                fig.savefig(out_path / f"success_rate_{maze_type}_{algo}.png", dpi=200)
                plt.close(fig)
    except Exception:
        pass

    # MDP-specific success rate vs gamma
    mdp = df[df["algorithm"].isin(["mdp_vi", "mdp_pi"])].copy()
    if not mdp.empty:
        mdp["gamma_str"] = mdp["gamma"].astype(str)

        sr = mdp.groupby(["maze_type", "size", "algorithm", "gamma"])["success"].mean().reset_index()
        for maze_type in sorted(sr["maze_type"].unique()):
            sub_type = sr[sr["maze_type"] == maze_type]
            for size in sorted(sub_type["size"].unique()):
                sub_size = sub_type[sub_type["size"] == size]

                fig, ax = plt.subplots(figsize=(7, 4))
                for algo in ["mdp_vi", "mdp_pi"]:
                    a = sub_size[sub_size["algorithm"] == algo].sort_values("gamma")
                    if a.empty:
                        continue
                    ax.plot(a["gamma"].values, a["success"].values, marker="o", label=algo)

                ax.set_ylim(0, 1.05)
                ax.set_title(f"MDP success rate vs gamma | {maze_type} | size={size}")
                ax.set_xlabel("gamma")
                ax.set_ylabel("Success rate")
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_path / f"mdp_success_vs_gamma_{maze_type}_size{size}.png", dpi=200)
                plt.close(fig)

    # MDP gamma sensitivity (runtime and iterations)
    mdp2 = df[df["algorithm"].isin(["mdp_vi", "mdp_pi"])].dropna(subset=["gamma"])
    if not mdp2.empty:
        for metric, fname in [("iterations", "mdp_iterations_vs_gamma"), ("runtime_ms", "mdp_runtime_vs_gamma")]:
            fig, ax = plt.subplots(figsize=(7, 4))
            for algo in ["mdp_vi", "mdp_pi"]:
                sub = mdp2[mdp2["algorithm"] == algo].dropna(subset=[metric])
                if sub.empty:
                    continue
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