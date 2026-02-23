from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import matplotlib.pyplot as plt


SEARCH_ALGOS = {"bfs", "dfs", "astar_manhattan", "astar_euclidean"}
MDP_ALGOS = {"mdp_vi", "mdp_pi"}


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        df[col] = pd.NA
    return pd.to_numeric(df[col], errors="coerce")


def _available_sizes(df: pd.DataFrame) -> list[int]:
    return sorted(df["size"].dropna().astype(int).unique().tolist())


def _pick_sizes(df: pd.DataFrame, preferred: list[int]) -> list[int]:
    sizes = set(_available_sizes(df))
    return [s for s in preferred if s in sizes]


def analyze_experiments(input_csv: str, out_dir: str):
    """
    Create summary tables and plots from the raw experiment CSV.

    Produces:
      - summary.csv (mean/std grouped by maze_type, size, algorithm, gamma)
      - Distribution boxplots per algorithm (runtime, nodes expanded, path length, frontier_max, efficiency)
      - Success rate plots (overall and MDP vs gamma)
      - MDP gamma sensitivity (runtime and iterations vs gamma)
      - Scaling plots (mean vs size)
      - Additional "report-ready" bar and line charts:
          * mean runtime bar charts for key sizes
          * search scaling runtime/nodes line charts
          * MDP iterations vs size
          * MDP runtime vs gamma at key sizes
          * search vs MDP bar chart at a key size
          * search vs MDP scaling line chart
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Normalise numeric columns (safe if column absent)
    df["gamma"] = _safe_numeric(df, "gamma")
    df["theta"] = _safe_numeric(df, "theta")
    df["iterations"] = _safe_numeric(df, "iterations")
    df["path_length"] = _safe_numeric(df, "path_length")
    df["runtime_ms"] = _safe_numeric(df, "runtime_ms")
    df["nodes_expanded"] = _safe_numeric(df, "nodes_expanded")
    df["frontier_max"] = _safe_numeric(df, "frontier_max")
    if "peak_mem_kb" in df.columns:
        df["peak_mem_kb"] = _safe_numeric(df, "peak_mem_kb")

    df["size"] = df["width"].astype(int)

    # Efficiency ratio (search only)
    df["efficiency"] = pd.NA
    mask_search_eff = df["algorithm"].isin(SEARCH_ALGOS) & df["path_length"].notna() & (df["nodes_expanded"] > 0)
    df.loc[mask_search_eff, "efficiency"] = df.loc[mask_search_eff, "path_length"] / df.loc[mask_search_eff, "nodes_expanded"]
    df["efficiency"] = pd.to_numeric(df["efficiency"], errors="coerce")

    # Summary table (include peak_mem_kb if present)
    group_cols = ["maze_type", "size", "algorithm", "gamma"]
    metrics = ["runtime_ms", "nodes_expanded", "frontier_max", "path_length", "iterations", "efficiency"]
    if "peak_mem_kb" in df.columns:
        metrics.append("peak_mem_kb")

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

            ax.boxplot(series, tick_labels=[str(s) for s in sizes])
            ax.set_title(f"{title} | {algo}")
            ax.set_xlabel("Maze size (N x N)")
            ax.set_ylabel(metric)

            fig.tight_layout()
            fig.savefig(out_path / f"{filename_prefix}_{algo}.png", dpi=200)
            plt.close(fig)

    def bar_mean_by_algo_for_size(
        data: pd.DataFrame,
        size: int,
        metric: str,
        filename: str,
        title: str,
        algo_order: Optional[list[str]] = None,
        only_success: bool = True,
    ):
        sub = data[data["size"] == size].copy()
        if only_success and "success" in sub.columns:
            sub = sub[sub["success"] == True]  # noqa: E712
        sub = sub.dropna(subset=[metric])

        if sub.empty:
            return

        means = sub.groupby("algorithm")[metric].mean()

        if algo_order:
            algos = [a for a in algo_order if a in means.index]
        else:
            algos = sorted(means.index.tolist())

        vals = [means[a] for a in algos]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(algos, vals)
        ax.set_title(title)
        ax.set_xlabel("Algorithm")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=25)

        fig.tight_layout()
        fig.savefig(out_path / filename, dpi=200)
        plt.close(fig)

    def line_mean_vs_size(
        data: pd.DataFrame,
        metric: str,
        filename: str,
        title: str,
        algorithms: Optional[Iterable[str]] = None,
        only_success: bool = True,
    ):
        sub = data.copy()
        if only_success and "success" in sub.columns:
            sub = sub[sub["success"] == True]  # noqa: E712
        sub = sub.dropna(subset=[metric])

        if sub.empty:
            return

        algos = sorted(sub["algorithm"].unique()) if algorithms is None else [a for a in algorithms if a in set(sub["algorithm"].unique())]

        fig, ax = plt.subplots(figsize=(7, 4))
        for algo in algos:
            a = sub[sub["algorithm"] == algo]
            means = a.groupby("size")[metric].mean().sort_index()
            if not means.empty:
                ax.plot(means.index.values, means.values, marker="o", label=algo)

        ax.set_title(title)
        ax.set_xlabel("Maze size (N x N)")
        ax.set_ylabel(metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path / filename, dpi=200)
        plt.close(fig)

    def mdp_runtime_vs_gamma_for_size(data: pd.DataFrame, size: int, filename: str):
        mdp = data[(data["algorithm"].isin(MDP_ALGOS)) & (data["size"] == size)].copy()
        mdp = mdp.dropna(subset=["gamma", "runtime_ms"])
        if mdp.empty:
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        for algo in ["mdp_vi", "mdp_pi"]:
            a = mdp[mdp["algorithm"] == algo]
            means = a.groupby("gamma")["runtime_ms"].mean().sort_index()
            if not means.empty:
                ax.plot(means.index.values, means.values, marker="o", label=algo)

        ax.set_title(f"MDP runtime vs gamma (size={size})")
        ax.set_xlabel("gamma")
        ax.set_ylabel("runtime_ms")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path / filename, dpi=200)
        plt.close(fig)

    boxplot_metric(df, "runtime_ms", "Runtime distribution", "runtime_boxplot")
    boxplot_metric(df, "nodes_expanded", "Nodes expanded distribution", "expanded_boxplot")
    boxplot_metric(df, "path_length", "Path length distribution", "pathlen_boxplot")

    boxplot_metric(df[df["algorithm"].isin(SEARCH_ALGOS)], "frontier_max", "Frontier max (memory proxy)", "frontier_boxplot")
    boxplot_metric(df[df["algorithm"].isin(SEARCH_ALGOS)], "efficiency", "Efficiency (path_length / nodes_expanded)", "efficiency_boxplot")

    if "peak_mem_kb" in df.columns:
        boxplot_metric(df[df["algorithm"].isin(SEARCH_ALGOS)], "peak_mem_kb", "Peak Python allocation (tracemalloc) KB", "peakmem_boxplot")

    # Success rate overall vs size (per algorithm)
    if "success" in df.columns:
        succ = df.groupby(["maze_type", "size", "algorithm"])["success"].mean().reset_index()
        for maze_type in sorted(succ["maze_type"].unique()):
            sub_mt = succ[succ["maze_type"] == maze_type]
            for algo in sorted(sub_mt["algorithm"].unique()):
                a = sub_mt[sub_mt["algorithm"] == algo]
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

    # MDP success vs gamma by size (if present)
    mdp = df[df["algorithm"].isin(MDP_ALGOS)].copy()
    if not mdp.empty and "success" in mdp.columns:
        sr = mdp.dropna(subset=["gamma"]).groupby(["maze_type", "size", "algorithm", "gamma"])["success"].mean().reset_index()
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

    # MDP gamma sensitivity plots (overall)
    if not mdp.empty:
        mdp2 = mdp.dropna(subset=["gamma"])
        for metric, fname in [("iterations", "mdp_iterations_vs_gamma"), ("runtime_ms", "mdp_runtime_vs_gamma")]:
            subm = mdp2.dropna(subset=[metric])
            if subm.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            for algo in ["mdp_vi", "mdp_pi"]:
                a = subm[subm["algorithm"] == algo]
                means = a.groupby("gamma")[metric].mean().sort_index()
                if not means.empty:
                    ax.plot(means.index.values, means.values, marker="o", label=algo)
            ax.set_title(f"MDP sensitivity: {metric} vs gamma")
            ax.set_xlabel("gamma")
            ax.set_ylabel(metric)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_path / f"{fname}.png", dpi=200)
            plt.close(fig)

    # Scaling plots (mean vs size) for standard metrics
    scaling_metrics = ["runtime_ms", "nodes_expanded", "frontier_max"]
    if "peak_mem_kb" in df.columns:
        scaling_metrics.append("peak_mem_kb")

    for metric in scaling_metrics:
        line_mean_vs_size(
            df,
            metric=metric,
            filename=f"{metric}_vs_size.png",
            title=f"Mean {metric} vs maze size",
            algorithms=None,
            only_success=True,
        )

    preferred_sizes = _pick_sizes(df, [31, 51])

    # 1) Bar charts: mean runtime by algorithm at key sizes
    for s in preferred_sizes:
        bar_mean_by_algo_for_size(
            df,
            size=s,
            metric="runtime_ms",
            filename=f"bar_mean_runtime_size{s}.png",
            title=f"Mean runtime by algorithm (size={s})",
            algo_order=["dfs", "bfs", "astar_manhattan", "astar_euclidean", "mdp_vi", "mdp_pi"],
            only_success=True,
        )

    # 2) Search scaling: runtime vs size and nodes expanded vs size
    line_mean_vs_size(
        df[df["algorithm"].isin(SEARCH_ALGOS)],
        metric="runtime_ms",
        filename="search_runtime_vs_size.png",
        title="Search scaling: mean runtime vs size",
        algorithms=["dfs", "bfs", "astar_manhattan", "astar_euclidean"],
        only_success=True,
    )
    line_mean_vs_size(
        df[df["algorithm"].isin(SEARCH_ALGOS)],
        metric="nodes_expanded",
        filename="search_nodes_vs_size.png",
        title="Search scaling: mean nodes expanded vs size",
        algorithms=["dfs", "bfs", "astar_manhattan", "astar_euclidean"],
        only_success=True,
    )

    # 3) MDP iterations vs size (mean)
    line_mean_vs_size(
        df[df["algorithm"].isin(MDP_ALGOS)],
        metric="iterations",
        filename="mdp_iterations_vs_size.png",
        title="MDP scaling: mean iterations vs size",
        algorithms=["mdp_vi", "mdp_pi"],
        only_success=True,
    )

    # 4) MDP runtime vs gamma at key sizes (if gamma present)
    for s in preferred_sizes:
        mdp_runtime_vs_gamma_for_size(df, size=s, filename=f"mdp_runtime_vs_gamma_size{s}.png")

    # 5) Search vs MDP runtime bar chart at size 31 (if present)
    if 31 in set(_available_sizes(df)):
        bar_mean_by_algo_for_size(
            df,
            size=31,
            metric="runtime_ms",
            filename="bar_search_vs_mdp_runtime_size31.png",
            title="Search vs MDP: mean runtime (size=31)",
            algo_order=["dfs", "bfs", "astar_manhattan", "astar_euclidean", "mdp_vi", "mdp_pi"],
            only_success=True,
        )

    # 6) Search vs MDP scaling (use sizes 15/31/51 if present)
    compare_sizes = _pick_sizes(df, [15, 31, 51])
    if len(compare_sizes) >= 2:
        sub_cmp = df[df["size"].isin(compare_sizes)].copy()
        fig, ax = plt.subplots(figsize=(7, 4))

        # choose representative search line: A* manhattan (best informed)
        for algo in ["astar_manhattan", "mdp_vi", "mdp_pi"]:
            a = sub_cmp[sub_cmp["algorithm"] == algo].dropna(subset=["runtime_ms"])
            if "success" in a.columns:
                a = a[a["success"] == True]  # noqa: E712
            means = a.groupby("size")["runtime_ms"].mean().sort_index()
            if not means.empty:
                ax.plot(means.index.values, means.values, marker="o", label=algo)

        ax.set_title("Search vs MDP scaling: mean runtime vs size")
        ax.set_xlabel("Maze size (N x N)")
        ax.set_ylabel("runtime_ms")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path / "search_vs_mdp_runtime_vs_size.png", dpi=200)
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