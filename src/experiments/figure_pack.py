from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class PackItem:
    src_rel: str
    dst_name: str
    required: bool = True


def _copy_if_exists(src: Path, dst: Path, required: bool):
    if not src.exists():
        if required:
            raise FileNotFoundError(f"Required figure not found: {src}")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def pack_figures(
    perfect_dir: str,
    random_dir: str,
    out_dir: str = "report/figures",
):
    """
    Copy a curated set of plots into a single folder with clean names.

    - perfect_dir/random_dir should point to analysis output folders produced by analyze.py
    - out_dir will be created if it doesn't exist
    """
    perfect = Path(perfect_dir)
    random = Path(random_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    items: list[tuple[Path, PackItem]] = [
        # Perfect mazes: scaling behaviour
        (perfect, PackItem("runtime_ms_vs_size.png", "perfect_runtime_vs_size.png", required=True)),
        (perfect, PackItem("nodes_expanded_vs_size.png", "perfect_nodes_vs_size.png", required=True)),
        (perfect, PackItem("frontier_max_vs_size.png", "perfect_frontier_vs_size.png", required=True)),
        (perfect, PackItem("efficiency_boxplot_astar_manhattan.png", "perfect_efficiency_astar_manhattan.png", required=False)),
        (perfect, PackItem("efficiency_boxplot_bfs.png", "perfect_efficiency_bfs.png", required=False)),

        # Random mazes: scaling behaviour
        (random, PackItem("runtime_ms_vs_size.png", "random_runtime_vs_size.png", required=True)),
        (random, PackItem("nodes_expanded_vs_size.png", "random_nodes_vs_size.png", required=True)),
        (random, PackItem("frontier_max_vs_size.png", "random_frontier_vs_size.png", required=True)),

        # MDP sensitivity plots (from either folder; they’ll look similar)
        (perfect, PackItem("mdp_runtime_vs_gamma.png", "mdp_runtime_vs_gamma.png", required=True)),
        (perfect, PackItem("mdp_iterations_vs_gamma.png", "mdp_iterations_vs_gamma.png", required=True)),
    ]

    copied = 0
    for base, item in items:
        src = base / item.src_rel
        dst = out / item.dst_name
        if _copy_if_exists(src, dst, required=item.required):
            copied += 1

    # Also copy any success-rate plots for MDP if present
    for p in perfect.glob("mdp_success_vs_gamma_*.png"):
        shutil.copy2(p, out / f"perfect_{p.name}")
        copied += 1

    for p in random.glob("mdp_success_vs_gamma_*.png"):
        shutil.copy2(p, out / f"random_{p.name}")
        copied += 1

    print(f"Packed {copied} figure(s) into: {out.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pack selected analysis plots into report/figures.")
    parser.add_argument("--perfect-dir", required=True, help="Analysis folder for perfect mazes (output of analyze.py)")
    parser.add_argument("--random-dir", required=True, help="Analysis folder for random mazes (output of analyze.py)")
    parser.add_argument("--out-dir", default="report/figures", help="Output folder for packed figures")
    args = parser.parse_args()

    pack_figures(args.perfect_dir, args.random_dir, args.out_dir)