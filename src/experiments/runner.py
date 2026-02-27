from __future__ import annotations

from pathlib import Path
from typing import List
import csv
import tracemalloc

from maze.generator import generate_maze
from maze.io import save_maze_json

from solvers.bfs import BFSSolver
from solvers.dfs import DFSSolver
from solvers.astar import AStarSolver
from solvers.mdp_value_iter import ValueIterationSolver
from solvers.mdp_policy_iter import PolicyIterationSolver


def run_experiments(
    maze_type: str,
    sizes: List[int],
    seeds: List[int],
    gammas: List[float],
    theta: float,
    output_csv: str,
    mazes_dir: str,
    wall_density: float = 0.25,
    p_intended: float = 1.0,
):
    """
    Full experiment sweep: search + MDP.

    p_intended:
      - 1.0 => deterministic MDP transitions
      - <1.0 => stochastic slip model in MDP solvers (intended prob p_intended)
    """
    Path(mazes_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "maze_id",
        "maze_file",
        "maze_type",
        "width",
        "height",
        "seed",
        "wall_density",
        "algorithm",
        "gamma",
        "theta",
        "p_intended",
        "runtime_ms",
        "nodes_expanded",
        "frontier_max",
        "path_length",
        "success",
        "iterations",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for size in sizes:
            for seed in seeds:
                maze_id = f"{maze_type}_{size}x{size}_seed{seed}"
                maze_file = f"{mazes_dir}/{maze_id}.json"

                maze = generate_maze(
                    width=size,
                    height=size,
                    maze_type=maze_type,
                    seed=seed,
                    wall_density=wall_density,
                    ensure_solvable=True,
                )
                save_maze_json(maze_file, maze)

                # Search algorithms
                search_algos = [
                    BFSSolver(),
                    DFSSolver(),
                    AStarSolver("manhattan"),
                    AStarSolver("euclidean"),
                ]
                for solver in search_algos:
                    res = solver.solve(maze)
                    writer.writerow({
                        "maze_id": maze_id,
                        "maze_file": maze_file,
                        "maze_type": maze_type,
                        "width": size,
                        "height": size,
                        "seed": seed,
                        "wall_density": wall_density if maze_type == "random" else "",
                        "algorithm": res.algorithm,
                        "gamma": "",
                        "theta": "",
                        "p_intended": "",
                        "runtime_ms": res.runtime_ms,
                        "nodes_expanded": res.nodes_expanded,
                        "frontier_max": res.frontier_max,
                        "path_length": res.path_length if res.path_length is not None else "",
                        "success": res.success,
                        "iterations": "",
                    })

                # MDP algorithms
                for gamma in gammas:
                    mdp_algos = [
                        ValueIterationSolver(gamma=gamma, theta=theta, p_intended=p_intended),
                        PolicyIterationSolver(gamma=gamma, theta=theta, p_intended=p_intended),
                    ]
                    for solver in mdp_algos:
                        res = solver.solve(maze)
                        writer.writerow({
                            "maze_id": maze_id,
                            "maze_file": maze_file,
                            "maze_type": maze_type,
                            "width": size,
                            "height": size,
                            "seed": seed,
                            "wall_density": wall_density if maze_type == "random" else "",
                            "algorithm": res.algorithm,
                            "gamma": gamma,
                            "theta": theta,
                            "p_intended": p_intended,
                            "runtime_ms": res.runtime_ms,
                            "nodes_expanded": res.nodes_expanded,
                            "frontier_max": res.frontier_max,
                            "path_length": res.path_length if res.path_length is not None else "",
                            "success": res.success,
                            "iterations": res.meta.get("iterations", "") if res.meta else "",
                        })

    print(f"Experiment results written to: {out_path}")
    print(f"Mazes saved to: {mazes_dir}")


def run_experiments_search_only(
    maze_type: str,
    sizes: List[int],
    seeds: List[int],
    output_csv: str,
    mazes_dir: str,
    wall_density: float = 0.25,
    track_peak_mem: bool = False,
):
    """
    Search-only experiment sweep (BFS/DFS/A*).
    If track_peak_mem=True, records Python allocation peak via tracemalloc as peak_mem_kb.
    """
    Path(mazes_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "maze_id",
        "maze_file",
        "maze_type",
        "width",
        "height",
        "seed",
        "wall_density",
        "algorithm",
        "gamma",
        "theta",
        "p_intended",
        "runtime_ms",
        "nodes_expanded",
        "frontier_max",
        "path_length",
        "success",
        "iterations",
        "peak_mem_kb",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for size in sizes:
            for seed in seeds:
                maze_id = f"{maze_type}_{size}x{size}_seed{seed}"
                maze_file = f"{mazes_dir}/{maze_id}.json"

                maze = generate_maze(
                    width=size,
                    height=size,
                    maze_type=maze_type,
                    seed=seed,
                    wall_density=wall_density,
                    ensure_solvable=True,
                )
                save_maze_json(maze_file, maze)

                search_algos = [
                    BFSSolver(),
                    DFSSolver(),
                    AStarSolver("manhattan"),
                    AStarSolver("euclidean"),
                ]

                for solver in search_algos:
                    peak_kb = ""

                    if track_peak_mem:
                        tracemalloc.start()
                        res = solver.solve(maze)
                        _, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        peak_kb = peak / 1024.0
                    else:
                        res = solver.solve(maze)

                    writer.writerow({
                        "maze_id": maze_id,
                        "maze_file": maze_file,
                        "maze_type": maze_type,
                        "width": size,
                        "height": size,
                        "seed": seed,
                        "wall_density": wall_density if maze_type == "random" else "",
                        "algorithm": res.algorithm,
                        "gamma": "",
                        "theta": "",
                        "p_intended": "",
                        "runtime_ms": res.runtime_ms,
                        "nodes_expanded": res.nodes_expanded,
                        "frontier_max": res.frontier_max,
                        "path_length": res.path_length if res.path_length is not None else "",
                        "success": res.success,
                        "iterations": "",
                        "peak_mem_kb": peak_kb,
                    })

    print(f"Search-only results written to: {out_path}")
    print(f"Mazes saved to: {mazes_dir}")