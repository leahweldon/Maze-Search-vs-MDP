from __future__ import annotations

import csv
from pathlib import Path
from typing import List

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
    mazes_dir: str = "results/mazes",
):
    """
    Execute a full experimental sweep and write results to a single CSV.

    Reproducibility:
      - Each generated maze is saved to disk (JSON) using a deterministic name.
      - The CSV records maze_id + maze_file so experiments can be reproduced exactly.

    Algorithms:
      - Search: BFS, DFS, A* (Manhattan), A* (Euclidean)
      - MDP: Value Iteration, Policy Iteration (sweeping gamma, fixed theta)

    Metrics recorded:
      - runtime_ms, nodes_expanded, frontier_max, path_length, success
      - MDP: iterations (from result.meta)
    """
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mazes_path = Path(mazes_dir)
    mazes_path.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "maze_id",
        "maze_file",
        "maze_type",
        "width",
        "height",
        "seed",
        "algorithm",
        "gamma",
        "theta",
        "runtime_ms",
        "nodes_expanded",
        "frontier_max",
        "path_length",
        "success",
        "iterations",
    ]

    with open(out_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for size in sizes:
            for seed in seeds:
                maze_id = f"{maze_type}_{size}x{size}_seed{seed}"
                maze_file = mazes_path / f"{maze_id}.json"

                maze = generate_maze(
                    width=size,
                    height=size,
                    maze_type=maze_type,
                    seed=seed,
                    ensure_solvable=True,
                )
                save_maze_json(str(maze_file), maze)

                # --- SEARCH ALGORITHMS ---
                search_solvers = [
                    BFSSolver(),
                    DFSSolver(),
                    AStarSolver("manhattan"),
                    AStarSolver("euclidean"),
                ]

                for solver in search_solvers:
                    result = solver.solve(maze)

                    writer.writerow(
                        {
                            "maze_id": maze_id,
                            "maze_file": str(maze_file),
                            "maze_type": maze_type,
                            "width": size,
                            "height": size,
                            "seed": seed,
                            "algorithm": result.algorithm,
                            "gamma": "",
                            "theta": "",
                            "runtime_ms": result.runtime_ms,
                            "nodes_expanded": result.nodes_expanded,
                            "frontier_max": result.frontier_max,
                            "path_length": result.path_length if result.path_length is not None else "",
                            "success": result.success,
                            "iterations": "",
                        }
                    )

                # --- MDP ALGORITHMS ---
                for gamma in gammas:
                    mdp_solvers = [
                        ValueIterationSolver(gamma=gamma, theta=theta),
                        PolicyIterationSolver(gamma=gamma, theta=theta),
                    ]

                    for solver in mdp_solvers:
                        result = solver.solve(maze)

                        writer.writerow(
                            {
                                "maze_id": maze_id,
                                "maze_file": str(maze_file),
                                "maze_type": maze_type,
                                "width": size,
                                "height": size,
                                "seed": seed,
                                "algorithm": result.algorithm,
                                "gamma": gamma,
                                "theta": theta,
                                "runtime_ms": result.runtime_ms,
                                "nodes_expanded": result.nodes_expanded,
                                "frontier_max": result.frontier_max,
                                "path_length": result.path_length if result.path_length is not None else "",
                                "success": result.success,
                                "iterations": result.meta.get("iterations", ""),
                            }
                        )

    print(f"Experiment results written to: {out_path}")
    print(f"Mazes saved to: {mazes_path}")