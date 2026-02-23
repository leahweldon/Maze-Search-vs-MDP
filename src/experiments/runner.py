from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import csv

from maze.generator import generate_maze
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
):
    """
    Execute full experiment sweep.

    Adds:
      - wall_density column for random mazes
    """

    Path(mazes_dir).mkdir(parents=True, exist_ok=True)

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
        "runtime_ms",
        "nodes_expanded",
        "frontier_max",
        "path_length",
        "success",
        "iterations",
    ]

    with open(output_csv, "w", newline="") as f:
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
                    wall_density=wall_density if maze_type == "random" else None,
                    ensure_solvable=True,
                )

                # SEARCH ALGORITHMS
                search_algos = [
                    BFSSolver(),
                    DFSSolver(),
                    AStarSolver("manhattan"),
                    AStarSolver("euclidean"),
                ]

                for solver in search_algos:
                    result = solver.solve(maze)
                    writer.writerow({
                        "maze_id": maze_id,
                        "maze_file": maze_file,
                        "maze_type": maze_type,
                        "width": size,
                        "height": size,
                        "seed": seed,
                        "wall_density": wall_density if maze_type == "random" else "",
                        "algorithm": result.algorithm,
                        "gamma": "",
                        "theta": "",
                        "runtime_ms": result.runtime_ms,
                        "nodes_expanded": result.nodes_expanded,
                        "frontier_max": result.frontier_max,
                        "path_length": result.path_length,
                        "success": result.success,
                        "iterations": "",
                    })

                # MDP ALGORITHMS
                for gamma in gammas:
                    mdp_algos = [
                        ValueIterationSolver(gamma, theta),
                        PolicyIterationSolver(gamma, theta),
                    ]

                    for solver in mdp_algos:
                        result = solver.solve(maze)
                        writer.writerow({
                            "maze_id": maze_id,
                            "maze_file": maze_file,
                            "maze_type": maze_type,
                            "width": size,
                            "height": size,
                            "seed": seed,
                            "wall_density": wall_density if maze_type == "random" else "",
                            "algorithm": result.algorithm,
                            "gamma": gamma,
                            "theta": theta,
                            "runtime_ms": result.runtime_ms,
                            "nodes_expanded": result.nodes_expanded,
                            "frontier_max": result.frontier_max,
                            "path_length": result.path_length,
                            "success": result.success,
                            "iterations": result.meta.get("iterations") if result.meta else "",
                        })

    print(f"Experiment results written to: {output_csv}")
    print(f"Mazes saved to: {mazes_dir}")