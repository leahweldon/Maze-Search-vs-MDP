from __future__ import annotations

import argparse
import sys
from pathlib import Path

from maze.io import load_maze_json
from maze.render import render_ascii, render_png
from solvers.bfs import BFSSolver


def build_solver(algo: str):
    """
    Create a solver instance based on the algorithm name.
    Will extend mapping as DFS, A*, value iteration, and policy iteration are implemented.
    """
    algo = algo.lower().strip()

    if algo == "bfs":
        return BFSSolver()

    raise ValueError(
        f"Unknown algorithm '{algo}'. Currently implemented: bfs. "
        "Will add more algorithms as they are implemented."
    )


def cmd_solve(args: argparse.Namespace) -> int:
    maze_path = Path(args.maze)
    if not maze_path.exists():
        print(f"Error: maze file not found: {maze_path}", file=sys.stderr)
        return 2

    maze = load_maze_json(str(maze_path))
    solver = build_solver(args.algo)

    result = solver.solve(maze)

    # Console summary
    print(f"Algorithm: {result.algorithm}")
    print(f"Success:   {result.success}")
    print(f"Runtime:   {result.runtime_ms:.3f} ms")
    print(f"Expanded:  {result.nodes_expanded}")
    print(f"Frontier:  {result.frontier_max}")
    print(f"Path len:  {result.path_length}")

    if args.ascii:
        print("\n=== Maze (ASCII) ===")
        print(render_ascii(maze, result.path))

    if args.png:
        out_path = Path(args.png)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        title = f"{result.algorithm.upper()} | success={result.success} | len={result.path_length}"
        render_png(maze, result.path, str(out_path), title=title)
        print(f"\nSaved PNG: {out_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cs7is2-assignment1",
        description="CS7IS2 Assignment 1: Maze Search vs MDP Solvers",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    solve_p = sub.add_parser("solve", help="Solve a maze using a selected algorithm")
    solve_p.add_argument("--maze", required=True, help="Path to maze JSON file")
    solve_p.add_argument(
        "--algo",
        required=True,
        help="Algorithm name (e.g., bfs). More will be added as implemented.",
    )
    solve_p.add_argument(
        "--ascii",
        action="store_true",
        help="Print an ASCII rendering of the maze and solution path",
    )
    solve_p.add_argument(
        "--png",
        default=None,
        help="Optional output path to save a PNG visualization",
    )
    solve_p.set_defaults(func=cmd_solve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())