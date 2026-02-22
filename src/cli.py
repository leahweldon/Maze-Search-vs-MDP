from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

from maze.generator import generate_maze
from maze.io import load_maze_json, save_maze_json
from maze.render import render_ascii, render_png
from solvers.bfs import BFSSolver


Position = Tuple[int, int]


def _parse_pos(value: str) -> Position:
    """
    Parse a position from 'row,col' (e.g., '0,0' or '10,15').
    """
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Position must be in the form 'row,col'")
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError("Position must contain two integers") from e


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


def cmd_generate(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    maze = generate_maze(
        width=args.width,
        height=args.height,
        maze_type=args.type,
        seed=args.seed,
        wall_density=args.density,
        ensure_solvable=args.ensure_solvable,
        max_tries=args.max_tries,
        start=args.start,
        goal=args.goal,
    )

    save_maze_json(str(out_path), maze)
    print(f"Saved maze: {out_path}")
    print(f"Type: {args.type} | size: {maze.width}x{maze.height} | seed: {args.seed}")

    if args.ascii:
        print("\n=== Maze (ASCII) ===")
        print(render_ascii(maze, path=None))

    if args.png:
        png_path = Path(args.png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        title = f"MAZE | {args.type} | {maze.width}x{maze.height} | seed={args.seed}"
        render_png(maze, path=None, out_path=str(png_path), title=title)
        print(f"Saved PNG: {png_path}")

    return 0


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

    gen_p = sub.add_parser("generate", help="Generate a maze and save it to JSON")
    gen_p.add_argument("--type", required=True, choices=["perfect", "random"], help="Maze generator type")
    gen_p.add_argument("--width", type=int, required=True, help="Maze width (columns)")
    gen_p.add_argument("--height", type=int, required=True, help="Maze height (rows)")
    gen_p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    gen_p.add_argument(
        "--density",
        type=float,
        default=0.25,
        help="Wall density for random mazes (ignored for perfect mazes)",
    )
    gen_p.add_argument(
        "--ensure-solvable",
        action="store_true",
        help="Regenerate until start and goal are connected (up to max_tries)",
    )
    gen_p.add_argument(
        "--max-tries",
        type=int,
        default=50,
        help="Maximum regeneration attempts when ensure-solvable is enabled",
    )
    gen_p.add_argument(
        "--start",
        type=_parse_pos,
        default=None,
        help="Optional start position as 'row,col'. Defaults depend on maze type.",
    )
    gen_p.add_argument(
        "--goal",
        type=_parse_pos,
        default=None,
        help="Optional goal position as 'row,col'. Defaults depend on maze type.",
    )
    gen_p.add_argument("--out", required=True, help="Output JSON file path")
    gen_p.add_argument("--ascii", action="store_true", help="Print ASCII view of generated maze")
    gen_p.add_argument("--png", default=None, help="Optional output path to save a PNG visualization")
    gen_p.set_defaults(func=cmd_generate)

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