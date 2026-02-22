from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

from maze.generator import generate_maze
from maze.io import load_maze_json, save_maze_json
from maze.render import render_ascii, render_png
from solvers.bfs import BFSSolver
from solvers.dfs import DFSSolver


Position = Tuple[int, int]


def _parse_pos(value: str) -> Position:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Position must be in the form 'row,col'")
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError("Position must contain two integers") from e


def build_solver(algo: str):
    algo = algo.lower().strip()

    if algo == "bfs":
        return BFSSolver()
    if algo == "dfs":
        return DFSSolver()

    raise ValueError(
        f"Unknown algorithm '{algo}'. Currently implemented: bfs, dfs."
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
    gen_p.add_argument("--type", required=True, choices=["perfect", "random"])
    gen_p.add_argument("--width", type=int, required=True)
    gen_p.add_argument("--height", type=int, required=True)
    gen_p.add_argument("--seed", type=int, default=None)
    gen_p.add_argument("--density", type=float, default=0.25)
    gen_p.add_argument("--ensure-solvable", action="store_true")
    gen_p.add_argument("--max-tries", type=int, default=50)
    gen_p.add_argument("--start", type=_parse_pos, default=None)
    gen_p.add_argument("--goal", type=_parse_pos, default=None)
    gen_p.add_argument("--out", required=True)
    gen_p.add_argument("--ascii", action="store_true")
    gen_p.add_argument("--png", default=None)
    gen_p.set_defaults(func=cmd_generate)

    solve_p = sub.add_parser("solve", help="Solve a maze")
    solve_p.add_argument("--maze", required=True)
    solve_p.add_argument("--algo", required=True)
    solve_p.add_argument("--ascii", action="store_true")
    solve_p.add_argument("--png", default=None)
    solve_p.set_defaults(func=cmd_solve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())