from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, List

from experiments.runner import run_experiments
from maze.generator import generate_maze
from maze.io import load_maze_json, save_maze_json
from maze.render import render_ascii, render_png, render_path_animation

from solvers.bfs import BFSSolver
from solvers.dfs import DFSSolver
from solvers.astar import AStarSolver
from solvers.mdp_value_iter import ValueIterationSolver
from solvers.mdp_policy_iter import PolicyIterationSolver


Position = Tuple[int, int]


def _parse_pos(value: str) -> Position:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Position must be in the form 'row,col'")
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError("Position must contain two integers") from e


def _parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def build_solver(algo: str, gamma: float, theta: float):
    algo = algo.lower().strip()

    if algo == "bfs":
        return BFSSolver()
    if algo == "dfs":
        return DFSSolver()
    if algo == "astar_manhattan":
        return AStarSolver("manhattan")
    if algo == "astar_euclidean":
        return AStarSolver("euclidean")
    if algo == "mdp_vi":
        return ValueIterationSolver(gamma=gamma, theta=theta)
    if algo == "mdp_pi":
        return PolicyIterationSolver(gamma=gamma, theta=theta)

    raise ValueError(
        "Unknown algorithm. Supported: "
        "bfs, dfs, astar_manhattan, astar_euclidean, mdp_vi, mdp_pi"
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
        render_png(maze, None, str(png_path), title=title)
        print(f"Saved PNG: {png_path}")

    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    maze_path = Path(args.maze)
    if not maze_path.exists():
        print(f"Error: maze file not found: {maze_path}", file=sys.stderr)
        return 2

    maze = load_maze_json(str(maze_path))
    solver = build_solver(args.algo, gamma=args.gamma, theta=args.theta)

    result = solver.solve(maze)

    print(f"Algorithm: {result.algorithm}")
    print(f"Success:   {result.success}")
    print(f"Runtime:   {result.runtime_ms:.3f} ms")
    print(f"Expanded:  {result.nodes_expanded}")
    print(f"Frontier:  {result.frontier_max}")
    print(f"Path len:  {result.path_length}")

    if result.meta:
        for k, v in result.meta.items():
            print(f"{k}: {v}")

    if args.ascii:
        print("\n=== Maze (ASCII) ===")
        print(render_ascii(maze, result.path))

    if args.png:
        out_path = Path(args.png)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        title = f"{result.algorithm} | success={result.success} | len={result.path_length}"
        render_png(maze, result.path, str(out_path), title=title)
        print(f"Saved PNG: {out_path}")

    return 0


def cmd_animate(args: argparse.Namespace) -> int:
    maze_path = Path(args.maze)
    if not maze_path.exists():
        print(f"Error: maze file not found: {maze_path}", file=sys.stderr)
        return 2

    maze = load_maze_json(str(maze_path))
    solver = build_solver(args.algo, gamma=args.gamma, theta=args.theta)
    result = solver.solve(maze)

    if not result.success or not result.path:
        print("No solution found; cannot animate.", file=sys.stderr)
        return 3

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    title = f"{result.algorithm} | len={result.path_length}"
    render_path_animation(maze, result.path, str(out_path), title=title, fps=args.fps)
    print(f"Saved animation: {out_path}")
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    run_experiments(
        maze_type=args.type,
        sizes=args.sizes,
        seeds=args.seeds,
        gammas=args.gammas,
        theta=args.theta,
        output_csv=args.out,
        mazes_dir=args.mazes_dir,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cs7is2-assignment1",
        description="CS7IS2 Assignment 1: Maze Search vs MDP Solvers",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # generate
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

    # solve
    solve_p = sub.add_parser("solve", help="Solve a maze and print metrics")
    solve_p.add_argument("--maze", required=True)
    solve_p.add_argument("--algo", required=True)
    solve_p.add_argument("--gamma", type=float, default=0.95)
    solve_p.add_argument("--theta", type=float, default=1e-6)
    solve_p.add_argument("--ascii", action="store_true")
    solve_p.add_argument("--png", default=None)
    solve_p.set_defaults(func=cmd_solve)

    # animate
    anim_p = sub.add_parser("animate", help="Animate a solution path and save as GIF/MP4")
    anim_p.add_argument("--maze", required=True)
    anim_p.add_argument("--algo", required=True)
    anim_p.add_argument("--gamma", type=float, default=0.95)
    anim_p.add_argument("--theta", type=float, default=1e-6)
    anim_p.add_argument("--fps", type=int, default=10)
    anim_p.add_argument("--out", required=True, help="Output .gif or .mp4 file")
    anim_p.set_defaults(func=cmd_animate)

    # experiment
    exp_p = sub.add_parser("experiment", help="Run an experiment sweep and write CSV")
    exp_p.add_argument("--type", required=True, choices=["perfect", "random"])
    exp_p.add_argument("--sizes", type=_parse_int_list, required=True)
    exp_p.add_argument("--seeds", type=_parse_int_list, required=True)
    exp_p.add_argument("--gammas", type=_parse_float_list, required=True)
    exp_p.add_argument("--theta", type=float, default=1e-6)
    exp_p.add_argument("--mazes-dir", default="results/mazes")
    exp_p.add_argument("--out", required=True)
    exp_p.set_defaults(func=cmd_experiment)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())