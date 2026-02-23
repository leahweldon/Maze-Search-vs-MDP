from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List

from experiments.runner import run_experiments, run_experiments_search_only
from experiments.yaml_presets import load_presets
from maze.generator import generate_maze
from maze.io import load_maze_json, save_maze_json
from maze.render import (
    render_ascii,
    render_png,
    render_path_animation,
    render_frontier_animation,
    render_frontier_live,
    render_frontier_live_compare,
    CompareRenderOptions,
)

from solvers.bfs import BFSSolver
from solvers.dfs import DFSSolver
from solvers.astar import AStarSolver
from solvers.mdp_value_iter import ValueIterationSolver
from solvers.mdp_policy_iter import PolicyIterationSolver

Position = Tuple[int, int]


def _parse_pos(value: str) -> Position:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Position must be in form 'row,col'")
    return int(parts[0]), int(parts[1])


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


def build_tracing_solver(algo: str, stride: int, max_frames: int):
    algo = algo.lower().strip()

    if algo == "bfs":
        return BFSSolver(trace=True, trace_stride=stride, trace_max_frames=max_frames)
    if algo == "dfs":
        return DFSSolver(trace=True, trace_stride=stride, trace_max_frames=max_frames)
    if algo == "astar_manhattan":
        return AStarSolver("manhattan", trace=True, trace_stride=stride, trace_max_frames=max_frames)
    if algo == "astar_euclidean":
        return AStarSolver("euclidean", trace=True, trace_stride=stride, trace_max_frames=max_frames)

    raise ValueError("Tracing only supported for bfs, dfs, astar_manhattan, astar_euclidean.")

# COMMAND HANDLERS
def cmd_generate(args):
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
        render_png(maze, None, str(png_path))
        print(f"Saved PNG: {png_path}")

    return 0


def cmd_solve(args):
    maze = load_maze_json(args.maze)
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
            if v is not None:
                print(f"{k}: {v}")

    if args.ascii:
        print(render_ascii(maze, result.path))

    if args.png:
        out_path = Path(args.png)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        render_png(maze, result.path, str(out_path))

    return 0


def cmd_animate(args):
    maze = load_maze_json(args.maze)
    solver = build_solver(args.algo, gamma=args.gamma, theta=args.theta)
    result = solver.solve(maze)

    if not result.success or not result.path:
        print("No solution to animate.")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    render_path_animation(maze, result.path, str(out_path), fps=args.fps)
    print(f"Saved animation: {out_path}")
    return 0


def cmd_animate_frontier(args):
    maze = load_maze_json(args.maze)
    solver = build_tracing_solver(args.algo, args.stride, args.max_frames)
    result = solver.solve(maze)

    expanded = result.meta.get("expanded_order") if result.meta else None
    frontier = result.meta.get("frontier_frames") if result.meta else None
    indices = result.meta.get("frame_indices") if result.meta else None

    if not expanded or not frontier or not indices:
        print("No trace data available.")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    render_frontier_animation(maze, expanded, frontier, indices, str(out_path), fps=args.fps)
    print(f"Saved frontier animation: {out_path}")
    return 0


def cmd_animate_live(args):
    maze = load_maze_json(args.maze)
    solver = build_tracing_solver(args.algo, args.stride, args.max_frames)
    result = solver.solve(maze)

    expanded = result.meta.get("expanded_order") if result.meta else None
    frontier = result.meta.get("frontier_frames") if result.meta else None
    indices = result.meta.get("frame_indices") if result.meta else None

    if not expanded or not frontier or not indices:
        print("No trace data available.")
        return 1

    render_frontier_live(maze, expanded, frontier, indices, delay=args.delay)
    return 0


def cmd_animate_live_compare(args):
    maze = load_maze_json(args.maze)

    solver1 = build_tracing_solver(args.algo1, args.stride, args.max_frames)
    solver2 = build_tracing_solver(args.algo2, args.stride, args.max_frames)

    res1 = solver1.solve(maze)
    res2 = solver2.solve(maze)

    e1 = res1.meta.get("expanded_order") if res1.meta else None
    f1 = res1.meta.get("frontier_frames") if res1.meta else None
    i1 = res1.meta.get("frame_indices") if res1.meta else None

    e2 = res2.meta.get("expanded_order") if res2.meta else None
    f2 = res2.meta.get("frontier_frames") if res2.meta else None
    i2 = res2.meta.get("frame_indices") if res2.meta else None

    if not e1 or not f1 or not i1 or not e2 or not f2 or not i2:
        print("Trace data missing for comparison.")
        return 1

    opts = CompareRenderOptions(
        delay=args.delay,
        final_pause_s=args.final_pause,
        auto_close=not args.keep_open,
    )

    render_frontier_live_compare(
        maze=maze,
        left_title=args.algo1,
        left_expanded=e1,
        left_frontier_frames=f1,
        left_frame_indices=i1,
        right_title=args.algo2,
        right_expanded=e2,
        right_frontier_frames=f2,
        right_frame_indices=i2,
        left_path=res1.path,
        right_path=res2.path,
        options=opts,
    )
    return 0


def cmd_animate_compare_gif(args):
    """
    Export two separate GIFs (frontier + optional final path) for side-by-side use in the report.
    This is simpler and more reliable than generating one combined GIF.
    """
    maze = load_maze_json(args.maze)

    solver1 = build_tracing_solver(args.algo1, args.stride, args.max_frames)
    solver2 = build_tracing_solver(args.algo2, args.stride, args.max_frames)

    res1 = solver1.solve(maze)
    res2 = solver2.solve(maze)

    e1 = res1.meta.get("expanded_order") if res1.meta else None
    f1 = res1.meta.get("frontier_frames") if res1.meta else None
    i1 = res1.meta.get("frame_indices") if res1.meta else None

    e2 = res2.meta.get("expanded_order") if res2.meta else None
    f2 = res2.meta.get("frontier_frames") if res2.meta else None
    i2 = res2.meta.get("frame_indices") if res2.meta else None

    if not e1 or not f1 or not i1 or not e2 or not f2 or not i2:
        print("Trace data missing; cannot export GIFs.")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gif1 = out_dir / f"{args.algo1}_frontier.gif"
    gif2 = out_dir / f"{args.algo2}_frontier.gif"

    render_frontier_animation(maze, e1, f1, i1, str(gif1), fps=args.fps)
    render_frontier_animation(maze, e2, f2, i2, str(gif2), fps=args.fps)

    print(f"Saved frontier GIFs:\n  {gif1}\n  {gif2}")

    if args.export_paths:
        if res1.success and res1.path:
            p1 = out_dir / f"{args.algo1}_path.gif"
            render_path_animation(maze, res1.path, str(p1), fps=args.fps)
            print(f"Saved path GIF: {p1}")

        if res2.success and res2.path:
            p2 = out_dir / f"{args.algo2}_path.gif"
            render_path_animation(maze, res2.path, str(p2), fps=args.fps)
            print(f"Saved path GIF: {p2}")

    return 0


def cmd_export_solutions(args):
    """
    Export solution PNGs (and optionally GIFs) for all algorithms on a single maze.
    This is ideal for report appendices and quick demo assets.
    """
    maze = load_maze_json(args.maze)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gamma = args.gamma
    theta = args.theta

    algo_list = [
        ("bfs", BFSSolver()),
        ("dfs", DFSSolver()),
        ("astar_manhattan", AStarSolver("manhattan")),
        ("astar_euclidean", AStarSolver("euclidean")),
        ("mdp_vi", ValueIterationSolver(gamma=gamma, theta=theta)),
        ("mdp_pi", PolicyIterationSolver(gamma=gamma, theta=theta)),
    ]

    for name, solver in algo_list:
        res = solver.solve(maze)

        png_path = out_dir / f"{name}.png"
        render_png(maze, res.path if res.success else None, str(png_path), title=f"{name} | success={res.success}")
        print(f"Saved PNG: {png_path}")

        if args.export_path_gifs and res.success and res.path:
            gif_path = out_dir / f"{name}_path.gif"
            render_path_animation(maze, res.path, str(gif_path), fps=args.fps)
            print(f"Saved path GIF: {gif_path}")

        if args.export_frontier_gifs and name in {"bfs", "dfs", "astar_manhattan", "astar_euclidean"}:
            tracer = build_tracing_solver(name, stride=args.stride, max_frames=args.max_frames)
            t_res = tracer.solve(maze)

            e = t_res.meta.get("expanded_order") if t_res.meta else None
            f = t_res.meta.get("frontier_frames") if t_res.meta else None
            i = t_res.meta.get("frame_indices") if t_res.meta else None

            if e and f and i:
                frontier_gif = out_dir / f"{name}_frontier.gif"
                render_frontier_animation(maze, e, f, i, str(frontier_gif), fps=args.fps)
                print(f"Saved frontier GIF: {frontier_gif}")

    return 0


def cmd_experiment(args):
    run_experiments(
        maze_type=args.type,
        sizes=args.sizes,
        seeds=args.seeds,
        gammas=args.gammas,
        theta=args.theta,
        output_csv=args.out,
        mazes_dir=args.mazes_dir,
        wall_density=args.density,
    )
    return 0

def cmd_experiment_search(args):
    run_experiments_search_only(
        maze_type=args.type,
        sizes=args.sizes,
        seeds=args.seeds,
        output_csv=args.out,
        mazes_dir=args.mazes_dir,
        wall_density=args.density,
        track_peak_mem=args.track_peak_mem,
    )
    return 0

def cmd_experiment_from_yaml(args):
    presets = load_presets(args.config)
    if args.preset not in presets:
        print(f"Preset not found: {args.preset}")
        print("Available presets:", ", ".join(sorted(presets.keys())))
        return 2

    p = presets[args.preset]

    if p.mode == "full":
        if not p.gammas:
            raise ValueError("full mode preset must include gammas")
        run_experiments(
            maze_type=p.maze_type,
            sizes=p.sizes,
            seeds=p.seeds,
            gammas=p.gammas,
            theta=p.theta,
            output_csv=p.out_csv,
            mazes_dir=p.mazes_dir,
            wall_density=p.density,
        )
        return 0

    if p.mode == "search_only":
        run_experiments_search_only(
            maze_type=p.maze_type,
            sizes=p.sizes,
            seeds=p.seeds,
            output_csv=p.out_csv,
            mazes_dir=p.mazes_dir,
            wall_density=p.density,
            track_peak_mem=p.track_peak_mem,
        )
        return 0

    raise ValueError(f"Unknown preset mode: {p.mode}")

# PARSER
def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--type", required=True, choices=["perfect", "random"])
    g.add_argument("--width", type=int, required=True)
    g.add_argument("--height", type=int, required=True)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--density", type=float, default=0.25)
    g.add_argument("--ensure-solvable", action="store_true")
    g.add_argument("--max-tries", type=int, default=50)
    g.add_argument("--start", type=_parse_pos, default=None)
    g.add_argument("--goal", type=_parse_pos, default=None)
    g.add_argument("--out", required=True)
    g.add_argument("--ascii", action="store_true")
    g.add_argument("--png", default=None)
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("solve")
    s.add_argument("--maze", required=True)
    s.add_argument("--algo", required=True)
    s.add_argument("--gamma", type=float, default=0.95)
    s.add_argument("--theta", type=float, default=1e-6)
    s.add_argument("--ascii", action="store_true")
    s.add_argument("--png", default=None)
    s.set_defaults(func=cmd_solve)

    a = sub.add_parser("animate")
    a.add_argument("--maze", required=True)
    a.add_argument("--algo", required=True)
    a.add_argument("--gamma", type=float, default=0.95)
    a.add_argument("--theta", type=float, default=1e-6)
    a.add_argument("--fps", type=int, default=10)
    a.add_argument("--out", required=True)
    a.set_defaults(func=cmd_animate)

    af = sub.add_parser("animate_frontier")
    af.add_argument("--maze", required=True)
    af.add_argument("--algo", required=True)
    af.add_argument("--stride", type=int, default=10)
    af.add_argument("--max-frames", type=int, default=400)
    af.add_argument("--fps", type=int, default=12)
    af.add_argument("--out", required=True)
    af.set_defaults(func=cmd_animate_frontier)

    al = sub.add_parser("animate_live")
    al.add_argument("--maze", required=True)
    al.add_argument("--algo", required=True)
    al.add_argument("--stride", type=int, default=10)
    al.add_argument("--max-frames", type=int, default=400)
    al.add_argument("--delay", type=float, default=0.05)
    al.set_defaults(func=cmd_animate_live)

    alc = sub.add_parser("animate_live_compare")
    alc.add_argument("--maze", required=True)
    alc.add_argument("--algo1", required=True)
    alc.add_argument("--algo2", required=True)
    alc.add_argument("--stride", type=int, default=10)
    alc.add_argument("--max-frames", type=int, default=400)
    alc.add_argument("--delay", type=float, default=0.02)
    alc.add_argument("--final-pause", type=float, default=2.0)
    alc.add_argument("--keep-open", action="store_true")
    alc.set_defaults(func=cmd_animate_live_compare)

    cg = sub.add_parser("animate_compare_gif", help="Export two GIFs for side-by-side comparison")
    cg.add_argument("--maze", required=True)
    cg.add_argument("--algo1", required=True)
    cg.add_argument("--algo2", required=True)
    cg.add_argument("--stride", type=int, default=10)
    cg.add_argument("--max-frames", type=int, default=400)
    cg.add_argument("--fps", type=int, default=12)
    cg.add_argument("--out-dir", required=True)
    cg.add_argument("--export-paths", action="store_true")
    cg.set_defaults(func=cmd_animate_compare_gif)

    es = sub.add_parser("export_solutions", help="Export solution PNGs (and optional GIFs) for all algorithms")
    es.add_argument("--maze", required=True)
    es.add_argument("--out-dir", required=True)
    es.add_argument("--gamma", type=float, default=0.95)
    es.add_argument("--theta", type=float, default=1e-6)
    es.add_argument("--fps", type=int, default=12)
    es.add_argument("--export-path-gifs", action="store_true")
    es.add_argument("--export-frontier-gifs", action="store_true")
    es.add_argument("--stride", type=int, default=10)
    es.add_argument("--max-frames", type=int, default=400)
    es.set_defaults(func=cmd_export_solutions)

    esrch = sub.add_parser("experiment_search", help="Search-only experiment sweep (useful for 100x100)")
    esrch.add_argument("--type", required=True, choices=["perfect", "random"])
    esrch.add_argument("--sizes", type=_parse_int_list, required=True)
    esrch.add_argument("--seeds", type=_parse_int_list, required=True)
    esrch.add_argument("--mazes-dir", default="results/mazes")
    esrch.add_argument("--density", type=float, default=0.25)
    esrch.add_argument("--out", required=True)
    esrch.set_defaults(func=cmd_experiment_search)
    esrch.add_argument("--track-peak-mem", action="store_true", help="Record peak tracemalloc memory in KB (search-only)")

    e = sub.add_parser("experiment")
    e.add_argument("--type", required=True, choices=["perfect", "random"])
    e.add_argument("--sizes", type=_parse_int_list, required=True)
    e.add_argument("--seeds", type=_parse_int_list, required=True)
    e.add_argument("--gammas", type=_parse_float_list, required=True)
    e.add_argument("--theta", type=float, default=1e-6)
    e.add_argument("--mazes-dir", default="results/mazes")
    e.add_argument("--density", type=float, default=0.25)
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_experiment)

    xy = sub.add_parser("experiment_from_yaml", help="Run an experiment preset from a YAML config file")
    xy.add_argument("--config", default="config/experiments.yaml")
    xy.add_argument("--preset", required=True, help="Preset name under 'presets:' in YAML")
    xy.set_defaults(func=cmd_experiment_from_yaml)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())