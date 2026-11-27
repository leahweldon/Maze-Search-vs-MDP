"""Draw the animated figures used in the README.

Both animations are generated from this project's own solvers:

  * search-race.gif     DFS, BFS and both A* variants exploring the same random
                        maze, one expansion at a time, using each solver's trace.
  * value-iteration.gif Value Iteration on a perfect maze, sweep by sweep,
                        using the solver's own Bellman backup.

Run from the repository root:

    PYTHONPATH=src:. python scripts/readme_animations.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from maze.generator import generate_maze
from solvers.astar import AStarSolver
from solvers.bfs import BFSSolver
from solvers.dfs import DFSSolver
from solvers.mdp_value_iter import ValueIterationSolver

OUT = Path("docs/assets")

PAPER = (244, 240, 230)
FREE = (251, 248, 241)
WALL = (28, 27, 25)
MUTED = (125, 117, 102)
INK = (28, 27, 25)
SLATE = (63, 94, 115)
EXPANDED = (178, 196, 207)
RUST = (165, 69, 47)
GOLD = (201, 162, 39)
KICKER = (122, 74, 31)


def font(size: int, mono: bool = False, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSansMono" if mono else "DejaVuSerif"
    if bold:
        name += "-Bold"
    for folder in ("/usr/share/fonts/truetype/dejavu", "/Library/Fonts", "C:/Windows/Fonts"):
        path = Path(folder) / f"{name}.ttf"
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_cells(draw, maze, x0, y0, cell, colour_of):
    for r in range(maze.height):
        for c in range(maze.width):
            draw.rectangle(
                [x0 + c * cell, y0 + r * cell, x0 + (c + 1) * cell - 1, y0 + (r + 1) * cell - 1],
                fill=colour_of((r, c)),
            )


def draw_path(draw, path, x0, y0, cell, colour, width):
    centres = [(x0 + c * cell + cell / 2, y0 + r * cell + cell / 2) for r, c in path]
    draw.line(centres, fill=colour, width=width, joint="curve")


def draw_endpoints(draw, maze, x0, y0, cell):
    for (r, c), colour in ((maze.start, GOLD), (maze.goal, RUST)):
        pad = max(1, cell // 6)
        draw.ellipse(
            [x0 + c * cell + pad, y0 + r * cell + pad, x0 + (c + 1) * cell - pad, y0 + (r + 1) * cell - pad],
            fill=colour,
        )


def save_gif(frames, durations, path):
    frames = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


def search_race(seed: int = 7, size: int = 41, cell: int = 8) -> None:
    maze = generate_maze(size, size, "random", seed=seed, wall_density=0.25, ensure_solvable=True)
    solvers = [
        ("DFS", DFSSolver(trace=True)),
        ("BFS", BFSSolver(trace=True)),
        ("A* Euclidean", AStarSolver("euclidean", trace=True)),
        ("A* Manhattan", AStarSolver("manhattan", trace=True)),
    ]
    runs = [(name, solver.solve(maze)) for name, solver in solvers]
    shortest = min(res.path_length for _, res in runs if res.path_length is not None)

    board = size * cell
    gap, margin, header = 28, 40, 120
    width = margin * 2 + 4 * board + 3 * gap
    height = header + board + 74
    longest = max(len(res.meta["expanded_order"]) for _, res in runs)
    steps = 90
    per_frame = max(1, -(-longest // steps))

    frames, durations = [], []
    for step in range(steps + 1):
        shown = step * per_frame
        image = Image.new("RGB", (width, height), PAPER)
        draw = ImageDraw.Draw(image)
        draw.text((margin, 26), "SEARCH, ONE EXPANSION AT A TIME", font=font(15, mono=True, bold=True), fill=KICKER)
        draw.text(
            (margin, 50),
            f"Four solvers on the same {size} \u00d7 {size} random maze (wall density 0.25, seed {seed})",
            font=font(22),
            fill=INK,
        )
        for i, (name, res) in enumerate(runs):
            x0 = margin + i * (board + gap)
            y0 = header
            order = res.meta["expanded_order"]
            visible = set(order[: min(shown, len(order))])
            done = shown >= len(order)
            draw_cells(
                draw,
                maze,
                x0,
                y0,
                cell,
                lambda p: WALL if maze.grid[p[0]][p[1]] else (EXPANDED if p in visible else FREE),
            )
            if done and res.path:
                draw_path(draw, res.path, x0, y0, cell, RUST, max(2, cell // 3))
            draw_endpoints(draw, maze, x0, y0, cell)
            draw.text((x0, y0 - 30), name, font=font(19, bold=True), fill=INK)
            count = f"{min(shown, len(order)):,} expanded"
            draw.text((x0, y0 + board + 12), count, font=font(15, mono=True), fill=SLATE)
            if done and res.path_length is not None:
                tag = "shortest" if res.path_length == shortest else "longer"
                draw.text(
                    (x0, y0 + board + 36),
                    f"path {res.path_length} ({tag})",
                    font=font(15, mono=True, bold=True),
                    fill=RUST if tag == "shortest" else MUTED,
                )
        frames.append(image)
        durations.append(2600 if step == steps else 70)
    save_gif(frames, durations, OUT / "search-race.gif")
    print("search-race.gif:", {name: len(res.meta["expanded_order"]) for name, res in runs})


def value_iteration(seed: int = 3, size: int = 31, cell: int = 14, gamma: float = 0.95) -> None:
    maze = generate_maze(size, size, "perfect", seed=seed)
    solver = ValueIterationSolver(gamma=gamma)
    final = solver.solve(maze)

    states = [(r, c) for r in range(maze.height) for c in range(maze.width) if maze.grid[r][c] == 0]
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    V = {s: 0.0 for s in states}
    snapshots = []
    while True:
        delta = 0.0
        for s in states:
            if s == maze.goal:
                continue
            old = V[s]
            V[s] = max(solver._q_value(maze, V, s, a) for a in actions)
            delta = max(delta, abs(old - V[s]))
        snapshots.append(dict(V))
        if delta < solver.theta:
            break

    converged_values = snapshots[-1]
    # Colour by rank of the final value, so the gradient runs evenly from the
    # goal to the furthest cell even though values flatten out far away.
    ranked = sorted(states, key=lambda s: -converged_values[s])
    position = {s: i / max(1, len(ranked) - 1) for i, s in enumerate(ranked)}
    ramp = np.array([(246, 227, 161), GOLD, RUST, SLATE, (143, 168, 184)], dtype=float)

    def colour_for(state):
        t = position[state] * (len(ramp) - 1)
        i = min(int(t), len(ramp) - 2)
        mix = ramp[i] + (ramp[i + 1] - ramp[i]) * (t - i)
        return tuple(int(v) for v in mix)

    board = maze.width * cell
    margin, header = 40, 120
    width = margin * 2 + board + 300
    height = header + board + 60

    picks = sorted(set(list(range(0, len(snapshots), max(1, len(snapshots) // 70))) + [len(snapshots) - 1]))
    frames, durations = [], []
    for k, index in enumerate(picks):
        snap = snapshots[index]
        settled = {s for s in states if abs(snap[s] - converged_values[s]) < 1e-3}
        image = Image.new("RGB", (width, height), PAPER)
        draw = ImageDraw.Draw(image)
        draw.text((margin, 26), "VALUE ITERATION, SWEEP BY SWEEP", font=font(15, mono=True, bold=True), fill=KICKER)
        draw.text((margin, 50), f"{maze.width} \u00d7 {maze.height} perfect maze, \u03b3 = {gamma}", font=font(22), fill=INK)
        draw_cells(
            draw,
            maze,
            margin,
            header,
            cell,
            lambda p: WALL if maze.grid[p[0]][p[1]] else (colour_for(p) if p in settled else FREE),
        )
        last = index == len(snapshots) - 1
        if last and final.path:
            draw_path(draw, final.path, margin, header, cell, (255, 255, 255), max(2, cell // 5))
        draw_endpoints(draw, maze, margin, header, cell)

        tx = margin + board + 34
        draw.text((tx, header), "SWEEP", font=font(13, mono=True, bold=True), fill=MUTED)
        draw.text((tx, header + 20), f"{index + 1}", font=font(44, bold=True), fill=INK)
        draw.text((tx, header + 90), "CELLS SETTLED", font=font(13, mono=True, bold=True), fill=MUTED)
        draw.text((tx, header + 110), f"{len(settled) / len(states):.0%}", font=font(44, bold=True), fill=SLATE)
        for j, line in enumerate(
            [
                "Each sweep applies the",
                "Bellman update to every",
                "open cell. Value spreads",
                "out from the goal until",
                "every cell knows its",
                "best move.",
            ]
        ):
            draw.text((tx, header + 190 + j * 24), line, font=font(16), fill=INK)
        if last:
            draw.text((tx, header + 350), "Greedy policy path", font=font(16, bold=True), fill=INK)
            draw.text((tx, header + 374), f"{final.path_length} steps", font=font(16, mono=True), fill=INK)
        frames.append(image)
        durations.append(3000 if last else (900 if k == 0 else 80))
    save_gif(frames, durations, OUT / "value-iteration.gif")
    print("value-iteration.gif:", len(snapshots), "sweeps; solver reported", final.meta["iterations"])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    search_race()
    value_iteration()
