from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .grid import Maze

Position = Tuple[int, int]


def render_ascii(maze: Maze, path: List[Position] | None = None) -> str:
    """
    Render a maze as ASCII characters.

    Legend:
      # = wall
      . = free space
      S = start
      G = goal
      * = path
    """
    path_set = set(path) if path else set()
    lines: List[str] = []

    for r in range(maze.height):
        row_chars: List[str] = []
        for c in range(maze.width):
            pos = (r, c)

            if pos == maze.start:
                row_chars.append("S")
            elif pos == maze.goal:
                row_chars.append("G")
            elif pos in path_set:
                row_chars.append("*")
            elif maze.grid[r][c] == 1:
                row_chars.append("#")
            else:
                row_chars.append(".")

        lines.append("".join(row_chars))

    return "\n".join(lines)


def render_png(maze: Maze, path: List[Position] | None, out_path: str, title: str | None = None):
    """
    Save a maze (and optional solution path) as a PNG image.

    This is intended for:
      - report figures
      - demo evidence (showing the solver runs)
      - quick visual inspection when debugging
    """
    grid = np.array(maze.grid, dtype=np.int8)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, interpolation="nearest")

    if path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, linewidth=2)

    ax.scatter([maze.start[1]], [maze.start[0]], marker="o", s=80)
    ax.scatter([maze.goal[1]], [maze.goal[0]], marker="X", s=80)

    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)