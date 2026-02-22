from __future__ import annotations

from typing import List, Tuple, Optional

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


def render_path_animation(
    maze: Maze,
    path: List[Position],
    out_path: str,
    title: Optional[str] = None,
    fps: int = 10,
):
    """
    Render an animation of a solution path being traversed.

    Output format is inferred from out_path:
      - .gif uses PillowWriter (no external dependencies)
      - .mp4 typically requires ffmpeg installed on your system

    This is designed for demo/report evidence and is intentionally lightweight:
      it animates the final path, not the internal search frontier.
    """
    import matplotlib.animation as animation

    if not path:
        raise ValueError("Cannot animate an empty path")

    grid = np.array(maze.grid, dtype=np.int8)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, interpolation="nearest")

    ax.scatter([maze.start[1]], [maze.start[0]], marker="o", s=80)
    ax.scatter([maze.goal[1]], [maze.goal[0]], marker="X", s=80)

    if title:
        ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])

    # line that grows as the animation proceeds
    line, = ax.plot([], [], linewidth=2)

    xs_all = [p[1] for p in path]
    ys_all = [p[0] for p in path]

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame: int):
        line.set_data(xs_all[: frame + 1], ys_all[: frame + 1])
        return (line,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(path),
        init_func=init,
        interval=int(1000 / fps),
        blit=True,
        repeat=False,
    )

    out_path_lower = out_path.lower()
    if out_path_lower.endswith(".gif"):
        writer = animation.PillowWriter(fps=fps)
        ani.save(out_path, writer=writer)
    elif out_path_lower.endswith(".mp4"):
        ani.save(out_path, fps=fps)
    else:
        raise ValueError("Animation output must end with .gif or .mp4")

    plt.close(fig)