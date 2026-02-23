from __future__ import annotations

from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

from .grid import Maze

Position = Tuple[int, int]

# ASCII RENDERING
def render_ascii(maze: Maze, path: List[Position] | None = None) -> str:
    """
    Render maze as ASCII text.
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

# STATIC PNG RENDER
def render_png(maze: Maze, path: List[Position] | None, out_path: str, title: Optional[str] = None):
    grid = np.array(maze.grid, dtype=np.int8)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="gray_r", interpolation="nearest")

    if path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, linewidth=2, color="tab:blue")

    ax.scatter([maze.start[1]], [maze.start[0]], marker="o", s=80, color="green")
    ax.scatter([maze.goal[1]], [maze.goal[0]], marker="X", s=80, color="red")

    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# FINAL PATH ANIMATION
def render_path_animation(
    maze: Maze,
    path: List[Position],
    out_path: str,
    title: Optional[str] = None,
    fps: int = 10,
):
    """
    Animate final solution path.
    """
    import matplotlib.animation as animation

    if not path:
        raise ValueError("Cannot animate empty path")

    grid = np.array(maze.grid, dtype=np.int8)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="gray_r", interpolation="nearest")

    ax.scatter([maze.start[1]], [maze.start[0]], marker="o", s=80, color="green")
    ax.scatter([maze.goal[1]], [maze.goal[0]], marker="X", s=80, color="red")

    if title:
        ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])

    line, = ax.plot([], [], linewidth=2, color="tab:blue")

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

    if out_path.lower().endswith(".gif"):
        writer = animation.PillowWriter(fps=fps)
        ani.save(out_path, writer=writer)
    else:
        ani.save(out_path, fps=fps)

    plt.close(fig)

# FRONTIER ANIMATION (GIF/MP4)
def render_frontier_animation(
    maze: Maze,
    expanded_order: List[Position],
    frontier_frames: List[List[Position]],
    frame_indices: List[int],
    out_path: str,
    title: Optional[str] = None,
    fps: int = 12,
):
    import matplotlib.animation as animation

    grid = np.array(maze.grid, dtype=np.int8)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="gray_r", interpolation="nearest")

    ax.scatter([maze.start[1]], [maze.start[0]], marker="o", s=80, color="green")
    ax.scatter([maze.goal[1]], [maze.goal[0]], marker="X", s=80, color="red")

    if title:
        ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])

    explored_scatter = ax.scatter([], [], s=15, color="tab:blue", alpha=0.5)
    frontier_scatter = ax.scatter([], [], s=25, color="orange", alpha=0.9)
    current_scatter = ax.scatter([], [], s=55, color="yellow", edgecolors="black")

    def init():
        explored_scatter.set_offsets(np.empty((0, 2)))
        frontier_scatter.set_offsets(np.empty((0, 2)))
        current_scatter.set_offsets(np.empty((0, 2)))
        return explored_scatter, frontier_scatter, current_scatter

    def update(i: int):
        idx = frame_indices[i]
        explored = expanded_order[:idx]
        frontier = frontier_frames[i]

        ex = np.array([[p[1], p[0]] for p in explored], dtype=float) if explored else np.empty((0, 2))
        fr = np.array([[p[1], p[0]] for p in frontier], dtype=float) if frontier else np.empty((0, 2))

        explored_scatter.set_offsets(ex)
        frontier_scatter.set_offsets(fr)

        if idx > 0:
            current_node = expanded_order[idx - 1]
            current_scatter.set_offsets(np.array([[current_node[1], current_node[0]]], dtype=float))

        return explored_scatter, frontier_scatter, current_scatter

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frontier_frames),
        init_func=init,
        interval=int(1000 / fps),
        blit=True,
        repeat=False,
    )

    if out_path.lower().endswith(".gif"):
        writer = animation.PillowWriter(fps=fps)
        ani.save(out_path, writer=writer)
    else:
        ani.save(out_path, fps=fps)

    plt.close(fig)

# LIVE FRONTIER WINDOW
def render_frontier_live(
    maze: Maze,
    expanded_order: List[Position],
    frontier_frames: List[List[Position]],
    frame_indices: List[int],
    title: Optional[str] = None,
    delay: float = 0.05,
):
    plt.ion()

    grid = np.array(maze.grid, dtype=np.int8)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="gray_r", interpolation="nearest")

    ax.scatter([maze.start[1]], [maze.start[0]], marker="o", s=80, color="green")
    ax.scatter([maze.goal[1]], [maze.goal[0]], marker="X", s=80, color="red")

    if title:
        ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])

    explored_scatter = ax.scatter([], [], s=15, color="tab:blue", alpha=0.5)
    frontier_scatter = ax.scatter([], [], s=25, color="orange", alpha=0.9)
    current_scatter = ax.scatter([], [], s=60, color="yellow", edgecolors="black")

    for i, idx in enumerate(frame_indices):
        explored = expanded_order[:idx]
        frontier = frontier_frames[i]

        ex = np.array([[p[1], p[0]] for p in explored], dtype=float) if explored else np.empty((0, 2))
        fr = np.array([[p[1], p[0]] for p in frontier], dtype=float) if frontier else np.empty((0, 2))

        explored_scatter.set_offsets(ex)
        frontier_scatter.set_offsets(fr)

        if idx > 0:
            current_node = expanded_order[idx - 1]
            current_scatter.set_offsets(np.array([[current_node[1], current_node[0]]], dtype=float))

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(delay)

    plt.ioff()
    plt.show()