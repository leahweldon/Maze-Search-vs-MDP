from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

from .grid import Maze

Position = Tuple[int, int]

# Styling constants
COLOR_PATH = "tab:blue"
COLOR_EXPLORED = "tab:blue"
COLOR_FRONTIER = "orange"
COLOR_CURRENT = "yellow"
COLOR_START = "green"
COLOR_GOAL = "red"

TEXT_BOX_STYLE = dict(
    facecolor="white",
    alpha=0.85,
    edgecolor="none",
    boxstyle="round,pad=0.25",
)


@dataclass(frozen=True)
class CompareRenderOptions:
    delay: float = 0.05
    max_display_points: int = 2500
    tail_only: bool = True
    final_pause_s: float = 2.0
    auto_close: bool = True

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
        ax.plot(xs, ys, linewidth=2, color=COLOR_PATH)

    ax.scatter([maze.start[1]], [maze.start[0]], marker="o", s=80, color=COLOR_START)
    ax.scatter([maze.goal[1]], [maze.goal[0]], marker="X", s=80, color=COLOR_GOAL)

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

    ax.scatter([maze.start[1]], [maze.start[0]], marker="o", s=80, color=COLOR_START)
    ax.scatter([maze.goal[1]], [maze.goal[0]], marker="X", s=80, color=COLOR_GOAL)

    if title:
        ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])

    line, = ax.plot([], [], linewidth=2, color=COLOR_PATH)

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

    ax.scatter([maze.start[1]], [maze.start[0]], marker="o", s=80, color=COLOR_START)
    ax.scatter([maze.goal[1]], [maze.goal[0]], marker="X", s=80, color=COLOR_GOAL)

    if title:
        ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])

    explored_scatter = ax.scatter([], [], s=15, color=COLOR_EXPLORED, alpha=0.5)
    frontier_scatter = ax.scatter([], [], s=25, color=COLOR_FRONTIER, alpha=0.9)
    current_scatter = ax.scatter([], [], s=55, color=COLOR_CURRENT, edgecolors="black")

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

    ax.scatter([maze.start[1]], [maze.start[0]], marker="o", s=80, color=COLOR_START)
    ax.scatter([maze.goal[1]], [maze.goal[0]], marker="X", s=80, color=COLOR_GOAL)

    if title:
        ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])

    explored_scatter = ax.scatter([], [], s=15, color=COLOR_EXPLORED, alpha=0.5)
    frontier_scatter = ax.scatter([], [], s=25, color=COLOR_FRONTIER, alpha=0.9)
    current_scatter = ax.scatter([], [], s=60, color=COLOR_CURRENT, edgecolors="black")

    for idx, frontier in zip(frame_indices, frontier_frames):
        explored = expanded_order[:idx]

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

# LIVE FRONTIER WINDOW (SIDE-BY-SIDE COMPARE)
def render_frontier_live_compare(
    maze: Maze,
    left_title: str,
    left_expanded: List[Position],
    left_frontier_frames: List[List[Position]],
    left_frame_indices: List[int],
    right_title: str,
    right_expanded: List[Position],
    right_frontier_frames: List[List[Position]],
    right_frame_indices: List[int],
    left_path: Optional[List[Position]] = None,
    right_path: Optional[List[Position]] = None,
    options: CompareRenderOptions = CompareRenderOptions(),
):
    """
    Live side-by-side comparison of two search traces on the same maze.

    The optional final paths are overlayed at the end for a clean demo finish.
    """
    if not left_frontier_frames or not right_frontier_frames:
        raise ValueError("Missing trace frames for comparison animation")

    n_frames = min(
        len(left_frontier_frames),
        len(right_frontier_frames),
        len(left_frame_indices),
        len(right_frame_indices),
    )

    plt.ion()
    grid = np.array(maze.grid, dtype=np.int8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax_l, ax_r = axes[0], axes[1]

    for ax, t in [(ax_l, left_title), (ax_r, right_title)]:
        ax.imshow(grid, cmap="gray_r", interpolation="nearest")
        ax.scatter([maze.start[1]], [maze.start[0]], marker="o", s=80, color=COLOR_START)
        ax.scatter([maze.goal[1]], [maze.goal[0]], marker="X", s=80, color=COLOR_GOAL)
        ax.set_title(t)
        ax.set_xticks([])
        ax.set_yticks([])

    # Left
    l_ex = ax_l.scatter([], [], s=15, color=COLOR_EXPLORED, alpha=0.5)
    l_fr = ax_l.scatter([], [], s=25, color=COLOR_FRONTIER, alpha=0.9)
    l_cur = ax_l.scatter([], [], s=60, color=COLOR_CURRENT, edgecolors="black")
    l_path_line, = ax_l.plot([], [], linewidth=2.5, color=COLOR_PATH, alpha=0.95)
    l_text = ax_l.text(
        0.02, 0.02, "",
        transform=ax_l.transAxes,
        fontsize=10,
        family="monospace",
        bbox=TEXT_BOX_STYLE,
    )

    # Right
    r_ex = ax_r.scatter([], [], s=15, color=COLOR_EXPLORED, alpha=0.5)
    r_fr = ax_r.scatter([], [], s=25, color=COLOR_FRONTIER, alpha=0.9)
    r_cur = ax_r.scatter([], [], s=60, color=COLOR_CURRENT, edgecolors="black")
    r_path_line, = ax_r.plot([], [], linewidth=2.5, color=COLOR_PATH, alpha=0.95)
    r_text = ax_r.text(
        0.02, 0.02, "",
        transform=ax_r.transAxes,
        fontsize=10,
        family="monospace",
        bbox=TEXT_BOX_STYLE,
    )

    def cap_points(points: List[Position]) -> np.ndarray:
        if not points:
            return np.empty((0, 2))
        if len(points) <= options.max_display_points:
            return np.array([[p[1], p[0]] for p in points], dtype=float)

        if options.tail_only:
            tail = points[-options.max_display_points:]
            return np.array([[p[1], p[0]] for p in tail], dtype=float)

        step = max(1, len(points) // options.max_display_points)
        sampled = points[::step][:options.max_display_points]
        return np.array([[p[1], p[0]] for p in sampled], dtype=float)

    for i in range(n_frames):
        fig.suptitle(f"Compare live: frame {i+1}/{n_frames}", fontsize=12)

        # Left frame
        l_idx = left_frame_indices[i]
        l_explored = left_expanded[:l_idx]
        l_frontier = left_frontier_frames[i]

        l_ex.set_offsets(cap_points(l_explored))
        l_fr.set_offsets(cap_points(l_frontier))

        l_current = None
        if l_idx > 0 and l_idx - 1 < len(left_expanded):
            l_current = left_expanded[l_idx - 1]
            l_cur.set_offsets(np.array([[l_current[1], l_current[0]]], dtype=float))

        l_text.set_text(
            "expanded: {:5d}\nfrontier: {:5d}\ncurrent:  {}".format(
                l_idx,
                len(l_frontier),
                str(l_current) if l_current else "None",
            )
        )

        # Right frame
        r_idx = right_frame_indices[i]
        r_explored = right_expanded[:r_idx]
        r_frontier = right_frontier_frames[i]

        r_ex.set_offsets(cap_points(r_explored))
        r_fr.set_offsets(cap_points(r_frontier))

        r_current = None
        if r_idx > 0 and r_idx - 1 < len(right_expanded):
            r_current = right_expanded[r_idx - 1]
            r_cur.set_offsets(np.array([[r_current[1], r_current[0]]], dtype=float))

        r_text.set_text(
            "expanded: {:5d}\nfrontier: {:5d}\ncurrent:  {}".format(
                r_idx,
                len(r_frontier),
                str(r_current) if r_current else "None",
            )
        )

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(options.delay)

    # Overlay final paths
    if left_path:
        l_path_line.set_data([p[1] for p in left_path], [p[0] for p in left_path])
    if right_path:
        r_path_line.set_data([p[1] for p in right_path], [p[0] for p in right_path])

    if left_path or right_path:
        fig.suptitle("Solved: final path overlay", fontsize=12)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(options.final_pause_s)

    plt.ioff()
    if options.auto_close:
        plt.close(fig)
    else:
        plt.show()