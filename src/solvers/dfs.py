from __future__ import annotations

from typing import Dict, Optional, List

from maze.grid import Maze, Position
from metrics import SearchStats, timer_ms
from solvers.base import SolveResult


class DFSSolver:
    """
    Depth-First Search solver for maze navigation (iterative stack).

    Optional tracing (for frontier animations):
      - expanded_order: nodes expanded in order
      - frontier_frames: stack snapshots at sampled steps
      - frame_indices: expansion count at each snapshot

    Tracing is disabled by default to keep experiments fast.
    """

    name = "dfs"

    def __init__(self, trace: bool = False, trace_stride: int = 10, trace_max_frames: int = 400):
        self.trace = trace
        self.trace_stride = max(1, trace_stride)
        self.trace_max_frames = max(1, trace_max_frames)

    def solve(self, maze: Maze) -> SolveResult:
        stats = SearchStats()
        stack: List[Position] = [maze.start]
        parent: Dict[Position, Optional[Position]] = {maze.start: None}

        expanded_order: List[Position] = []
        frontier_frames: List[List[Position]] = []
        frame_indices: List[int] = []
        frames_recorded = 0

        stats.frontier_max = max(stats.frontier_max, len(stack))

        with timer_ms() as elapsed:
            found = False

            while stack:
                current = stack.pop()
                stats.nodes_expanded += 1

                if self.trace:
                    expanded_order.append(current)

                if current == maze.goal:
                    found = True
                    break

                for nxt in maze.neighbors_4(current):
                    if nxt not in parent:
                        parent[nxt] = current
                        stack.append(nxt)

                stats.frontier_max = max(stats.frontier_max, len(stack))

                if (
                    self.trace
                    and (stats.nodes_expanded % self.trace_stride == 0)
                    and frames_recorded < self.trace_max_frames
                ):
                    frontier_frames.append(list(stack))
                    frame_indices.append(len(expanded_order))
                    frames_recorded += 1

        if not found:
            return SolveResult(
                algorithm=self.name,
                success=False,
                path=None,
                path_length=None,
                runtime_ms=elapsed(),
                nodes_expanded=stats.nodes_expanded,
                frontier_max=stats.frontier_max,
                meta={
                    "expanded_order": expanded_order if self.trace else None,
                    "frontier_frames": frontier_frames if self.trace else None,
                    "frame_indices": frame_indices if self.trace else None,
                    "trace_stride": self.trace_stride if self.trace else None,
                },
            )

        path = _reconstruct_path(parent, maze.goal)
        return SolveResult(
            algorithm=self.name,
            success=True,
            path=path,
            path_length=len(path) - 1,
            runtime_ms=elapsed(),
            nodes_expanded=stats.nodes_expanded,
            frontier_max=stats.frontier_max,
            meta={
                "expanded_order": expanded_order if self.trace else None,
                "frontier_frames": frontier_frames if self.trace else None,
                "frame_indices": frame_indices if self.trace else None,
                "trace_stride": self.trace_stride if self.trace else None,
            },
        )


def _reconstruct_path(parent: Dict[Position, Optional[Position]], goal: Position) -> List[Position]:
    path: List[Position] = []
    cur: Optional[Position] = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path