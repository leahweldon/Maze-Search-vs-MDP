from __future__ import annotations

import heapq
from typing import Dict, Optional, List, Callable, Tuple

from maze.grid import Maze, Position
from metrics import SearchStats, timer_ms
from solvers.base import SolveResult
from heuristics import manhattan, euclidean


class AStarSolver:
    """
    A* search implementation with optional tracing for frontier animation.

    f(n) = g(n) + h(n)

    - Goal test is applied when a node is removed from the priority queue.
    - Includes stale-entry protection to avoid expanding outdated heap entries.

    Optional tracing (disabled by default):
      - expanded_order: nodes expanded in order
      - frontier_frames: snapshots of the best-known open set at sampled steps
      - frame_indices: expansion index corresponding to each snapshot
    """

    def __init__(
        self,
        heuristic_name: str,
        trace: bool = False,
        trace_stride: int = 5,
        trace_max_frames: int = 500,
    ):
        self.heuristic_name = heuristic_name.lower().strip()

        if self.heuristic_name == "manhattan":
            self.heuristic: Callable[[Position, Position], float] = manhattan
        elif self.heuristic_name == "euclidean":
            self.heuristic = euclidean
        else:
            raise ValueError("Unsupported heuristic. Use 'manhattan' or 'euclidean'.")

        self.name = f"astar_{self.heuristic_name}"

        self.trace = trace
        self.trace_stride = max(1, trace_stride)
        self.trace_max_frames = max(1, trace_max_frames)

    def solve(self, maze: Maze) -> SolveResult:
        stats = SearchStats()

        # Heap entries: (f_score, tie_break, position)
        open_heap: List[Tuple[float, int, Position]] = []
        counter = 0  # deterministic tie-breaker

        g_score: Dict[Position, float] = {maze.start: 0.0}
        parent: Dict[Position, Optional[Position]] = {maze.start: None}

        # Track best-known frontier membership
        open_best: Dict[Position, float] = {}

        start_f = self.heuristic(maze.start, maze.goal)
        heapq.heappush(open_heap, (start_f, counter, maze.start))
        open_best[maze.start] = start_f

        expanded_order: List[Position] = []
        frontier_frames: List[List[Position]] = []
        frame_indices: List[int] = []
        frames_recorded = 0

        stats.frontier_max = 1

        with timer_ms() as elapsed:
            found = False

            while open_heap:
                f_current, _, current = heapq.heappop(open_heap)

                # Stale entry protection skip if this is not the best-known f for this node
                expected_f = g_score[current] + self.heuristic(current, maze.goal)
                if f_current > expected_f:
                    continue

                # Remove from best-known open set when it is the currently-best queued value
                if open_best.get(current) == f_current:
                    open_best.pop(current, None)

                stats.nodes_expanded += 1

                if self.trace:
                    expanded_order.append(current)

                if current == maze.goal:
                    found = True
                    break

                for neighbor in maze.neighbors_4(current):
                    tentative_g = g_score[current] + 1  # unit step cost

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        parent[neighbor] = current

                        counter += 1
                        f_score = tentative_g + self.heuristic(neighbor, maze.goal)
                        heapq.heappush(open_heap, (f_score, counter, neighbor))

                        prev = open_best.get(neighbor)
                        if prev is None or f_score < prev:
                            open_best[neighbor] = f_score

                stats.frontier_max = max(stats.frontier_max, len(open_best))

                if (
                    self.trace
                    and (stats.nodes_expanded % self.trace_stride == 0)
                    and frames_recorded < self.trace_max_frames
                ):
                    frontier_frames.append(list(open_best.keys()))
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
                    "heuristic": self.heuristic_name,
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
                "heuristic": self.heuristic_name,
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