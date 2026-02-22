from __future__ import annotations

import heapq
from typing import Dict, Optional, List, Callable

from maze.grid import Maze, Position
from metrics import SearchStats, timer_ms
from solvers.base import SolveResult
from heuristics import manhattan, euclidean


class AStarSolver:
    """
    A* search implementation.

    Uses f(n) = g(n) + h(n)
    where:
      g(n) is cost from start to n
      h(n) is heuristic estimate from n to goal

    Goal test is applied when a node is removed from the priority queue,
    which guarantees optimality when heuristic is admissible.
    """

    def __init__(self, heuristic_name: str):
        self.heuristic_name = heuristic_name.lower()

        if self.heuristic_name == "manhattan":
            self.heuristic: Callable[[Position, Position], float] = manhattan
        elif self.heuristic_name == "euclidean":
            self.heuristic = euclidean
        else:
            raise ValueError("Unsupported heuristic. Use 'manhattan' or 'euclidean'.")

        self.name = f"astar_{self.heuristic_name}"

    def solve(self, maze: Maze) -> SolveResult:
        stats = SearchStats()

        open_heap: List[tuple[float, Position]] = []
        heapq.heappush(open_heap, (0.0, maze.start))

        parent: Dict[Position, Optional[Position]] = {maze.start: None}
        g_score: Dict[Position, float] = {maze.start: 0.0}

        stats.frontier_max = 1

        with timer_ms() as elapsed:
            found = False

            while open_heap:
                _, current = heapq.heappop(open_heap)
                stats.nodes_expanded += 1

                if current == maze.goal:
                    found = True
                    break

                for neighbor in maze.neighbors_4(current):
                    tentative_g = g_score[current] + 1  # uniform cost

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + self.heuristic(neighbor, maze.goal)
                        parent[neighbor] = current
                        heapq.heappush(open_heap, (f_score, neighbor))

                stats.frontier_max = max(stats.frontier_max, len(open_heap))

        if not found:
            return SolveResult(
                algorithm=self.name,
                success=False,
                path=None,
                path_length=None,
                runtime_ms=elapsed(),
                nodes_expanded=stats.nodes_expanded,
                frontier_max=stats.frontier_max,
                meta={"heuristic": self.heuristic_name},
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
            meta={"heuristic": self.heuristic_name},
        )


def _reconstruct_path(parent: Dict[Position, Optional[Position]], goal: Position):
    path = []
    cur: Optional[Position] = goal

    while cur is not None:
        path.append(cur)
        cur = parent[cur]

    path.reverse()
    return path