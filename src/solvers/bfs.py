from __future__ import annotations

from collections import deque
from typing import Dict, Optional

from maze.grid import Maze, Position
from metrics import SearchStats, timer_ms
from solvers.base import SolveResult


class BFSSolver:
    name = "bfs"

    def solve(self, maze: Maze) -> SolveResult:
        stats = SearchStats()
        parent: Dict[Position, Optional[Position]] = {maze.start: None}

        q = deque([maze.start])
        stats.frontier_max = max(stats.frontier_max, len(q))

        with timer_ms() as elapsed:
            found = False

            while q:
                current = q.popleft()
                stats.nodes_expanded += 1

                if current == maze.goal:
                    found = True
                    break

                for nxt in maze.neighbors_4(current):
                    if nxt not in parent:
                        parent[nxt] = current
                        q.append(nxt)

                stats.frontier_max = max(stats.frontier_max, len(q))

        if not found:
            return SolveResult(
                algorithm=self.name,
                success=False,
                path=None,
                path_length=None,
                runtime_ms=elapsed(),
                nodes_expanded=stats.nodes_expanded,
                frontier_max=stats.frontier_max,
                meta={},
            )

        path = _reconstruct_path(parent, maze.goal)
        return SolveResult(
            algorithm=self.name,
            success=True,
            path=path,
            path_length=len(path) - 1,  # moves count
            runtime_ms=elapsed(),
            nodes_expanded=stats.nodes_expanded,
            frontier_max=stats.frontier_max,
            meta={},
        )


def _reconstruct_path(parent: Dict[Position, Optional[Position]], goal: Position):
    path = []
    cur: Optional[Position] = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path