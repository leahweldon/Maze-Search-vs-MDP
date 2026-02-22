from __future__ import annotations

from typing import Dict, Optional, List

from maze.grid import Maze, Position
from metrics import SearchStats, timer_ms
from solvers.base import SolveResult


class DFSSolver:
    """
    Depth-First Search solver for maze navigation.

    This implementation is iterative (stack-based) rather than recursive
    to avoid recursion depth limits on larger mazes.

    DFS does not guarantee optimality but is useful for comparison
    of exploration behaviour and memory characteristics.
    """

    name = "dfs"

    def solve(self, maze: Maze) -> SolveResult:
        stats = SearchStats()
        stack: List[Position] = [maze.start]
        parent: Dict[Position, Optional[Position]] = {maze.start: None}

        stats.frontier_max = max(stats.frontier_max, len(stack))

        with timer_ms() as elapsed:
            found = False

            while stack:
                current = stack.pop()
                stats.nodes_expanded += 1

                if current == maze.goal:
                    found = True
                    break

                # DFS explores deeply: push neighbours onto stack
                for nxt in maze.neighbors_4(current):
                    if nxt not in parent:
                        parent[nxt] = current
                        stack.append(nxt)

                stats.frontier_max = max(stats.frontier_max, len(stack))

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
            path_length=len(path) - 1,
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