from __future__ import annotations

from collections import deque
from typing import Dict, Tuple, List, Optional

from maze.grid import Maze, Position
from solvers.base import SolveResult
from metrics import timer_ms
from heuristics import manhattan

Action = Tuple[int, int]


class ValueIterationSolver:
    name = "mdp_vi"

    def __init__(self, gamma: float = 0.95, theta: float = 1e-6):
        self.gamma = gamma
        self.theta = theta

    def solve(self, maze: Maze) -> SolveResult:
        states = [
            (r, c)
            for r in range(maze.height)
            for c in range(maze.width)
            if maze.grid[r][c] == 0
        ]
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        V: Dict[Position, float] = {s: 0.0 for s in states}

        with timer_ms() as elapsed:
            iterations = 0
            while True:
                delta = 0.0
                iterations += 1

                for s in states:
                    if s == maze.goal:
                        continue

                    best = float("-inf")
                    for a in actions:
                        nxt = self._transition(maze, s, a)
                        r = self._reward(maze, nxt)
                        q = r + self.gamma * V[nxt]
                        if q > best:
                            best = q

                    delta = max(delta, abs(best - V[s]))
                    V[s] = best

                if delta < self.theta:
                    break

        path = self._extract_path_via_policy_graph(maze, V)

        return SolveResult(
            algorithm=self.name,
            success=path is not None,
            path=path,
            path_length=len(path) - 1 if path else None,
            runtime_ms=elapsed(),
            nodes_expanded=len(states),
            frontier_max=0,
            meta={"gamma": self.gamma, "theta": self.theta, "iterations": iterations},
        )

    def _transition(self, maze: Maze, state: Position, action: Action) -> Position:
        r, c = state
        dr, dc = action
        nxt = (r + dr, c + dc)
        if maze.in_bounds(nxt) and maze.is_free(nxt):
            return nxt
        return state

    def _reward(self, maze: Maze, state: Position) -> float:
        return 0.0 if state == maze.goal else -1.0

    def _best_successors(self, maze: Maze, V: Dict[Position, float], s: Position) -> List[Position]:
        """
        Return the set of best successor states under the greedy (argmax Q) policy.
        Excludes no-op moves if any real move exists.
        """
        if s == maze.goal:
            return []

        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        candidates: List[Tuple[float, float, Position]] = []
        real_move_exists = False

        for a in actions:
            nxt = self._transition(maze, s, a)
            if nxt != s:
                real_move_exists = True
            r = self._reward(maze, nxt)
            q = r + self.gamma * V[nxt]
            # tie-break toward goal - bigger is better -> use -manhattan
            candidates.append((q, -manhattan(nxt, maze.goal), nxt))

        # Filter no-op if there are real moves
        if real_move_exists:
            candidates = [c for c in candidates if c[2] != s]

        if not candidates:
            return []

        candidates.sort(reverse=True)
        best_q = candidates[0][0]

        # Collect all successors within a tiny tolerance of best_q (ties)
        eps = 1e-12
        best = [nxt for (q, _, nxt) in candidates if abs(q - best_q) <= eps]

        # Remove duplicates while preserving order
        seen = set()
        out = []
        for nxt in best:
            if nxt not in seen:
                out.append(nxt)
                seen.add(nxt)
        return out

    def _extract_path_via_policy_graph(self, maze: Maze, V: Dict[Position, float]) -> Optional[List[Position]]:
        """
        Robust extraction:
        Build a directed graph of greedy-optimal policy edges (ties allowed), then run BFS to goal.
        """
        start = maze.start
        goal = maze.goal

        # BFS over states using the induced policy edges
        q = deque([start])
        parent: Dict[Position, Optional[Position]] = {start: None}

        while q:
            cur = q.popleft()
            if cur == goal:
                break

            for nxt in self._best_successors(maze, V, cur):
                if nxt not in parent:
                    parent[nxt] = cur
                    q.append(nxt)

        if goal not in parent:
            return None

        # Reconstruct
        path: List[Position] = []
        cur: Optional[Position] = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path