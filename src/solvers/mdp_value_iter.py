from __future__ import annotations

from collections import deque
from typing import Dict, Tuple, List, Optional

from maze.grid import Maze, Position
from solvers.base import SolveResult
from metrics import timer_ms
from heuristics import manhattan

Action = Tuple[int, int]


class ValueIterationSolver:
    """
    MDP Value Iteration on a grid maze.

    - Deterministic transitions when p_intended = 1.0
    - Stochastic transitions otherwise:
        intended: p
        slip left: (1-p)/2
        slip right: (1-p)/2

    Reward:
        0 at goal
        -1 per step otherwise
    """

    name = "mdp_vi"

    def __init__(self, gamma: float = 0.95, theta: float = 1e-6, p_intended: float = 1.0):
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1].")
        if not (0.0 < theta):
            raise ValueError("theta must be > 0.")
        if not (0.0 < p_intended <= 1.0):
            raise ValueError("p_intended must be in (0, 1].")

        self.gamma = gamma
        self.theta = theta
        self.p_intended = p_intended

    def solve(self, maze: Maze) -> SolveResult:
        states = [
            (r, c)
            for r in range(maze.height)
            for c in range(maze.width)
            if maze.grid[r][c] == 0
        ]

        actions: List[Action] = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U, D, L, R
        V: Dict[Position, float] = {s: 0.0 for s in states}

        with timer_ms() as elapsed:
            iterations = 0
            while True:
                delta = 0.0
                iterations += 1

                for s in states:
                    if s == maze.goal:
                        continue

                    old = V[s]
                    V[s] = max(self._q_value(maze, V, s, a) for a in actions)
                    delta = max(delta, abs(old - V[s]))

                if delta < self.theta:
                    break

        path = self._extract_path_via_policy_graph(maze, V)

        return SolveResult(
            algorithm=self.name,
            success=path is not None,
            path=path,
            path_length=(len(path) - 1) if path else None,
            runtime_ms=elapsed(),
            nodes_expanded=len(states),
            frontier_max=0,
            meta={
                "gamma": self.gamma,
                "theta": self.theta,
                "p_intended": self.p_intended,
                "iterations": iterations,
            },
        )

    # Transition / reward model
    def _transition(self, maze: Maze, state: Position, action: Action) -> Position:
        r, c = state
        dr, dc = action
        nxt = (r + dr, c + dc)
        if maze.in_bounds(nxt) and maze.is_free(nxt):
            return nxt
        return state

    def _reward(self, maze: Maze, next_state: Position) -> float:
        return 0.0 if next_state == maze.goal else -1.0

    def _perpendicular(self, action: Action) -> Tuple[Action, Action]:
        # For U/D, left/right are L/R. For L/R, left/right are U/D.
        if action in [(-1, 0), (1, 0)]:  # vertical
            return (0, -1), (0, 1)
        return (-1, 0), (1, 0)

    def _outcomes(self, maze: Maze, state: Position, intended: Action) -> List[Tuple[float, Position]]:
        """
        Return list of (probability, next_state) outcomes for intended action,
        using slip dynamics when p_intended < 1.0.
        """
        if self.p_intended >= 1.0:
            return [(1.0, self._transition(maze, state, intended))]

        left, right = self._perpendicular(intended)

        p = self.p_intended
        slip = (1.0 - p) / 2.0

        return [
            (p, self._transition(maze, state, intended)),
            (slip, self._transition(maze, state, left)),
            (slip, self._transition(maze, state, right)),
        ]

    def _q_value(self, maze: Maze, V: Dict[Position, float], s: Position, a: Action) -> float:
        """
        Q(s,a) = Σ_{s'} P(s'|s,a) [ R(s,a,s') + gamma * V(s') ]
        """
        total = 0.0
        for prob, nxt in self._outcomes(maze, s, a):
            r = self._reward(maze, nxt)
            total += prob * (r + self.gamma * V[nxt])
        return total

    # Robust extraction
    def _best_successors(self, maze: Maze, V: Dict[Position, float], s: Position) -> List[Position]:
        if s == maze.goal:
            return []

        actions: List[Action] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        candidates: List[Tuple[float, float, Position]] = []

        for a in actions:
            q = self._q_value(maze, V, s, a)
            # Use intended next state just for tie-break toward goal
            nxt_intended = self._transition(maze, s, a)
            candidates.append((q, -manhattan(nxt_intended, maze.goal), nxt_intended))

        candidates.sort(reverse=True)
        best_q = candidates[0][0]
        eps = 1e-12
        best = [nxt for (q, _, nxt) in candidates if abs(q - best_q) <= eps]

        seen = set()
        out = []
        for nxt in best:
            if nxt not in seen:
                out.append(nxt)
                seen.add(nxt)
        return out

    def _extract_path_via_policy_graph(self, maze: Maze, V: Dict[Position, float]) -> Optional[List[Position]]:
        """
        Build greedy-optimal policy graph (ties allowed) then BFS from start to goal.
        """
        start = maze.start
        goal = maze.goal

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

        path: List[Position] = []
        cur: Optional[Position] = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path