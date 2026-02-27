from __future__ import annotations

from collections import deque
from typing import Dict, Tuple, List, Optional

from maze.grid import Maze, Position
from solvers.base import SolveResult
from metrics import timer_ms
from heuristics import manhattan

Action = Tuple[int, int]


class PolicyIterationSolver:
    """
    MDP Policy Iteration on a grid maze.

    Transition model:
      - deterministic if p_intended = 1.0
      - stochastic slip model otherwise

    Reward:
      0 at goal, -1 per step otherwise
    """

    name = "mdp_pi"

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
        actions: List[Action] = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        V: Dict[Position, float] = {s: 0.0 for s in states}
        policy: Dict[Position, Action] = {s: actions[0] for s in states if s != maze.goal}

        with timer_ms() as elapsed:
            iterations = 0
            while True:
                iterations += 1

                # Policy evaluation
                while True:
                    delta = 0.0
                    for s in states:
                        if s == maze.goal:
                            continue
                        old = V[s]
                        a = policy[s]
                        V[s] = self._q_value(maze, V, s, a)
                        delta = max(delta, abs(old - V[s]))
                    if delta < self.theta:
                        break

                stable = True
                for s in states:
                    if s == maze.goal:
                        continue
                    old_a = policy[s]
                    best_a = max(actions, key=lambda a: self._q_value(maze, V, s, a))
                    policy[s] = best_a
                    if best_a != old_a:
                        stable = False

                if stable:
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
        if action in [(-1, 0), (1, 0)]:
            return (0, -1), (0, 1)
        return (-1, 0), (1, 0)

    def _outcomes(self, maze: Maze, state: Position, intended: Action) -> List[Tuple[float, Position]]:
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