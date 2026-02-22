from __future__ import annotations

from typing import Dict, Tuple, List

from maze.grid import Maze, Position
from solvers.base import SolveResult
from metrics import timer_ms

Action = Tuple[int, int]


class PolicyIterationSolver:
    name = "mdp_pi"

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
        policy: Dict[Position, Action] = {
            s: actions[0] for s in states if s != maze.goal
        }

        with timer_ms() as elapsed:
            iterations = 0

            while True:
                iterations += 1

                # Policy Evaluation
                while True:
                    delta = 0.0
                    for s in states:
                        if s == maze.goal:
                            continue

                        a = policy[s]
                        nxt = self._transition(maze, s, a)
                        reward = self._reward(maze, nxt)
                        new_v = reward + self.gamma * V[nxt]
                        delta = max(delta, abs(new_v - V[s]))
                        V[s] = new_v

                    if delta < self.theta:
                        break

                # Policy Improvement
                stable = True
                for s in states:
                    if s == maze.goal:
                        continue

                    old_action = policy[s]
                    best_action = None
                    best_value = float("-inf")

                    for a in actions:
                        nxt = self._transition(maze, s, a)
                        reward = self._reward(maze, nxt)
                        value = reward + self.gamma * V[nxt]

                        if value > best_value:
                            best_value = value
                            best_action = a

                    policy[s] = best_action
                    if best_action != old_action:
                        stable = False

                if stable:
                    break

        path = self._extract_policy_path(maze, policy)

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

    def _extract_policy_path(self, maze: Maze, policy: Dict[Position, Action]):
        current = maze.start
        path = [current]
        visited = {current}

        while current != maze.goal:
            if current not in policy:
                return None

            action = policy[current]
            current = self._transition(maze, current, action)

            if current in visited:
                return None

            path.append(current)
            visited.add(current)

        return path