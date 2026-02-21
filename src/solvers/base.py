from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

from maze.grid import Maze, Position


@dataclass
class SolveResult:
    algorithm: str
    success: bool
    path: Optional[List[Position]]
    path_length: Optional[int]

    runtime_ms: float
    nodes_expanded: int
    frontier_max: int

    meta: dict


class Solver(Protocol):
    name: str

    def solve(self, maze: Maze) -> SolveResult:
        ...