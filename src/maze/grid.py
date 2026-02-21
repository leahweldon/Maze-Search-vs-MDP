from dataclasses import dataclass
from typing import List, Tuple

Position = Tuple[int, int]


@dataclass
class Maze:
    grid: List[List[int]]  # 0 = free, 1 = wall
    start: Position
    goal: Position

    def __post_init__(self):
        self.height = len(self.grid)
        self.width = len(self.grid[0]) if self.height > 0 else 0
        self.validate()

    def in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def is_free(self, pos: Position) -> bool:
        r, c = pos
        return self.grid[r][c] == 0

    def neighbors_4(self, pos: Position) -> List[Position]:
        r, c = pos
        candidates = [
            (r - 1, c),  # up
            (r + 1, c),  # down
            (r, c - 1),  # left
            (r, c + 1),  # right
        ]
        return [
            p for p in candidates
            if self.in_bounds(p) and self.is_free(p)
        ]

    def validate(self):
        if not self.in_bounds(self.start):
            raise ValueError("Start position out of bounds")
        if not self.in_bounds(self.goal):
            raise ValueError("Goal position out of bounds")
        if not self.is_free(self.start):
            raise ValueError("Start position is a wall")
        if not self.is_free(self.goal):
            raise ValueError("Goal position is a wall")