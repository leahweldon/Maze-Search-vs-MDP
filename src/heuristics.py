from __future__ import annotations

import math
from typing import Tuple

Position = Tuple[int, int]


def manhattan(a: Position, b: Position) -> float:
    """
    Manhattan distance heuristic.

    Admissible for 4-direction grid movement with uniform step cost.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: Position, b: Position) -> float:
    """
    Euclidean distance heuristic.

    Also admissible for 4-direction movement (never overestimates
    true shortest path cost).
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)