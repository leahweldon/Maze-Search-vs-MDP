from __future__ import annotations

import random
from collections import deque
from typing import List, Optional, Tuple

from .grid import Maze, Position


def generate_maze(
    width: int,
    height: int,
    maze_type: str,
    seed: Optional[int] = None,
    wall_density: float = 0.25,
    ensure_solvable: bool = False,
    max_tries: int = 50,
    start: Optional[Position] = None,
    goal: Optional[Position] = None,
) -> Maze:
    """
    Generate Maze instance.

    Two maze familiessupported:
      - "perfect": a perfect maze carved using DFS backtracking (one unique simple path between any two open cells).
                  This generator works on odd dimensions; if even dimensions are provided, they are increased by 1.
      - "random": a grid with random walls sampled by wall_density (0..1).

    If ensure_solvable=True, the generator will retry (up to max_tries) until a solvable maze is produced.
    This is useful for fair algorithm comparison on larger mazes, where random walls can frequently disconnect start/goal.
    """
    if width < 3 or height < 3:
        raise ValueError("Maze dimensions must be at least 3x3")

    maze_type = maze_type.lower().strip()
    rng = random.Random(seed)

    if ensure_solvable and max_tries < 1:
        raise ValueError("max_tries must be >= 1 when ensure_solvable is enabled")

    last_error: Optional[Exception] = None

    for _ in range(max_tries if ensure_solvable else 1):
        try:
            if maze_type == "perfect":
                w, h = _make_odd(width), _make_odd(height)
                s = start if start is not None else (1, 1)
                g = goal if goal is not None else (h - 2, w - 2)
                grid = _generate_perfect_grid(w, h, rng)
                grid = _force_free(grid, s, g)
                maze = Maze(grid=grid, start=s, goal=g)

            elif maze_type == "random":
                s = start if start is not None else (0, 0)
                g = goal if goal is not None else (height - 1, width - 1)
                grid = _generate_random_grid(width, height, rng, wall_density=wall_density)
                grid = _force_free(grid, s, g)
                maze = Maze(grid=grid, start=s, goal=g)

            else:
                raise ValueError("maze_type must be one of: perfect, random")

            if ensure_solvable and not _is_reachable(maze):
                continue

            return maze

        except Exception as e:
            last_error = e
            continue

    if ensure_solvable:
        raise RuntimeError(
            "Failed to generate a solvable maze after "
            f"{max_tries} attempt(s). Last error: {last_error}"
        )

    raise RuntimeError(f"Failed to generate maze. Last error: {last_error}")


def _make_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def _force_free(grid: List[List[int]], start: Position, goal: Position) -> List[List[int]]:
    """
    Ensure the start and goal cells are traversable.
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0

    for (r, c) in (start, goal):
        if not (0 <= r < h and 0 <= c < w):
            raise ValueError("Start/goal position out of bounds for generated grid")
        grid[r][c] = 0

    return grid


def _generate_random_grid(width: int, height: int, rng: random.Random, wall_density: float) -> List[List[int]]:
    """
    Random obstacle maze: each cell becomes a wall with probability wall_density.
    """
    if not (0.0 <= wall_density <= 1.0):
        raise ValueError("wall_density must be between 0 and 1")

    grid: List[List[int]] = []
    for _ in range(height):
        row = [1 if rng.random() < wall_density else 0 for _ in range(width)]
        grid.append(row)

    return grid


def _generate_perfect_grid(width: int, height: int, rng: random.Random) -> List[List[int]]:
    """
    Perfect maze generator using DFS backtracking carving.

    The grid is initialized as all walls (1), then passages (0) are carved.
    Works best on odd dimensions.

    Representation:
      - cells are on odd coordinates (1,1), (1,3), ...
      - walls between cells are removed to create corridors
    """
    grid = [[1 for _ in range(width)] for _ in range(height)]

    start_cell = (1, 1)
    grid[start_cell[0]][start_cell[1]] = 0

    stack: List[Tuple[int, int]] = [start_cell]
    visited = {start_cell}

    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    while stack:
        r, c = stack[-1]

        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 1 <= nr < height - 1 and 1 <= nc < width - 1 and (nr, nc) not in visited:
                neighbors.append((nr, nc, dr, dc))

        if not neighbors:
            stack.pop()
            continue

        nr, nc, dr, dc = rng.choice(neighbors)

        # Carve wall between current cell and next cell
        wall_r, wall_c = r + dr // 2, c + dc // 2
        grid[wall_r][wall_c] = 0
        grid[nr][nc] = 0

        visited.add((nr, nc))
        stack.append((nr, nc))

    return grid


def _is_reachable(maze: Maze) -> bool:
    """
    Simple BFS reachability check used by the generator when ensure_solvable=True.

    This is intentionally implemented locally to avoid coupling the generator to solver implementations.
    """
    q = deque([maze.start])
    seen = {maze.start}

    while q:
        cur = q.popleft()
        if cur == maze.goal:
            return True

        for nxt in maze.neighbors_4(cur):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)

    return False