from typing import List, Tuple
from .grid import Maze

Position = Tuple[int, int]


def render_ascii(maze: Maze, path: List[Position] | None = None) -> str:
    path_set = set(path) if path else set()
    lines = []

    for r in range(maze.height):
        row_chars = []
        for c in range(maze.width):
            pos = (r, c)

            if pos == maze.start:
                row_chars.append("S")
            elif pos == maze.goal:
                row_chars.append("G")
            elif pos in path_set:
                row_chars.append("*")
            elif maze.grid[r][c] == 1:
                row_chars.append("#")
            else:
                row_chars.append(".")

        lines.append("".join(row_chars))

    return "\n".join(lines)