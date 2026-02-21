import json
from .grid import Maze


def save_maze_json(path: str, maze: Maze):
    data = {
        "grid": maze.grid,
        "start": maze.start,
        "goal": maze.goal,
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load_maze_json(path: str) -> Maze:
    with open(path, "r") as f:
        data = json.load(f)

    return Maze(
        grid=data["grid"],
        start=tuple(data["start"]),
        goal=tuple(data["goal"]),
    )